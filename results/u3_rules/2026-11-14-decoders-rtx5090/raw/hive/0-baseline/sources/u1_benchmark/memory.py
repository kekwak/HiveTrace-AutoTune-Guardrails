"""Per-process GPU samples only in dedicated memory runs."""
import os
import subprocess
import threading
import time

from .core import write


def process_mib(text, pid, uuid):
    values = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid and parts[1] == str(pid):
            try:
                values.append(float(parts[2]))
            except ValueError:
                pass
    return sum(values) if values else None


class Sampler:
    def __init__(self, gpu_uuid, interval=.05):
        self.uuid, self.interval = gpu_uuid, interval
        self.phase = "setup"
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.boundaries = {"setup_start": time.monotonic()}

    def sample(self):
        phase, start = self.phase, time.monotonic()
        try:
            output = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    check=True, timeout=2).stdout
            value = process_mib(output, os.getpid(), self.uuid)
            import psutil
            rss = psutil.Process().memory_info().rss
            if phase == self.phase:
                self.rows.append({"phase": phase, "started": start, "finished": time.monotonic(),
                                  "gpu_process_mib": value, "host_rss_bytes": rss})
        except Exception as e:
            self.errors.append({"phase": phase, "error": str(e)})

    def loop(self):
        while not self.stop_event.is_set():
            start = time.monotonic()
            self.sample()
            self.stop_event.wait(max(0, self.interval - (time.monotonic() - start)))

    def start(self):
        self.thread.start()

    def inference(self):
        self.boundaries["inference_start"] = time.monotonic()
        self.phase = "inference"

    def stop(self, path):
        self.boundaries["inference_end" if self.phase == "inference" else "setup_end"] = time.monotonic()
        self.stop_event.set()
        self.thread.join(timeout=3)
        windows = {}
        for phase in ("setup", "inference"):
            rows = [r for r in self.rows if r["phase"] == phase]
            values = [r["gpu_process_mib"] for r in rows if r["gpu_process_mib"] is not None]
            gaps = [b["started"] - a["started"] for a, b in zip(rows, rows[1:])]
            windows[phase] = {"sampled_gpu_peak_mib": max(values) if values else None,
                              "host_rss_peak_bytes": max((r["host_rss_bytes"] for r in rows), default=None),
                              "samples": len(rows), "valid_gpu_samples": len(values),
                              "actual_intervals_seconds": gaps,
                              "status": "measured" if values else "unavailable"}
        result = {"pid": os.getpid(), "gpu_uuid": self.uuid, "requested_interval_seconds": self.interval,
                  "windows": windows, "boundaries": self.boundaries, "samples": self.rows,
                  "errors": self.errors, "sampling_may_miss_short_peaks": True}
        write(path, result)
        return result
