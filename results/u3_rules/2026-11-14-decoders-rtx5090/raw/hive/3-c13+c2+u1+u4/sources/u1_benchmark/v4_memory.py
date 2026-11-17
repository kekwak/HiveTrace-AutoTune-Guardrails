"""Receive-time process GPU memory sampling, with no CUDA imports.

One nvidia-smi process lives for the worker's sampling lifetime and inherits its
process group. Samples are observations, not CUDA allocator counters; short peaks
can be missed. No command output or exception text is persisted.
"""
import csv
import math
import os
import subprocess
import threading
import time

from .core import write

METHOD_ID = "nvidia-smi-stream-100ms-v2"
METHOD_VERSION = 2
INTERVAL_MS = 100
COMMAND = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
           "--format=csv,noheader,nounits", "--loop-ms=100")
_SHUTDOWN_TIMEOUT = 0.5


def _value(text):
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def process_mib(text, pid, uuid):
    """Parse a snapshot, rejecting zero/N/A/nonfinite matching observations."""
    values = []
    for parts in csv.reader(text.splitlines(), skipinitialspace=True):
        parts = [part.strip() for part in parts]
        if len(parts) == 3 and parts[:2] == [uuid, str(pid)]:
            value = _value(parts[2])
            if value is None:
                return None
            values.append(value)
    total = sum(values)
    return total if values and math.isfinite(total) else None


class Sampler:
    """Worker-local sampler with bounded termination and JSON-safe diagnostics.

    start()/inference()/stop(path) retain the old worker interface. The fixed
    100 ms requested interval is not a claim about actual receive intervals.
    A window is measured only with positive observations and no window errors;
    otherwise it is partial (some valid values) or unavailable (none).
    checkpoint(path) atomically persists in-progress evidence without stopping
    the reader. The worker must call it outside measured inference intervals.
    """

    def __init__(self, gpu_uuid, interval=0.1):
        if not isinstance(gpu_uuid, str) or not gpu_uuid:
            raise ValueError("A GPU UUID is required")
        if type(interval) not in (int, float) or interval != INTERVAL_MS / 1000:
            raise ValueError("The v4 sampler interval is fixed at 100 ms")
        self.uuid, self.interval, self.pid = gpu_uuid, interval, os.getpid()
        self.phase = "setup"
        self.rows, self.errors, self.boundaries = [], [], {}
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.process = self.thread = None
        self._started = self._stopped = False
        self._result = None

    def _error(self, code, received=None):
        received = time.monotonic() if received is None else received
        with self._lock:
            boundary = self.boundaries.get("inference_start")
            phase = "inference" if boundary is not None and received >= boundary else "setup"
            self.errors.append({"phase": phase, "received": received,
                                "error": code})

    def start(self):
        if self._started or self._stopped:
            raise RuntimeError("Sampler can only be started once")
        self._started = True
        self.boundaries["setup_start"] = time.monotonic()
        try:
            # Do not create a new session/group: parent worker-group kills must
            # also reach this child. Merging stderr avoids a second blocked pipe.
            self.process = subprocess.Popen(list(COMMAND), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            encoding="utf-8", errors="replace", bufsize=1)
        except (OSError, subprocess.SubprocessError):
            self._error("spawn_failed")
            return
        self.thread = threading.Thread(target=self._read, name="v4-memory", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop_event.is_set():
                line = self.process.stdout.readline()
                received = time.monotonic()
                if self.stop_event.is_set():
                    break
                if not line:
                    self._error("unexpected_stream_end", received)
                    break
                self._receive(line, received)
        except Exception:
            if not self.stop_event.is_set():
                self._error("stream_read_failed")

    def _receive(self, line, received):
        if not line.strip():
            return
        try:
            parts = next(csv.reader([line], skipinitialspace=True))
            parts = [part.strip() for part in parts]
        except (csv.Error, StopIteration):
            self._error("malformed_output", received)
            return
        if len(parts) != 3 or not parts[1].isdigit():
            self._error("malformed_output", received)
            return
        if parts[:2] != [self.uuid, str(self.pid)]:
            return
        value = _value(parts[2])
        with self._lock:
            if self.stop_event.is_set():
                return
            # Resolve phase by receipt time, even if inference() acquired the
            # lock between readline() and this append.
            boundary = self.boundaries.get("inference_start")
            phase = "inference" if boundary is not None and received >= boundary else "setup"
            self.rows.append({"phase": phase, "received": received,
                              "started": received, "finished": received,
                              "gpu_process_mib": value, "host_rss_bytes": None})
            if value is None:
                self.errors.append({"phase": phase, "received": received,
                                    "error": "unavailable_gpu_value"})

    def inference(self):
        with self._lock:
            if not self._started or self._stopped or self.phase == "inference":
                raise RuntimeError("Invalid inference boundary")
            now = time.monotonic()
            self.boundaries.update(setup_end=now, inference_start=now)
            self.phase = "inference"

    def _shutdown(self):
        if self.process is not None:
            try:
                code = self.process.poll()
            except (OSError, subprocess.SubprocessError):
                self._error("process_status_failed")
                code = None
            if code is not None and code != 0:
                self._error("process_failed")
            if code is None:
                try:
                    self.process.terminate()
                except (OSError, subprocess.SubprocessError):
                    self._error("terminate_failed")
                try:
                    self.process.wait(timeout=_SHUTDOWN_TIMEOUT)
                except (OSError, subprocess.SubprocessError):
                    self._error("terminate_timeout")
                    try:
                        self.process.kill()
                    except (OSError, subprocess.SubprocessError):
                        self._error("kill_failed")
                    try:
                        self.process.wait(timeout=_SHUTDOWN_TIMEOUT)
                    except (OSError, subprocess.SubprocessError):
                        self._error("reap_failed")
        if self.thread is not None:
            self.thread.join(timeout=_SHUTDOWN_TIMEOUT)
            if self.thread.is_alive():
                self._error("reader_join_timeout")
        # Closing a buffered pipe while another thread holds its read lock can
        # block forever. Leave that rare failure to worker-process cleanup.
        if self.process is not None and (self.thread is None or not self.thread.is_alive()):
            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except OSError:
                    self._error("stream_close_failed")

    def _snapshot(self, in_progress=False):
        with self._lock:
            rows = [dict(row) for row in self.rows]
            errors = [dict(error) for error in self.errors]
            boundaries = dict(self.boundaries)
            captured = time.monotonic()
        windows = {}
        for phase in ("setup", "inference"):
            samples = [row for row in rows if row["phase"] == phase]
            values = [row["gpu_process_mib"] for row in samples if row["gpu_process_mib"] is not None]
            bad = any(error["phase"] == phase for error in errors)
            status = "unavailable" if not values else "partial" if bad or in_progress else "measured"
            windows[phase] = {"sampled_gpu_peak_mib": max(values) if values else None,
                              "host_rss_peak_bytes": None,
                              "samples": len(samples), "valid_gpu_samples": len(values),
                              "actual_intervals_seconds": [b["received"] - a["received"]
                                                           for a, b in zip(samples, samples[1:])],
                              "status": status}
        statuses = [window["status"] for window in windows.values()]
        status = ("measured" if all(s == "measured" for s in statuses) else
                  "unavailable" if all(s == "unavailable" for s in statuses) else "partial")
        result = {"pid": self.pid, "gpu_uuid": self.uuid,
                  "method_id": METHOD_ID, "method_version": METHOD_VERSION,
                  "method": {"id": METHOD_ID, "version": METHOD_VERSION,
                             "interval_ms": INTERVAL_MS, "timestamp_source": "monotonic_receive"},
                  "requested_interval_seconds": self.interval, "status": status,
                  "windows": windows, "boundaries": boundaries, "samples": rows,
                  "errors": errors, "sampling_may_miss_short_peaks": True}
        if in_progress:
            result.update(status="partial", checkpoint=True, checkpoint_monotonic=captured)
        return result

    def checkpoint(self, path):
        """Atomically save current raw samples; do not close windows or stop sampling.

        Call from the worker outside request timers. I/O happens after releasing
        the sample lock. A completed stop result is never downgraded to partial.
        """
        if not self._started:
            raise RuntimeError("Sampler has not been started")
        if self._result is not None:
            result = self._result
        elif self._stopped:
            raise RuntimeError("Sampler is stopping")
        else:
            result = self._snapshot(in_progress=True)
        write(path, result)
        return result

    def stop(self, path):
        if not self._started:
            raise RuntimeError("Sampler has not been started")
        if self._result is not None:
            write(path, self._result)
            return self._result
        with self._lock:
            self._stopped = True
            self.boundaries[self.phase + "_end"] = time.monotonic()
            self.stop_event.set()
        self._shutdown()
        result = self._snapshot()
        self._result = result
        write(path, result)
        return result


# Worker integration name; both names expose the same lifecycle and metadata.
ProcessMemory = Sampler
