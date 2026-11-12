"""CPU-only sampling tests: fake Popen streams, plus temporary atomic-write checks."""
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from u1_benchmark import v4_memory as m
from u1_benchmark.core import write as atomic_write


class FakeStream:
    def __init__(self):
        self.queue = queue.Queue()
        self.closed = False

    def readline(self):
        value = self.queue.get(timeout=5)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self):
        self.stdout = FakeStream()
        self.returncode = None
        self.ignore_terminate = self.ignore_kill = False
        self.terminated = self.killed = 0
        self.waits = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.ignore_terminate:
            self.returncode = -15
            self.stdout.queue.put("")

    def kill(self):
        self.killed += 1
        if not self.ignore_kill:
            self.returncode = -9
            self.stdout.queue.put("")

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class ParserTests(unittest.TestCase):
    def test_exact_pid_uuid_and_positive_finite_values(self):
        self.assertEqual(m.process_mib("GPU-a, 123, 4\nGPU-b, 123, 999\nGPU-a, 12, 999\nGPU-a, 123, 6", 123, "GPU-a"), 10)
        self.assertIsNone(m.process_mib("GPU-a, 12, 999", 123, "GPU-a"))
        for value in ("N/A", "[N/A]", "0", "-1", "nan", "inf", "-inf", "1e309", "PRIVATE"):
            with self.subTest(value=value):
                self.assertIsNone(m.process_mib(f"GPU-a, 123, {value}", 123, "GPU-a"))
                self.assertIsNone(m.process_mib(f"GPU-a, 123, 12\nGPU-a, 123, {value}", 123, "GPU-a"))


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.process = FakeProcess()
        popen = patch.object(m.subprocess, "Popen", return_value=self.process)
        archive = patch.object(m, "write")
        self.popen, self.archive = popen.start(), archive.start()
        self.addCleanup(popen.stop)
        self.addCleanup(archive.stop)
        self.sampler = m.Sampler("GPU-a")
        self.addCleanup(self.cleanup_sampler)

    def cleanup_sampler(self):
        self.process.ignore_terminate = self.process.ignore_kill = False
        self.process.stdout.queue.put("")
        if self.sampler._started:
            self.sampler.stop("fake.json")
        if self.sampler.thread is not None:
            self.sampler.thread.join(timeout=1)

    def send(self, value, pid=None, uuid="GPU-a"):
        pid = self.sampler.pid if pid is None else pid
        self.process.stdout.queue.put(f"{uuid}, {pid}, {value}\n")

    def until(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.sampler._lock:
                if predicate():
                    return
            time.sleep(0.001)
        self.fail("Fake reader did not reach expected state")

    def stop(self):
        value = self.sampler.stop("fake.json")
        self.archive.assert_called_with("fake.json", value)
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        return value

    def test_single_popen_inherited_process_group_and_old_windows(self):
        self.sampler.start()
        self.send(10)
        self.send(12)
        self.until(lambda: len(self.sampler.rows) == 2)
        self.sampler.inference()
        self.send(20)
        self.send(24)
        self.until(lambda: len(self.sampler.rows) == 4)
        result = self.stop()
        self.popen.assert_called_once_with(list(m.COMMAND), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace", bufsize=1)
        self.assertEqual(m.COMMAND[-1], "--loop-ms=100")
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["method_id"], "nvidia-smi-stream-100ms-v2")
        self.assertEqual(result["method_version"], 2)
        self.assertEqual(result["method"]["interval_ms"], 100)
        self.assertEqual(result["requested_interval_seconds"], .1)
        self.assertEqual(result["windows"]["setup"]["sampled_gpu_peak_mib"], 12)
        self.assertEqual(result["windows"]["inference"]["sampled_gpu_peak_mib"], 24)
        self.assertEqual(result["errors"], [])
        self.assertEqual([r["phase"] for r in result["samples"]], ["setup"] * 2 + ["inference"] * 2)
        received = [r["received"] for r in result["samples"]]
        self.assertEqual(received, sorted(received))
        bounds = result["boundaries"]
        self.assertEqual(bounds["setup_end"], bounds["inference_start"])
        for row in result["samples"]:
            self.assertEqual(row["started"], row["finished"])
            self.assertGreaterEqual(row["received"], bounds[row["phase"] + "_start"])
            self.assertLessEqual(row["received"], bounds[row["phase"] + "_end"])
        for window in result["windows"].values():
            self.assertEqual(window["valid_gpu_samples"], 2)
            self.assertEqual(window["status"], "measured")
            self.assertGreaterEqual(window["actual_intervals_seconds"][0], 0)
        self.assertFalse(self.sampler.thread.is_alive())
        self.assertTrue(self.process.stdout.closed)
        self.assertEqual((self.process.terminated, self.process.killed), (1, 0))

    def test_checkpoint_retains_raw_windows_without_stopping_and_stop_is_final(self):
        with self.assertRaises(RuntimeError):
            self.sampler.checkpoint("fake.json")
        self.sampler.start()
        self.send(10)
        self.until(lambda: len(self.sampler.rows) == 1)
        self.archive.assert_not_called()
        setup = self.sampler.checkpoint("fake.json")
        self.archive.assert_called_once_with("fake.json", setup)
        self.assertEqual(setup["status"], "partial")
        self.assertTrue(setup["checkpoint"])
        self.assertEqual(setup["windows"]["setup"]["status"], "partial")
        self.assertEqual(setup["windows"]["inference"]["status"], "unavailable")
        self.assertEqual(setup["samples"][0]["gpu_process_mib"], 10)
        self.assertGreaterEqual(setup["checkpoint_monotonic"], setup["samples"][0]["received"])
        self.assertNotIn("setup_end", setup["boundaries"])
        self.assertIsNone(self.sampler._result)
        self.assertFalse(self.sampler.stop_event.is_set())
        self.assertTrue(self.sampler.thread.is_alive())
        self.assertEqual((self.process.terminated, self.process.killed, self.process.waits), (0, 0, []))
        self.sampler.inference()
        self.send(20)
        self.until(lambda: len(self.sampler.rows) == 2)
        progress = self.sampler.checkpoint("fake.json")
        self.assertEqual([r["phase"] for r in progress["samples"]], ["setup", "inference"])
        self.assertEqual(progress["windows"]["inference"]["sampled_gpu_peak_mib"], 20)
        self.assertTrue(all(w["status"] == "partial" for w in progress["windows"].values()))
        self.assertNotIn("inference_end", progress["boundaries"])
        self.assertEqual(len(setup["samples"]), 1)
        self.assertNotIn("setup_end", setup["boundaries"])
        progress["samples"][0]["gpu_process_mib"] = 999
        final = self.stop()
        self.assertEqual(final["samples"][0]["gpu_process_mib"], 10)
        self.assertEqual(final["status"], "measured")
        self.assertNotIn("checkpoint", final)
        self.assertIn("inference_end", final["boundaries"])
        self.assertEqual(self.sampler.checkpoint("fake.json"), final)
        self.archive.assert_called_with("fake.json", final)
        self.assertEqual(self.process.terminated, 1)
        self.popen.assert_called_once()

    def test_checkpoint_io_does_not_hold_sample_lock_or_run_on_reader(self):
        self.sampler.start()
        self.send(10)
        self.until(lambda: len(self.sampler.rows) == 1)
        def persist(path, value):
            self.assertTrue(self.sampler._lock.acquire(blocking=False))
            self.sampler._lock.release()
            self.send(12)
            self.until(lambda: len(self.sampler.rows) == 2)
            self.assertEqual(len(value["samples"]), 1)
        with patch.object(m, "write", side_effect=persist) as saved:
            value = self.sampler.checkpoint("fake.json")
        saved.assert_called_once_with("fake.json", value)
        self.assertEqual(len(self.sampler.rows), 2)
        self.assertTrue(self.sampler.thread.is_alive())

    def test_checkpoint_with_no_valid_values_keeps_windows_unavailable(self):
        self.sampler.start()
        empty = self.sampler.checkpoint("fake.json")
        self.assertEqual(empty["status"], "partial")
        self.assertEqual(empty["samples"], [])
        self.send("N/A")
        self.send(0)
        self.until(lambda: len(self.sampler.rows) == 2)
        value = self.sampler.checkpoint("fake.json")
        self.assertEqual(value["status"], "partial")
        self.assertTrue(all(w["status"] == "unavailable" for w in value["windows"].values()))
        self.assertTrue(all(r["gpu_process_mib"] is None for r in value["samples"]))
        self.assertEqual(len(value["errors"]), 2)
        value["errors"][0]["error"] = "changed snapshot only"
        self.assertEqual(self.stop()["errors"][0]["error"], "unavailable_gpu_value")

    def test_checkpoint_atomic_file_survives_failed_replace_before_stop(self):
        self.sampler.start()
        self.send(10)
        self.until(lambda: len(self.sampler.rows) == 1)
        with tempfile.TemporaryDirectory(prefix="v4-memory-checkpoint-") as directory:
            path = Path(directory) / "memory.json"
            pending = path.with_suffix(".json.tmp")
            with patch.object(m, "write", side_effect=atomic_write):
                first = self.sampler.checkpoint(path)
                self.assertEqual(json.loads(path.read_text()), first)
                self.assertFalse(pending.exists())
                self.assertEqual(first["samples"][0]["gpu_process_mib"], 10)
                self.sampler.inference()
                self.send(20)
                self.until(lambda: len(self.sampler.rows) == 2)
                def fail_replace(source, target):
                    self.assertEqual(source, pending)
                    self.assertEqual(target, path)
                    self.assertEqual(json.loads(path.read_text()), first)
                    self.assertEqual(len(json.loads(source.read_text())["samples"]), 2)
                    raise OSError("Simulated interruption before atomic replacement")
                with patch.object(Path, "replace", autospec=True, side_effect=fail_replace):
                    with self.assertRaises(OSError):
                        self.sampler.checkpoint(path)
                self.assertEqual(json.loads(path.read_text()), first)
                self.assertFalse(self.sampler.stop_event.is_set())
                self.assertEqual(self.process.terminated, 0)
                latest = self.sampler.checkpoint(path)
                self.assertEqual(json.loads(path.read_text()), latest)
                self.assertEqual(latest["status"], "partial")
                self.assertFalse(pending.exists())
                final = self.sampler.stop(path)
                self.assertEqual(json.loads(path.read_text()), final)
                self.assertEqual(final["status"], "measured")
                self.assertEqual(len(final["samples"]), 2)

    def test_pid_uuid_filter_and_unavailable_values_never_measured(self):
        self.sampler.start()
        self.send(999, pid=self.sampler.pid + 1)
        self.send(999, uuid="GPU-other")
        for value in ("N/A", "0", "nan", "inf", "-1"):
            self.send(value)
        self.until(lambda: len(self.sampler.rows) == 5)
        result = self.stop()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["windows"]["setup"]["samples"], 5)
        self.assertTrue(all(r["gpu_process_mib"] is None for r in result["samples"]))
        self.assertTrue(all(w["sampled_gpu_peak_mib"] is None for w in result["windows"].values()))

    def test_partial_on_invalid_values_and_safe_diagnostics(self):
        self.sampler.start()
        self.send(12)
        self.send("N/A")
        self.process.stdout.queue.put("PRIVATE stderr /secret/model/path\n")
        self.until(lambda: len(self.sampler.rows) == 2 and len(self.sampler.errors) == 2)
        self.sampler.inference()
        self.send(16)
        self.until(lambda: len(self.sampler.rows) == 3)
        result = self.stop()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["windows"]["setup"]["status"], "partial")
        self.assertEqual(result["windows"]["inference"]["status"], "measured")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("/secret", json.dumps(result))
        self.assertEqual(set(result["samples"][0]), {"phase", "received", "started", "finished", "gpu_process_mib", "host_rss_bytes"})

    def test_no_samples_or_foreign_processes_are_unavailable(self):
        self.sampler.start()
        self.send(999, pid=self.sampler.pid + 1)
        self.sampler.inference()
        result = self.stop()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["samples"], [])

    def test_spawn_failure_and_exception_text_not_archived(self):
        self.popen.side_effect = FileNotFoundError("PRIVATE /secret")
        self.sampler.start()
        self.sampler.inference()
        result = self.stop()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["errors"][0]["error"], "spawn_failed")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertIsNone(self.sampler.thread)

    def test_unexpected_eof_and_failed_process_are_not_measured(self):
        self.sampler.start()
        self.send(12)
        self.until(lambda: len(self.sampler.rows) == 1)
        self.process.returncode = 1
        self.process.stdout.queue.put("")
        self.until(lambda: bool(self.sampler.errors))
        result = self.stop()
        self.assertEqual(result["windows"]["setup"]["status"], "partial")
        self.assertEqual({e["error"] for e in result["errors"]}, {"unexpected_stream_end", "process_failed"})

    def test_reader_failure(self):
        self.sampler.start()
        self.process.stdout.queue.put(OSError("PRIVATE broken pipe"))
        self.until(lambda: bool(self.sampler.errors))
        result = self.stop()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["errors"][0]["error"], "stream_read_failed")
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_terminate_timeout_kills_and_joins(self):
        self.process.ignore_terminate = True
        self.sampler.start()
        self.send(12)
        self.until(lambda: len(self.sampler.rows) == 1)
        start = time.monotonic()
        result = self.stop()
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual((self.process.terminated, self.process.killed), (1, 1))
        self.assertEqual(self.process.waits, [.5, .5])
        self.assertFalse(self.sampler.thread.is_alive())
        self.assertEqual(result["status"], "partial")

    def test_unreapable_process_and_reader_have_bounded_shutdown(self):
        self.process.ignore_terminate = self.process.ignore_kill = True
        self.sampler.start()
        start = time.monotonic()
        with patch.object(m, "_SHUTDOWN_TIMEOUT", .02):
            result = self.stop()
        self.assertLess(time.monotonic() - start, .5)
        self.assertFalse(self.process.stdout.closed)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual({e["error"] for e in result["errors"]},
                         {"terminate_timeout", "reap_failed", "reader_join_timeout"})
        self.process.stdout.queue.put("")
        self.sampler.thread.join(timeout=1)

    def test_receipt_timestamp_resolves_boundary_race(self):
        self.sampler.start()
        received = time.monotonic()
        self.sampler.inference()
        self.sampler._receive(f"GPU-a, {self.sampler.pid}, 12\n", received)
        self.assertEqual(self.sampler.rows[0]["phase"], "setup")
        self.sampler._receive("PRIVATE invalid diagnostic\n", received)
        self.assertEqual(self.sampler.errors[0]["phase"], "setup")
        self.assertEqual(self.sampler.errors[0]["received"], received)

    def test_poll_failure_still_terminates_child(self):
        self.sampler.start()
        with patch.object(self.process, "poll", side_effect=OSError("PRIVATE")):
            result = self.stop()
        self.assertEqual(self.process.terminated, 1)
        self.assertFalse(self.sampler.thread.is_alive())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["errors"][0]["error"], "process_status_failed")

    def test_terminate_failure_still_kills_child(self):
        self.sampler.start()
        with patch.object(self.process, "terminate", side_effect=OSError("PRIVATE")):
            result = self.stop()
        self.assertEqual(self.process.killed, 1)
        self.assertFalse(self.sampler.thread.is_alive())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual({e["error"] for e in result["errors"]}, {"terminate_failed", "terminate_timeout"})

    def test_lifecycle_and_fixed_interval(self):
        self.assertIs(m.ProcessMemory, m.Sampler)
        for interval in (True, "0.1", .05, 1):
            with self.assertRaises(ValueError):
                m.Sampler("GPU-a", interval)
        with self.assertRaises(RuntimeError):
            self.sampler.inference()
        with self.assertRaises(RuntimeError):
            self.sampler.stop("fake.json")
        self.sampler.start()
        with self.assertRaises(RuntimeError):
            self.sampler.start()
        self.sampler.inference()
        with self.assertRaises(RuntimeError):
            self.sampler.inference()
        result = self.stop()
        self.assertEqual(self.stop(), result)
        with self.assertRaises(RuntimeError):
            self.sampler.inference()
        self.popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
