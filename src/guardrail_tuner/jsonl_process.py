"""Single-flight persistent JSONL subprocess with bounded request time and cleanup."""

import json
import os
import queue
import signal
import subprocess
import threading
from typing import Any


class PersistentJSONLRunner:
    def __init__(
        self, command: list[str], timeout: float, *, env: dict[str, str] | None = None
    ) -> None:
        self.timeout = timeout
        self.closed = False
        self.requests: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self.responses: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise ValueError(f"Runner failed: {type(exc).__name__}") from exc
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        while True:
            request = self.requests.get()
            if request is None:
                return
            try:
                self.process.stdin.write(request)
                self.process.stdin.flush()
                line = self.process.stdout.readline(4 * 1024 * 1024 + 1)
                if not line:
                    self.responses.put(("error", "Runner closed stdout before responding"))
                    return
                if len(line) > 4 * 1024 * 1024 or not line.endswith("\n"):
                    self.responses.put(
                        ("error", "Runner response exceeds line limit or lacks newline")
                    )
                    return
                self.responses.put(("ok", line))
            except (OSError, ValueError):
                self.responses.put(("error", "Runner pipe failed"))
                return

    def run(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise ValueError(f"Runner exited with status {self.process.returncode}")
        try:
            self.requests.put_nowait(json.dumps(row, allow_nan=False) + "\n")
            status, line = self.responses.get(timeout=self.timeout)
        except queue.Full as exc:
            raise ValueError("Runner already has a pending request") from exc
        except queue.Empty as exc:
            self.close()
            raise ValueError("Runner failed: TimeoutExpired") from exc
        if status != "ok":
            raise ValueError(str(line))
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Runner returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Runner response must be a JSON object")
        return value

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self._stop(force=False)
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._stop(force=True)
                self.process.wait()
        if os.name == "posix":
            self._stop(force=True)
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        # Closing stdin can flush buffered bytes into an exited child.
                        pass

    def _stop(self, *, force: bool) -> None:
        try:
            if os.name == "posix":
                os.killpg(self.process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                self.process.kill()
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass

    def __enter__(self) -> "PersistentJSONLRunner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
