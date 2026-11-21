"""Arka plan is parcacigi ve alt surec yonetimi."""
from __future__ import annotations

import subprocess
import threading
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal


class TaskCancelled(Exception):
    """Kullanici isi iptal ettiginde firlatilir."""


class TaskWorker(QThread):
    """GUI ve Photoshop'u kilitlemeden uzun isleri arka planda calistirir."""

    progress = Signal(int, str)
    log_line = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.cancel_event = threading.Event()
        self._proc_lock = threading.Lock()
        self._active_proc: Optional[subprocess.Popen[Any]] = None

    def attach_process(self, proc: Optional[subprocess.Popen[Any]]) -> None:
        with self._proc_lock:
            self._active_proc = proc

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._proc_lock:
            proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _emit_progress(self, pct: int, message: str) -> None:
        self.progress.emit(max(0, min(100, int(pct))), str(message))

    def _emit_log(self, line: str) -> None:
        self.log_line.emit(str(line))

    def run(self) -> None:
        try:
            res = self._fn(
                *self._args,
                progress_cb=self._emit_progress,
                log_cb=self._emit_log,
                cancel_event=self.cancel_event,
                attach_process=self.attach_process,
                **self._kwargs,
            )
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            self.succeeded.emit(res)
        except TaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            tb = traceback.format_exc(limit=4)
            self.failed.emit(f"{exc}\n\n{tb}")
