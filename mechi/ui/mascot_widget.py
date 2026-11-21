"""Tam otomatik, manuel kontrolsuz Tekir & Bombay maskot sahnesi."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from mechi.core.mascot_engine import MascotClip, MascotEngine


class MascotStageWidget(QFrame):
    """#262828 studyo zeminiyle cercevesiz eriyen otomatik kedi sahnesi."""

    def __init__(
        self,
        engine: MascotEngine,
        pause_between_sec: float = 2.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MascotStage")
        self.engine = engine
        self.pause_ms = max(300, int(pause_between_sec * 1000))
        self._state = "home"  # home | busy | success
        self._compact = False
        self._enabled = True
        self._movie: Optional[QMovie] = None
        self._waiting_for_next = False

        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.timeout.connect(self._play_next_home)

        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(0)

        self.lbl_canvas = QLabel()
        self.lbl_canvas.setObjectName("MascotCanvas")
        self.lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_canvas.setScaledContents(True)
        root.addWidget(self.lbl_canvas, 1, Qt.AlignmentFlag.AlignCenter)

        self.set_compact_mode(False)

    def set_compact_mode(self, compact: bool) -> None:
        self._compact = compact
        if compact:
            self.setFixedHeight(98)
            self.lbl_canvas.setFixedSize(164, 92)
        else:
            self.setFixedHeight(248)
            self.lbl_canvas.setFixedSize(426, 240)
        self._rescale_movie()

    def set_enabled(self, enabled: bool, pause_sec: float = 2.0) -> None:
        self._enabled = enabled
        self.pause_ms = max(300, int(pause_sec * 1000))
        self.setVisible(enabled)
        if not enabled:
            self._stop_movie()
            self._pause_timer.stop()
        elif self._movie is None:
            self.play_event("opening")

    def start_opening(self) -> None:
        if not self._enabled:
            return
        self.play_event("opening")

    def play_event(self, event: str) -> None:
        if not self._enabled or not self.engine.has_clips():
            return
        self._pause_timer.stop()
        if event == "busy":
            self._state = "busy"
        elif event == "success":
            self._state = "success"
        else:
            self._state = "home"
        clip = self.engine.pick_clip(event)
        if clip:
            self._start_clip(clip)

    def _stop_movie(self) -> None:
        if self._movie is not None:
            try:
                self._movie.frameChanged.disconnect(self._on_frame_changed)
            except Exception:
                pass
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None
        self.lbl_canvas.setMovie(None)

    def _target_movie_size(self) -> QSize:
        return self.lbl_canvas.size()

    def _rescale_movie(self) -> None:
        if self._movie is not None:
            self._movie.setScaledSize(self._target_movie_size())
            if self._movie.state() == QMovie.MovieState.Paused:
                cur = self._movie.currentFrameNumber()
                if cur >= 0:
                    self._movie.jumpToFrame(cur)

    def _start_clip(self, clip: MascotClip) -> None:
        self._stop_movie()
        self._waiting_for_next = False

        movie = QMovie(clip.path)
        if not movie.isValid() and clip.fallback_gif:
            movie = QMovie(clip.fallback_gif)
        if not movie.isValid():
            return

        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        movie.setScaledSize(self._target_movie_size())
        movie.frameChanged.connect(self._on_frame_changed)
        self._movie = movie
        self.lbl_canvas.setMovie(movie)
        movie.start()

    def _on_frame_changed(self, frame_no: int) -> None:
        if self._movie is None or self._waiting_for_next:
            return
        total = self._movie.frameCount()
        if total > 1 and frame_no >= total - 1:
            if self._state == "busy":
                # Islem surerken kuyruk takibi kesintisiz oynar
                return
            self._waiting_for_next = True
            self._movie.stop()
            self.lbl_canvas.clear()
            self._pause_timer.start(self.pause_ms)

    def _play_next_home(self) -> None:
        if not self._enabled:
            return
        self._state = "home"
        clip = self.engine.pick_clip("home")
        if clip:
            self._start_clip(clip)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_movie()
