"""MeChi Studio otomatik kedi maskot sahne secicisi."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from mechi.config import ASSETS_MASCOT_DIR


@dataclass(frozen=True)
class MascotClip:
    stem: str
    role: str
    path: str
    fallback_gif: Optional[str] = None


class MascotEngine:
    """mechi/assets/mascot klasorunu tarar; ana ekranda 01<->02 sirayla, islemde duruma gore otomatik secer."""

    def __init__(self, mascot_dir: str = ASSETS_MASCOT_DIR) -> None:
        self.mascot_dir = mascot_dir
        self._clips: List[MascotClip] = []
        self._home_idx: int = 0
        self.reload()

    def reload(self) -> None:
        self._clips.clear()
        self._home_idx = 0
        if not os.path.isdir(self.mascot_dir):
            return

        by_stem: Dict[str, Dict[str, str]] = {}
        for name in sorted(os.listdir(self.mascot_dir)):
            low = name.lower()
            if not (low.endswith(".webp") or low.endswith(".gif")):
                continue
            stem, ext = os.path.splitext(name)
            entry = by_stem.setdefault(stem, {})
            entry[ext.lower()] = os.path.join(self.mascot_dir, name)

        for stem in sorted(by_stem.keys()):
            files = by_stem[stem]
            primary = files.get(".webp") or files.get(".gif")
            if not primary:
                continue
            if stem.startswith("01_"):
                role = "opening"
            elif stem.startswith("04_"):
                role = "busy"
            elif stem.startswith("03_"):
                role = "success"
            else:
                role = "idle"
            self._clips.append(
                MascotClip(
                    stem=stem,
                    role=role,
                    path=primary,
                    fallback_gif=files.get(".gif"),
                )
            )

    @property
    def clips(self) -> List[MascotClip]:
        return list(self._clips)

    def has_clips(self) -> bool:
        return bool(self._clips)

    def _home_sequence(self) -> List[MascotClip]:
        # Ana ekranda sirayla: Tekir gobek gosterme (01) <-> Bombay cam koklama (02) + varsa diger bos sahneler
        seq = [c for c in self._clips if c.role in ("opening", "idle")]
        return seq or list(self._clips)

    def pick_home_sequence_next(self) -> Optional[MascotClip]:
        seq = self._home_sequence()
        if not seq:
            return None
        chosen = seq[self._home_idx % len(seq)]
        self._home_idx = (self._home_idx + 1) % len(seq)
        return chosen

    def pick_clip(self, event: str = "home") -> Optional[MascotClip]:
        if not self._clips:
            return None

        if event == "opening":
            self._home_idx = 0
            return self.pick_home_sequence_next()
        if event in ("home", "idle"):
            return self.pick_home_sequence_next()
        if event == "busy":
            busy = [c for c in self._clips if c.role == "busy"]
            return busy[0] if busy else self._clips[0]
        if event == "success":
            ok_clips = [c for c in self._clips if c.role == "success"]
            return ok_clips[0] if ok_clips else self._clips[0]

        return self.pick_home_sequence_next()
