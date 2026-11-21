"""Ileriye donuk eklenti / modul kayit sistemi (ModuleRegistry)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


class BaseToolModule(QWidget):
    """MeChi Studio'ya eklenecek tum arac modullerinin temel sinifi."""

    module_id: str = ""
    title: str = ""

    job_started = Signal(str)
    job_progress = Signal(int, str)
    job_finished = Signal(bool, str)
    handoff_requested = Signal(str, str)
    mascot_event = Signal(str)
    open_settings_requested = Signal()

    def __init__(self, context: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = context

    def receive_file(self, file_path: str, source_module: str = "") -> None:
        """Baska bir modulden tek tikla aktarilan gorsel veya belgeyi yukler."""

    def on_activated(self) -> None:
        """Kullanici bu modulu actiginda cagirilir."""

    def on_ps_context_changed(self, doc_info: Optional[dict]) -> None:
        """Photoshop aktif belge durumu degistiginde cagirilir."""

    def cancel_active_job(self) -> None:
        """Devam eden arka plan isini iptal eder."""


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    title: str
    short_desc: str
    badge_label: str
    order: int
    icon_char: str
    widget_cls: Type[BaseToolModule]


class ModuleRegistry:
    """Kayıtlı tum calisma modullerini yonetir."""

    def __init__(self) -> None:
        self._specs: Dict[str, ModuleSpec] = {}

    def register(self, spec: ModuleSpec) -> None:
        if not spec.module_id:
            raise ValueError("module_id bos olamaz")
        self._specs[spec.module_id] = spec

    def all_specs(self) -> List[ModuleSpec]:
        return sorted(self._specs.values(), key=lambda s: (s.order, s.title))

    def get(self, module_id: str) -> Optional[ModuleSpec]:
        return self._specs.get(module_id)
