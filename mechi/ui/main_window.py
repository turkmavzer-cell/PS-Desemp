"""MeChi Studio v1.1 ana penceresi — Sade, ferah ve moduler kabuk."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mechi import __version__
from mechi.config import LAUNCH_CTX_PATH, load_settings, save_settings
from mechi.core.api_bridge import ImageApiBridge
from mechi.core.mascot_engine import MascotEngine
from mechi.core.module_registry import BaseToolModule, ModuleRegistry, ModuleSpec
from mechi.core.prompt_engine import PromptEngine
from mechi.core.ps_bridge import PhotoshopBridge
from mechi.ui.mascot_widget import MascotStageWidget
from mechi.ui.settings_dialog import SettingsDialog


class _PsPollWorker(QThread):
    status_ready = Signal(dict)

    def __init__(self, ps_bridge: PhotoshopBridge, force: bool = False) -> None:
        super().__init__()
        self.ps_bridge = ps_bridge
        self.force = force

    def run(self) -> None:
        self.status_ready.emit(self.ps_bridge.get_status(force_jsx_probe=self.force))


class _ModuleCard(QFrame):
    clicked = Signal(str)

    def __init__(self, spec: ModuleSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModuleCard")
        self.module_id = spec.module_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)

        lbl_icon = QLabel(spec.icon_char)
        lbl_icon.setStyleSheet("font-size: 22px; color: #E0A34D;")
        lay.addWidget(lbl_icon)

        lbl_title = QLabel(spec.title)
        lbl_title.setObjectName("CardTitle")
        lay.addWidget(lbl_title)

        lbl_desc = QLabel(spec.short_desc)
        lbl_desc.setObjectName("MutedText")
        lbl_desc.setWordWrap(True)
        lay.addWidget(lbl_desc, 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.module_id)
        super().mousePressEvent(event)


class MeChiMainWindow(QMainWindow):
    """MeChi Studio ana penceresi."""

    def __init__(self, registry: ModuleRegistry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MeChi Studio")
        self.resize(1080, 880)
        self.setMinimumSize(880, 680)

        self.settings = load_settings()
        self.registry = registry
        self.ps_bridge = PhotoshopBridge(self.settings.get("ps_preferred_version", "auto"))
        self.prompt_engine = PromptEngine()
        self.api_bridge = ImageApiBridge()
        self.mascot_engine = MascotEngine()

        self.shared_context = {
            "ps_bridge": self.ps_bridge,
            "prompt_engine": self.prompt_engine,
            "api_bridge": self.api_bridge,
            "mascot_engine": self.mascot_engine,
        }

        self._modules: Dict[str, BaseToolModule] = {}
        self._nav_buttons: Dict[str, QPushButton] = {}
        self._active_page_id: str = "home"
        self._active_job_module: Optional[BaseToolModule] = None
        self._ps_worker: Optional[_PsPollWorker] = None

        self._build_ui()
        self._populate_modules()
        self._apply_always_on_top(bool(self.settings.get("always_on_top", False)))

        self._ps_timer = QTimer(self)
        self._ps_timer.timeout.connect(self.refresh_ps_status)
        self._ps_timer.start(4500)

        self._consume_launch_context()
        self.refresh_ps_status()
        self.select_page("home")
        if bool(self.settings.get("mascot_enabled", True)):
            self.mascot_stage.start_opening()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Ust Bar ---
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setFixedHeight(54)
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(20, 0, 20, 0)
        tb_lay.setSpacing(14)

        lbl_brand = QLabel("MeChi")
        lbl_brand.setStyleSheet("font-size: 17px; font-weight: 700; color: #F5F7F6;")
        tb_lay.addWidget(lbl_brand)

        self.btn_ps_status = QPushButton("○ Photoshop aranıyor...")
        self.btn_ps_status.setObjectName("PsStatusPill")
        self.btn_ps_status.setProperty("connected", False)
        self.btn_ps_status.setMinimumWidth(230)
        self.btn_ps_status.setToolTip("Photoshop'taki güncel aktif belgeyi yeniden okumak için tıklayın")
        self.btn_ps_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ps_status.clicked.connect(lambda: self.refresh_ps_status(force=True))
        tb_lay.addWidget(self.btn_ps_status)

        tb_lay.addStretch(1)

        self.btn_pin = QPushButton("Üstte Tut")
        self.btn_pin.setCheckable(True)
        self.btn_pin.setChecked(bool(self.settings.get("always_on_top", False)))
        self.btn_pin.toggled.connect(self._on_pin_toggled)
        tb_lay.addWidget(self.btn_pin)

        btn_settings = QPushButton("Ayarlar")
        btn_settings.clicked.connect(self.open_settings)
        tb_lay.addWidget(btn_settings)

        root.addWidget(top_bar)

        # --- Orta Alan ---
        mid_lay = QHBoxLayout()
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(0)

        # Sol Gezinme Seridi
        self.nav_rail = QFrame()
        self.nav_rail.setObjectName("NavRail")
        self.nav_rail.setFixedWidth(204)
        self.nav_layout = QVBoxLayout(self.nav_rail)
        self.nav_layout.setContentsMargins(12, 16, 12, 14)
        self.nav_layout.setSpacing(6)

        btn_home = self._make_nav_button("home", "Ana Ekran", "⌂")
        self.nav_layout.addWidget(btn_home)
        mid_lay.addWidget(self.nav_rail)

        # Sag Icerik Alani
        content_wrap = QWidget()
        c_lay = QVBoxLayout(content_wrap)
        c_lay.setContentsMargins(24, 16, 24, 12)
        c_lay.setSpacing(14)

        # Otomatik Kedi Maskot Sahnesi
        pause_sec = float(self.settings.get("mascot_pause_between_sec", 2.0))
        self.mascot_stage = MascotStageWidget(self.mascot_engine, pause_between_sec=pause_sec)
        self.mascot_stage.set_enabled(bool(self.settings.get("mascot_enabled", True)), pause_sec=pause_sec)
        c_lay.addWidget(self.mascot_stage)

        # Sayfa Yigini
        self.stack = QStackedWidget()
        c_lay.addWidget(self.stack, 1)

        mid_lay.addWidget(content_wrap, 1)
        root.addLayout(mid_lay, 1)

        # --- Alt Durum Cubugu ---
        footer = QFrame()
        footer.setObjectName("StatusFooter")
        footer.setFixedHeight(38)
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(20, 0, 20, 0)
        f_lay.setSpacing(12)

        self.lbl_footer_status = QLabel("Hazır")
        self.lbl_footer_status.setObjectName("MutedText")
        f_lay.addWidget(self.lbl_footer_status, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        f_lay.addWidget(self.progress_bar)

        self.btn_cancel_job = QPushButton("İptal")
        self.btn_cancel_job.setFixedHeight(26)
        self.btn_cancel_job.setVisible(False)
        self.btn_cancel_job.clicked.connect(self._cancel_active_job)
        f_lay.addWidget(self.btn_cancel_job)

        root.addWidget(footer)

    def _make_nav_button(self, page_id: str, label: str, icon_char: str) -> QPushButton:
        btn = QPushButton(f"{icon_char}  {label}")
        btn.setObjectName("NavButton")
        btn.setMinimumHeight(42)
        btn.setProperty("active", False)
        btn.clicked.connect(lambda: self.select_page(page_id))
        self._nav_buttons[page_id] = btn
        return btn

    def _populate_modules(self) -> None:
        home_scroll = QScrollArea()
        home_scroll.setWidgetResizable(True)
        home_page = QWidget()
        home_scroll.setWidget(home_page)
        outer_h = QHBoxLayout(home_page)
        outer_h.setContentsMargins(0, 4, 0, 12)
        outer_h.addStretch(1)

        col_w = QWidget()
        col_w.setMinimumWidth(720)
        col_w.setMaximumWidth(860)
        h_lay = QVBoxLayout(col_w)
        h_lay.setContentsMargins(0, 6, 0, 10)
        h_lay.setSpacing(12)

        hero = QLabel("MeChi'ye Hoş Geldiniz.")
        hero.setObjectName("HeroTitle")
        h_lay.addWidget(hero)

        sub = QLabel("Ne yapmak istersiniz?")
        sub.setObjectName("MutedText")
        h_lay.addWidget(sub)

        h_lay.addSpacing(4)

        cards_grid = QGridLayout()
        cards_grid.setSpacing(14)

        for idx, spec in enumerate(self.registry.all_specs()):
            mod = spec.widget_cls(self.shared_context)
            mod.job_started.connect(lambda msg, m=mod: self._on_job_started(m, msg))
            mod.job_progress.connect(self._on_job_progress)
            mod.job_finished.connect(self._on_job_finished)
            mod.handoff_requested.connect(self._on_handoff)
            mod.mascot_event.connect(self.mascot_stage.play_event)
            mod.open_settings_requested.connect(self.open_settings)

            self._modules[spec.module_id] = mod
            nav_btn = self._make_nav_button(spec.module_id, spec.title, spec.icon_char)
            self.nav_layout.addWidget(nav_btn)

            card = _ModuleCard(spec)
            card.clicked.connect(self.select_page)
            row, col_idx = divmod(idx, 2)
            cards_grid.addWidget(card, row, col_idx)

        cards_grid.setColumnStretch(0, 1)
        cards_grid.setColumnStretch(1, 1)
        h_lay.addLayout(cards_grid)
        h_lay.addStretch(1)
        outer_h.addWidget(col_w, 10)
        outer_h.addStretch(1)

        self.nav_layout.addStretch(1)
        lbl_ver = QLabel(f"v{__version__}")
        lbl_ver.setObjectName("SubtleHint")
        self.nav_layout.addWidget(lbl_ver)

        self.stack.addWidget(home_scroll)
        self._page_widgets: Dict[str, QWidget] = {"home": home_scroll}
        for spec in self.registry.all_specs():
            w = self._modules[spec.module_id]
            self.stack.addWidget(w)
            self._page_widgets[spec.module_id] = w

    def select_page(self, page_id: str) -> None:
        if page_id not in self._page_widgets:
            page_id = "home"
        self._active_page_id = page_id
        self.stack.setCurrentWidget(self._page_widgets[page_id])
        self.mascot_stage.set_compact_mode(False)

        for pid, btn in self._nav_buttons.items():
            btn.setProperty("active", pid == page_id)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if page_id in self._modules:
            self._modules[page_id].on_activated()

    def _on_handoff(self, target_module_id: str, file_path: str) -> None:
        mod = self._modules.get(target_module_id)
        if not mod:
            return
        mod.receive_file(file_path, source_module=self._active_page_id)
        self.select_page(target_module_id)

    def _on_job_started(self, mod: BaseToolModule, msg: str) -> None:
        self._active_job_module = mod
        self.lbl_footer_status.setText(msg)
        self.progress_bar.setValue(5)
        self.progress_bar.setVisible(True)
        self.btn_cancel_job.setVisible(True)

    def _on_job_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        self.lbl_footer_status.setText(msg)

    def _on_job_finished(self, ok: bool, msg: str) -> None:
        self._active_job_module = None
        self.progress_bar.setVisible(False)
        self.btn_cancel_job.setVisible(False)
        self.lbl_footer_status.setText(msg)
        self.refresh_ps_status()

    def _cancel_active_job(self) -> None:
        if self._active_job_module is not None:
            self._active_job_module.cancel_active_job()
            self.lbl_footer_status.setText("İptal isteniyor...")

    def refresh_ps_status(self, force: bool = False) -> None:
        if self._ps_worker is not None and self._ps_worker.isRunning():
            return
        w = _PsPollWorker(self.ps_bridge, force=bool(force))
        w.status_ready.connect(self._on_ps_status_ready)
        self._ps_worker = w
        w.start()

    def _on_ps_status_ready(self, st: dict) -> None:
        connected = bool(st.get("connected"))
        active_doc = st.get("active_doc")
        ps_lbl = f"PS {st['year']}" if st.get("year") else "PS"
        if connected and active_doc:
            self.btn_ps_status.setText(f"● {ps_lbl} Bağlı: {active_doc['name']}")
            self.btn_ps_status.setProperty("connected", True)
        elif connected:
            self.btn_ps_status.setText(f"● {ps_lbl} Bağlı (Açık belge yok)")
            self.btn_ps_status.setProperty("connected", True)
        else:
            self.btn_ps_status.setText("○ Bağımsız Mod (PS kapalı)")
            self.btn_ps_status.setProperty("connected", False)

        self.btn_ps_status.style().unpolish(self.btn_ps_status)
        self.btn_ps_status.style().polish(self.btn_ps_status)
        self.btn_ps_status.adjustSize()

        for mod in self._modules.values():
            mod.on_ps_context_changed(active_doc if connected else None)

    def _consume_launch_context(self) -> None:
        if not os.path.exists(LAUNCH_CTX_PATH):
            return
        try:
            with open(LAUNCH_CTX_PATH, "r", encoding="utf-8") as fp:
                ctx = json.load(fp)
            if isinstance(ctx, dict) and ctx.get("active_doc_name"):
                doc_info = {
                    "name": ctx["active_doc_name"],
                    "path": ctx.get("active_doc_path", ""),
                    "width_px": int(ctx.get("width_px") or 0),
                    "height_px": int(ctx.get("height_px") or 0),
                    "layer_count": int(ctx.get("layer_count") or 1),
                    "has_layers": int(ctx.get("layer_count") or 1) > 1,
                }
                self._on_ps_status_ready({"connected": True, "active_doc": doc_info})
        except Exception:
            pass

    def _on_pin_toggled(self, checked: bool) -> None:
        self.settings["always_on_top"] = checked
        save_settings(self.settings)
        self._apply_always_on_top(checked)

    def _apply_always_on_top(self, on_top: bool) -> None:
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on_top)
        if was_visible:
            self.show()

    def open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.settings = load_settings()
            self.ps_bridge.preferred_version = self.settings.get("ps_preferred_version", "auto")
            self.mascot_stage.set_enabled(
                bool(self.settings.get("mascot_enabled", True)),
                pause_sec=float(self.settings.get("mascot_pause_between_sec", 2.0)),
            )
            for mod in self._modules.values():
                mod.on_activated()

    def raise_from_ipc(self) -> None:
        self._consume_launch_context()
        self.refresh_ps_status(force=True)
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self._ps_timer.stop()
        if self._ps_worker is not None and self._ps_worker.isRunning():
            self._ps_worker.wait(4500)
        for mod in self._modules.values():
            mod.cancel_active_job()
        self.prompt_engine.close()
        super().closeEvent(event)
