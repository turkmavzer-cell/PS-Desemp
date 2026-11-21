"""Gorsel Uret Modulu — Sade tasarim, gizli prompt karti ve secilebilir API."""
from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mechi.config import get_env_key, load_settings, save_to_downloads
from mechi.core.api_bridge import ImageApiBridge
from mechi.core.module_registry import BaseToolModule
from mechi.core.prompt_engine import PromptEngine, PromptResult
from mechi.core.style_learner import learn_style_from_folder
from mechi.core.worker import TaskWorker


class StyleLearnDialog(QDialog):
    """Referans gorsel klasorunden Gemini Flash ile Stil DNA profili cikarir."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Klasörden Tasarım Dili Öğren")
        self.setMinimumWidth(500)
        self.folder_path = ""
        self.profile_name = ""

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        info = QLabel(
            "Örnek tasarımları topladığınız klasörü seçin. Marka ve logo adları elenerek "
            "yalnız görsel tasarım dili yeni bir stil profiline dönüştürülür."
        )
        info.setWordWrap(True)
        info.setObjectName("MutedText")
        lay.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Örn: Akdeniz Tenis Stili")
        form.addRow("Yeni Stil Adı:", self.txt_name)

        row_dir = QWidget()
        hl = QHBoxLayout(row_dir)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        self.txt_dir = QLineEdit()
        self.txt_dir.setPlaceholderText("Örnek görsellerin bulunduğu klasör...")
        btn_dir = QPushButton("Seç...")
        btn_dir.clicked.connect(self._browse)
        hl.addWidget(self.txt_dir, 1)
        hl.addWidget(btn_dir)
        form.addRow("Görsel Klasörü:", row_dir)
        lay.addLayout(form)

        key_ok = bool(get_env_key("GEMINI_API_KEY"))
        lbl_key = QLabel(
            "Ücretsiz Gemini API anahtarı hazır."
            if key_ok
            else "Not: Ayarlar > Görsel API bölümünden ücretsiz GEMINI_API_KEY girilmelidir."
        )
        lbl_key.setObjectName("BadgeOk" if key_ok else "BadgeWarn")
        lay.addWidget(lbl_key)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btn_cancel = QPushButton("İptal")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Analizi Başlat")
        btn_ok.setObjectName("PrimaryButton")
        btn_ok.setEnabled(key_ok)
        btn_ok.clicked.connect(self._accept)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        lay.addLayout(btns)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Referans Görsel Klasörü Seçin")
        if d:
            self.txt_dir.setText(d)

    def _accept(self) -> None:
        if not self.txt_name.text().strip():
            QMessageBox.warning(self, "Eksik Bilgi", "Lütfen yeni stil için bir ad girin.")
            return
        if not os.path.isdir(self.txt_dir.text().strip()):
            QMessageBox.warning(self, "Geçersiz Klasör", "Lütfen geçerli bir görsel klasörü seçin.")
            return
        self.profile_name = self.txt_name.text().strip()
        self.folder_path = self.txt_dir.text().strip()
        self.accept()


class TasarimModule(BaseToolModule):
    """Marka/Stil DNA secimi, gizli kompakt zar promptu ve opsiyonel API gorsel uretimi."""

    module_id = "tasarim"
    title = "Görsel Üret"

    def __init__(self, context: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(context, parent)
        self.engine: PromptEngine = context["prompt_engine"]
        self.api_bridge: ImageApiBridge = context["api_bridge"]
        self.ps_bridge = context["ps_bridge"]
        self._last_result: Optional[PromptResult] = None
        self._last_image_path: str = ""
        self._worker: Optional[TaskWorker] = None
        self._lock_buttons: Dict[str, QPushButton] = {}
        self._build_ui()
        self.refresh_profiles()
        self.refresh_api_state()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        wrap = QWidget()
        scroll.setWidget(wrap)
        wrap_h = QHBoxLayout(wrap)
        wrap_h.setContentsMargins(0, 4, 0, 16)
        wrap_h.addStretch(1)

        col = QWidget()
        col.setMinimumWidth(640)
        col.setMaximumWidth(820)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 4, 0, 12)
        lay.setSpacing(14)

        # Baslik
        hdr = QHBoxLayout()
        lbl_title = QLabel("Tasarım Stüdyosu")
        lbl_title.setObjectName("SectionTitle")
        hdr.addWidget(lbl_title)
        hdr.addStretch(1)

        self.btn_api_state = QPushButton("API Ayarla")
        self.btn_api_state.clicked.connect(self.open_settings_requested.emit)
        hdr.addWidget(self.btn_api_state)
        lay.addLayout(hdr)

        # 1) Sade Stil & Metin Karti
        sec_style = QFrame()
        sec_style.setObjectName("CleanSection")
        s_lay = QVBoxLayout(sec_style)
        s_lay.setContentsMargins(18, 16, 18, 16)
        s_lay.setSpacing(12)

        row_prof = QHBoxLayout()
        row_prof.setSpacing(10)
        lbl_p = QLabel("Stil Profili")
        lbl_p.setObjectName("SectionLabel")
        lbl_p.setFixedWidth(105)
        self.cmb_profile = QComboBox()
        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)
        btn_learn = QPushButton("Klasörden Stil Öğren")
        btn_learn.clicked.connect(self._open_style_learner)
        row_prof.addWidget(lbl_p)
        row_prof.addWidget(self.cmb_profile, 1)
        row_prof.addWidget(btn_learn)
        s_lay.addLayout(row_prof)

        self.row_theme_widget = QWidget()
        self.row_theme_widget.setStyleSheet("background: transparent;")
        row_theme = QHBoxLayout(self.row_theme_widget)
        row_theme.setContentsMargins(0, 0, 0, 0)
        row_theme.setSpacing(10)
        lbl_th = QLabel("Kapsül ve Stil")
        lbl_th.setObjectName("SectionLabel")
        lbl_th.setFixedWidth(105)
        self.cmb_theme = QComboBox()
        self.cmb_theme.currentIndexChanged.connect(self._on_theme_changed)
        self.cmb_render_mode = QComboBox()
        self.cmb_render_mode.setFixedWidth(250)
        row_theme.addWidget(lbl_th)
        row_theme.addWidget(self.cmb_theme, 1)
        row_theme.addWidget(self.cmb_render_mode)
        self.row_theme_widget.setVisible(False)
        s_lay.addWidget(self.row_theme_widget)

        row_opts = QHBoxLayout()
        row_opts.setSpacing(10)
        lbl_t = QLabel("Baskı Yazısı")
        lbl_t.setObjectName("SectionLabel")
        lbl_t.setFixedWidth(105)
        self.txt_brand = QLineEdit()
        self.txt_brand.setPlaceholderText("İsteğe bağlı (boş bırakılırsa [YOUR BRAND TEXT])")
        self.cmb_bg = QComboBox()
        self.cmb_bg.addItem("Açık Zemin", "acik")
        self.cmb_bg.addItem("Koyu Zemin", "koyu")
        self.cmb_bg.setFixedWidth(145)
        row_opts.addWidget(lbl_t)
        row_opts.addWidget(self.txt_brand, 1)
        row_opts.addWidget(self.cmb_bg)
        s_lay.addLayout(row_opts)

        self.lbl_profile_stats = QLabel("")
        self.lbl_profile_stats.setObjectName("SubtleHint")
        s_lay.addWidget(self.lbl_profile_stats)

        lay.addWidget(sec_style)

        # 2) Zar Kilitleri (ilk zardan sonra gorunur) & Ana Buton
        self.locks_bar = QWidget()
        self.locks_bar.setVisible(False)
        lk_lay = QHBoxLayout(self.locks_bar)
        lk_lay.setContentsMargins(0, 0, 0, 0)
        lk_lay.setSpacing(8)
        lbl_lk = QLabel("Sabit Tut:")
        lbl_lk.setObjectName("MutedText")
        lk_lay.addWidget(lbl_lk)
        for cat, text in (
            ("frame", "Çerçeve"),
            ("ornament", "Süsleme"),
            ("subject", "Obje"),
            ("typography", "Yazı Stili"),
        ):
            btn = QPushButton(text)
            btn.setObjectName("LockToggle")
            btn.setCheckable(True)
            self._lock_buttons[cat] = btn
            lk_lay.addWidget(btn)
        lk_lay.addStretch(1)
        lay.addWidget(self.locks_bar)

        self.btn_roll = QPushButton("Prompt Oluştur")
        self.btn_roll.setObjectName("PrimaryButton")
        self.btn_roll.setMinimumHeight(46)
        self.btn_roll.clicked.connect(self.roll_prompt)
        lay.addWidget(self.btn_roll)

        # 3) Kompakt Gizli Prompt Sonuc Karti
        self.card_result = QFrame()
        self.card_result.setObjectName("PromptResultCard")
        self.card_result.setVisible(False)
        r_lay = QVBoxLayout(self.card_result)
        r_lay.setContentsMargins(18, 16, 18, 16)
        r_lay.setSpacing(10)

        r_top = QHBoxLayout()
        self.lbl_roll_status = QLabel("✓ Prompt oluşturuldu")
        self.lbl_roll_status.setObjectName("CardTitle")
        r_top.addWidget(self.lbl_roll_status)
        r_top.addStretch(1)

        self.btn_toggle_preview = QPushButton("Önizle")
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.toggled.connect(self._toggle_prompt_preview)
        r_top.addWidget(self.btn_toggle_preview)
        r_lay.addLayout(r_top)

        self.lbl_roll_summary = QLabel("")
        self.lbl_roll_summary.setObjectName("MutedText")
        self.lbl_roll_summary.setWordWrap(True)
        r_lay.addWidget(self.lbl_roll_summary)

        self.txt_hidden_prompt = QPlainTextEdit()
        self.txt_hidden_prompt.setReadOnly(True)
        self.txt_hidden_prompt.setMaximumHeight(150)
        self.txt_hidden_prompt.setVisible(False)
        r_lay.addWidget(self.txt_hidden_prompt)

        r_actions = QHBoxLayout()
        r_actions.setSpacing(10)
        self.btn_copy_prompt = QPushButton("Promptu Kopyala")
        self.btn_copy_prompt.setMinimumHeight(38)
        self.btn_copy_prompt.clicked.connect(self.copy_prompt)
        r_actions.addWidget(self.btn_copy_prompt)

        self.btn_generate_img = QPushButton("Görseli Üret")
        self.btn_generate_img.setObjectName("PrimaryButton")
        self.btn_generate_img.setMinimumHeight(38)
        self.btn_generate_img.clicked.connect(self.generate_image_from_prompt)
        r_actions.addWidget(self.btn_generate_img)

        r_actions.addStretch(1)
        self.lbl_session_count = QLabel("")
        self.lbl_session_count.setObjectName("SubtleHint")
        r_actions.addWidget(self.lbl_session_count)
        r_lay.addLayout(r_actions)

        lay.addWidget(self.card_result)

        # 4) Uretilen Gorsel Karti
        self.card_image = QFrame()
        self.card_image.setObjectName("CleanSection")
        self.card_image.setVisible(False)
        i_lay = QHBoxLayout(self.card_image)
        i_lay.setContentsMargins(16, 16, 16, 16)
        i_lay.setSpacing(16)

        self.lbl_img_preview = QLabel()
        self.lbl_img_preview.setFixedSize(200, 200)
        self.lbl_img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_preview.setStyleSheet("background: #121416; border: 1px solid #2B3136; border-radius: 8px;")
        i_lay.addWidget(self.lbl_img_preview)

        i_right = QVBoxLayout()
        self.lbl_img_title = QLabel("Üretilen Tasarım")
        self.lbl_img_title.setObjectName("CardTitle")
        i_right.addWidget(self.lbl_img_title)

        self.lbl_img_path = QLabel("")
        self.lbl_img_path.setObjectName("MutedText")
        self.lbl_img_path.setWordWrap(True)
        i_right.addWidget(self.lbl_img_path)
        i_right.addStretch(1)

        self.btn_download = QPushButton("İndirilenlere İndir")
        self.btn_download.clicked.connect(self._save_to_downloads)
        btn_open_ps = QPushButton("Photoshop'ta Aç")
        btn_open_ps.clicked.connect(self._open_generated_in_ps)
        btn_to_net = QPushButton("Netleştir")
        btn_to_net.clicked.connect(lambda: self.handoff_requested.emit("netlestir", self._last_image_path))
        btn_to_emp = QPushButton("Emprime Ayır")
        btn_to_emp.setObjectName("PrimaryButton")
        btn_to_emp.clicked.connect(lambda: self.handoff_requested.emit("emprime", self._last_image_path))

        h_btns = QHBoxLayout()
        h_btns.setSpacing(8)
        h_btns.addWidget(self.btn_download)
        h_btns.addWidget(btn_open_ps)
        h_btns.addWidget(btn_to_net)
        h_btns.addWidget(btn_to_emp)
        i_right.addLayout(h_btns)
        i_lay.addLayout(i_right, 1)

        lay.addWidget(self.card_image)
        lay.addStretch(1)

        wrap_h.addWidget(col, 10)
        wrap_h.addStretch(1)

    def on_activated(self) -> None:
        self.refresh_api_state()

    def refresh_profiles(self, select_id: Optional[str] = None) -> None:
        cur_id = select_id or self.cmb_profile.currentData()
        self.cmb_profile.blockSignals(True)
        self.cmb_profile.clear()
        for p in self.engine.list_profiles():
            self.cmb_profile.addItem(p["name"], p["id"])
        if cur_id:
            idx = self.cmb_profile.findData(cur_id)
            if idx >= 0:
                self.cmb_profile.setCurrentIndex(idx)
        self.cmb_profile.blockSignals(False)
        self._on_profile_changed()

    def refresh_api_state(self) -> None:
        cfg = load_settings()
        ok, msg = self.api_bridge.is_configured(cfg)
        self.btn_api_state.setText("API Hazır" if ok else "API Ayarla")
        self.btn_generate_img.setVisible(ok)
        self.lbl_session_count.setText(
            f"Oturum: {self.api_bridge.session_count} görsel" if ok else "Yalnız kopyalama modu"
        )

    def _on_profile_changed(self) -> None:
        pid = self.cmb_profile.currentData()
        for btn in self._lock_buttons.values():
            btn.setChecked(False)
        self._last_result = None

        if not pid:
            self.row_theme_widget.setVisible(False)
            self.lbl_profile_stats.setText("Profil bulunamadı.")
            return

        try:
            themes = self.engine.get_profile_themes(pid)
            modes = self.engine.get_profile_render_modes(pid)

            self.cmb_theme.blockSignals(True)
            self.cmb_theme.clear()
            for th in themes:
                self.cmb_theme.addItem(th.get("label_tr", th.get("id", "")), th.get("id", "all"))
            self.cmb_theme.blockSignals(False)

            self.cmb_render_mode.blockSignals(True)
            self.cmb_render_mode.clear()
            for rm in modes:
                self.cmb_render_mode.addItem(rm.get("label_tr", rm.get("id", "")), rm.get("id", "auto"))
            self.cmb_render_mode.blockSignals(False)

            self.cmb_theme.setVisible(bool(themes))
            self.cmb_render_mode.setVisible(bool(modes))
            self.row_theme_widget.setVisible(bool(themes or modes))
        except Exception:
            self.row_theme_widget.setVisible(False)

        self._update_profile_stats()

    def _on_theme_changed(self) -> None:
        for btn in self._lock_buttons.values():
            btn.setChecked(False)
        self._update_profile_stats()

    def _update_profile_stats(self) -> None:
        pid = self.cmb_profile.currentData()
        if not pid:
            self.lbl_profile_stats.setText("Profil bulunamadı.")
            return
        try:
            theme_id = (self.cmb_theme.currentData() or "all") if self.cmb_theme.isVisible() else "all"
            total = self.engine.get_total_combinations(pid, theme_id=theme_id)
            used = self.engine.get_roll_count(pid)
            txt = f"{total:,} benzersiz kombinasyon  /  Üretilen: {used}".replace(",", ".")
            if self.cmb_theme.isVisible() and theme_id != "all":
                txt += f"  ({self.cmb_theme.currentText()})"
            self.lbl_profile_stats.setText(txt)
        except Exception as exc:
            self.lbl_profile_stats.setText(f"Profil hatası: {exc}")

    def roll_prompt(self) -> Optional[PromptResult]:
        pid = self.cmb_profile.currentData()
        if not pid:
            QMessageBox.warning(self, "Profil Yok", "Önce geçerli bir stil profili seçin.")
            return None

        locked: Dict[str, str] = {}
        if self._last_result is not None:
            for cat, btn in self._lock_buttons.items():
                if btn.isChecked() and cat in self._last_result.picked:
                    locked[cat] = self._last_result.picked[cat]["id"]

        theme_id = (self.cmb_theme.currentData() or "all") if self.cmb_theme.isVisible() else "all"
        render_mode = (self.cmb_render_mode.currentData() or "auto") if self.cmb_render_mode.isVisible() else "auto"

        res = self.engine.roll_unique(
            profile_id=pid,
            locked=locked,
            custom_text=self.txt_brand.text().strip(),
            bg_mode=self.cmb_bg.currentData() or "acik",
            theme_id=theme_id,
            render_mode=render_mode,
        )
        self._last_result = res
        self.locks_bar.setVisible(True)

        self.lbl_roll_status.setText(f"✓ Prompt oluşturuldu (#{res.roll_number})")
        if res.theme_label_tr and res.theme_id != "all":
            self.lbl_roll_summary.setText(f"[{res.theme_label_tr}] {res.summary_tr}")
        else:
            self.lbl_roll_summary.setText(res.summary_tr)
        self.txt_hidden_prompt.setPlainText(res.full_prompt)
        self.card_result.setVisible(True)
        self._update_profile_stats()
        return res

    def _toggle_prompt_preview(self, checked: bool) -> None:
        self.txt_hidden_prompt.setVisible(checked)
        self.btn_toggle_preview.setText("Gizle" if checked else "Önizle")

    def copy_prompt(self) -> None:
        if not self._last_result:
            return
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self._last_result.full_prompt)
        self.btn_copy_prompt.setText("Kopyalandı")
        QTimer.singleShot(1800, lambda: self.btn_copy_prompt.setText("Promptu Kopyala"))

    def generate_image_from_prompt(self) -> None:
        if not self._last_result:
            self.roll_prompt()
        if not self._last_result:
            return
        cfg = load_settings()
        ok, msg = self.api_bridge.is_configured(cfg)
        if not ok:
            QMessageBox.information(self, "API Yapılandırılmadı", msg)
            return

        out_dir = cfg.get("output_dir") or ""
        prompt_text = self._last_result.full_prompt
        prefix = f"{self._last_result.profile_id}_{self._last_result.roll_number}"

        def _job(progress_cb, **_kw):
            progress_cb(25, "Görsel üretim API isteği gönderiliyor...")
            g_ok, g_res = self.api_bridge.generate_image(
                prompt=prompt_text,
                out_dir=out_dir,
                aspect="3:4",
                filename_prefix=prefix,
                settings=cfg,
            )
            if not g_ok:
                raise RuntimeError(g_res)
            progress_cb(100, "Görsel üretildi.")
            return g_res

        self._start_worker(_job, "API ile görsel üretiliyor...", self._on_image_generated)

    def _on_image_generated(self, out_path: str) -> None:
        self._last_image_path = str(out_path)
        pix = QPixmap(self._last_image_path)
        if not pix.isNull():
            self.lbl_img_preview.setPixmap(
                pix.scaled(192, 192, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        self.lbl_img_path.setText(self._last_image_path)
        self.card_image.setVisible(True)
        self.refresh_api_state()
        self.mascot_event.emit("success")
        self.job_finished.emit(True, f"Görsel üretildi: {os.path.basename(self._last_image_path)}")

    def _save_to_downloads(self) -> None:
        ok, res = save_to_downloads(self._last_image_path, reveal_in_explorer=True)
        if not ok:
            QMessageBox.warning(self, "İndirme Hatası", res)
            return
        self.btn_download.setText("İndirilenlerde Gösterildi ✓")
        QTimer.singleShot(2000, lambda: self.btn_download.setText("İndirilenlere İndir"))

    def _open_generated_in_ps(self) -> None:
        if not self._last_image_path:
            return
        ok, msg = self.ps_bridge.open_file(self._last_image_path)
        if not ok:
            QMessageBox.warning(self, "Photoshop Hatası", msg)

    def _open_style_learner(self) -> None:
        dlg = StyleLearnDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        api_key = get_env_key("GEMINI_API_KEY")

        def _job(progress_cb, log_cb, cancel_event, **_kw):
            return learn_style_from_folder(
                folder_path=dlg.folder_path,
                profile_name=dlg.profile_name,
                api_key=api_key,
                progress_cb=progress_cb,
                log_cb=log_cb,
                cancel_event=cancel_event,
            )

        self._start_worker(_job, f"Stil öğreniliyor: {dlg.profile_name}", self._on_style_learned)

    def _on_style_learned(self, res: dict) -> None:
        pid = res.get("profile_id")
        counts = res.get("counts", {})
        self.refresh_profiles(select_id=pid)
        self.mascot_event.emit("success")
        self.job_finished.emit(True, f"Yeni stil profili eklendi: {pid}")
        QMessageBox.information(
            self,
            "Stil Öğrenme Tamamlandı",
            f"Yeni profil kaydedildi: {pid}\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in counts.items()),
        )

    def _start_worker(self, fn, title: str, on_success) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Meşgul", "Devam eden işlem bitmeden yeni işlem başlatılamaz.")
            return
        self.job_started.emit(title)
        self.mascot_event.emit("busy")
        w = TaskWorker(fn)
        w.progress.connect(self.job_progress.emit)
        w.succeeded.connect(on_success)
        w.failed.connect(self._on_worker_failed)
        w.cancelled.connect(lambda: self.job_finished.emit(False, "İşlem iptal edildi."))
        self._worker = w
        w.start()

    def _on_worker_failed(self, err: str) -> None:
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "İşlem başarısız.")
        QMessageBox.warning(self, "Hata", err)

    def cancel_active_job(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
