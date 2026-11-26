"""Gorsel Duzelt Modulu — Bozuk/giyilmis gorselden MeChi 3.3 ozel duzeltme promptu."""
from __future__ import annotations

import os
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mechi.config import get_env_key, load_settings, save_to_downloads
from mechi.core.api_bridge import ImageApiBridge
from mechi.core.module_registry import BaseToolModule
from mechi.core.repair_analyzer import RepairPromptResult, analyze_and_build_repair_prompt
from mechi.core.worker import TaskWorker


class DuzeltModule(BaseToolModule):
    """Bozuk, kirisik veya giyilmis gorseli analiz edip o gorsele ozel 2D onarim promptu uretir."""

    module_id = "duzelt"
    title = "Görsel Düzelt"

    def __init__(self, context: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(context, parent)
        self.ps_bridge = context["ps_bridge"]
        self.api_bridge: ImageApiBridge = context["api_bridge"]
        self._active_doc: Optional[dict] = None
        self._last_result: Optional[RepairPromptResult] = None
        self._last_src_path: str = ""
        self._last_generated_path: str = ""
        self._worker: Optional[TaskWorker] = None
        self.setAcceptDrops(True)
        self._build_ui()
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
        lbl_title = QLabel("Sıfırdan 2D Yeniden Yaratım (MeChi 3.3)")
        lbl_title.setObjectName("SectionTitle")
        hdr.addWidget(lbl_title)
        hdr.addStretch(1)
        self.btn_api_state = QPushButton("API Ayarla")
        self.btn_api_state.clicked.connect(self.open_settings_requested.emit)
        hdr.addWidget(self.btn_api_state)
        lay.addLayout(hdr)

        # 1) Kaynak Karti (+ Kucuk Onizleme)
        sec_src = QFrame()
        sec_src.setObjectName("CleanSection")
        src_outer = QHBoxLayout(sec_src)
        src_outer.setContentsMargins(18, 16, 18, 16)
        src_outer.setSpacing(16)

        src_left = QVBoxLayout()
        src_left.setSpacing(8)
        lbl_s = QLabel("Bozuk veya Giyilmiş Görsel")
        lbl_s.setObjectName("SectionLabel")
        src_left.addWidget(lbl_s)

        self.rb_ps = QRadioButton("Photoshop'ta açık belge (Açık belge yok)")
        self.rb_file = QRadioButton("Bilgisayardan seç (JPEG, PNG, TIF, PSD)")
        self.rb_file.setChecked(True)
        self.rb_ps.setEnabled(False)
        src_left.addWidget(self.rb_ps)
        src_left.addWidget(self.rb_file)

        f_row = QHBoxLayout()
        f_row.setSpacing(8)
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Bozuk görseli sürükleyin veya seçin...")
        self.txt_path.textChanged.connect(self._update_src_thumb)
        btn_br = QPushButton("Seç...")
        btn_br.clicked.connect(self._browse_file)
        f_row.addWidget(self.txt_path, 1)
        f_row.addWidget(btn_br)
        src_left.addLayout(f_row)
        src_outer.addLayout(src_left, 1)

        self.lbl_src_thumb = QLabel("Önizleme")
        self.lbl_src_thumb.setFixedSize(112, 112)
        self.lbl_src_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_src_thumb.setObjectName("SubtleHint")
        self.lbl_src_thumb.setStyleSheet("background: #121416; border: 1px solid #2B3136; border-radius: 8px;")
        src_outer.addWidget(self.lbl_src_thumb)
        lay.addWidget(sec_src)

        # Analiz modu bilgisi + tek ana buton
        self.lbl_vision_hint = QLabel("")
        self.lbl_vision_hint.setObjectName("SubtleHint")
        self.lbl_vision_hint.setWordWrap(True)
        lay.addWidget(self.lbl_vision_hint)

        self.btn_analyze = QPushButton("Görseli Analiz Et ve Prompt Oluştur")
        self.btn_analyze.setObjectName("PrimaryButton")
        self.btn_analyze.setMinimumHeight(46)
        self.btn_analyze.clicked.connect(self.run_analysis)
        lay.addWidget(self.btn_analyze)

        # 3) Kompakt Gizli Prompt Sonuc Karti
        self.card_result = QFrame()
        self.card_result.setObjectName("PromptResultCard")
        self.card_result.setVisible(False)
        r_lay = QVBoxLayout(self.card_result)
        r_lay.setContentsMargins(18, 16, 18, 16)
        r_lay.setSpacing(10)

        r_top = QHBoxLayout()
        self.lbl_res_title = QLabel("✓ Sıfırdan 2D yeniden yaratım promptu hazır")
        self.lbl_res_title.setObjectName("CardTitle")
        r_top.addWidget(self.lbl_res_title)
        r_top.addStretch(1)

        self.lbl_mode_badge = QLabel("")
        self.lbl_mode_badge.setObjectName("BadgeOk")
        r_top.addWidget(self.lbl_mode_badge)

        self.btn_toggle_preview = QPushButton("Önizle")
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.toggled.connect(self._toggle_preview)
        r_top.addWidget(self.btn_toggle_preview)
        r_lay.addLayout(r_top)

        self.lbl_res_summary = QLabel("")
        self.lbl_res_summary.setObjectName("MutedText")
        self.lbl_res_summary.setWordWrap(True)
        r_lay.addWidget(self.lbl_res_summary)

        # Algilanan spot palet seridi
        self.palette_row = QHBoxLayout()
        self.palette_row.setSpacing(6)
        r_lay.addLayout(self.palette_row)

        self.txt_hidden_prompt = QPlainTextEdit()
        self.txt_hidden_prompt.setReadOnly(True)
        self.txt_hidden_prompt.setMaximumHeight(160)
        self.txt_hidden_prompt.setVisible(False)
        r_lay.addWidget(self.txt_hidden_prompt)

        r_act = QHBoxLayout()
        r_act.setSpacing(10)
        self.btn_copy = QPushButton("Promptu Kopyala")
        self.btn_copy.setMinimumHeight(38)
        self.btn_copy.clicked.connect(self.copy_prompt)
        r_act.addWidget(self.btn_copy)

        self.btn_download_ref = QPushButton("Ön-Düzeltilmiş Referansı İndir")
        self.btn_download_ref.setMinimumHeight(38)
        self.btn_download_ref.setVisible(False)
        self.btn_download_ref.clicked.connect(self._download_rectified_ref)
        r_act.addWidget(self.btn_download_ref)

        self.btn_api_repair = QPushButton("API ile Sıfırdan 2D Yeniden Yarat")
        self.btn_api_repair.setObjectName("PrimaryButton")
        self.btn_api_repair.setMinimumHeight(38)
        self.btn_api_repair.clicked.connect(self.generate_repaired_image)
        r_act.addWidget(self.btn_api_repair)

        r_act.addStretch(1)
        self.lbl_mods = QLabel("")
        self.lbl_mods.setObjectName("SubtleHint")
        r_act.addWidget(self.lbl_mods)
        r_lay.addLayout(r_act)

        lay.addWidget(self.card_result)

        # 4) API Uretilen Duzeltilmis Gorsel Karti
        self.card_img = QFrame()
        self.card_img.setObjectName("CleanSection")
        self.card_img.setVisible(False)
        i_lay = QHBoxLayout(self.card_img)
        i_lay.setContentsMargins(16, 16, 16, 16)
        i_lay.setSpacing(16)

        self.lbl_out_preview = QLabel()
        self.lbl_out_preview.setFixedSize(180, 180)
        self.lbl_out_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_out_preview.setStyleSheet("background: #121416; border: 1px solid #2B3136; border-radius: 8px;")
        i_lay.addWidget(self.lbl_out_preview)

        i_right = QVBoxLayout()
        lbl_done = QLabel("Düzeltilmiş 2D Çıktı")
        lbl_done.setObjectName("CardTitle")
        i_right.addWidget(lbl_done)

        self.lbl_out_path = QLabel("")
        self.lbl_out_path.setObjectName("MutedText")
        self.lbl_out_path.setWordWrap(True)
        i_right.addWidget(self.lbl_out_path)
        i_right.addStretch(1)

        h_btns = QHBoxLayout()
        h_btns.setSpacing(8)
        self.btn_download = QPushButton("İndirilenlere İndir")
        self.btn_download.clicked.connect(self._save_to_downloads)
        btn_ps = QPushButton("Photoshop'ta Aç")
        btn_ps.clicked.connect(self._open_in_ps)
        btn_net = QPushButton("Netleştir")
        btn_net.clicked.connect(lambda: self.handoff_requested.emit("netlestir", self._last_generated_path))
        btn_emp = QPushButton("Emprime Ayır")
        btn_emp.setObjectName("PrimaryButton")
        btn_emp.clicked.connect(lambda: self.handoff_requested.emit("emprime", self._last_generated_path))
        h_btns.addWidget(self.btn_download)
        h_btns.addWidget(btn_ps)
        h_btns.addWidget(btn_net)
        h_btns.addWidget(btn_emp)
        i_right.addLayout(h_btns)
        i_lay.addLayout(i_right, 1)

        lay.addWidget(self.card_img)
        lay.addStretch(1)

        wrap_h.addWidget(col, 10)
        wrap_h.addStretch(1)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.receive_file(path, "drag_drop")

    def receive_file(self, file_path: str, source_module: str = "") -> None:
        if file_path and os.path.exists(file_path):
            self.rb_file.setChecked(True)
            self.txt_path.setText(file_path)

    def on_activated(self) -> None:
        self.refresh_api_state()

    def refresh_api_state(self) -> None:
        cfg = load_settings()
        ok, _ = self.api_bridge.is_configured(cfg)
        self.btn_api_state.setText("API Hazır" if ok else "API Ayarla")
        self.btn_api_repair.setVisible(ok)
        if get_env_key("GEMINI_API_KEY"):
            self.lbl_vision_hint.setText(
                "İnternet Derin Analiz (Gemini Vision) aktif — görseldeki motif, yazı ve geometri otomatik okunur."
            )
        else:
            self.lbl_vision_hint.setText(
                "Yerel Python Analizi aktif — görselin içeriğini (yazı/motif) otomatik okumak için sağ üstten ücretsiz Gemini API anahtarı ekleyebilirsiniz."
            )

    def on_ps_context_changed(self, doc_info: Optional[dict]) -> None:
        self._active_doc = doc_info
        if doc_info:
            self.rb_ps.setEnabled(True)
            self.rb_ps.setText(f"Photoshop'ta açık belge: {doc_info['name']}")
            if not self.txt_path.text().strip():
                self.rb_ps.setChecked(True)
        else:
            self.rb_ps.setEnabled(False)
            self.rb_ps.setText("Photoshop'ta açık belge (Açık belge yok)")
            if self.rb_ps.isChecked():
                self.rb_file.setChecked(True)

    def _update_src_thumb(self) -> None:
        p = self.txt_path.text().strip()
        if p and os.path.exists(p):
            pix = QPixmap(p)
            if not pix.isNull():
                self.lbl_src_thumb.setPixmap(
                    pix.scaled(104, 104, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
                return
        self.lbl_src_thumb.setPixmap(QPixmap())
        self.lbl_src_thumb.setText("Önizleme")

    def _browse_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Düzeltilecek Bozuk Görseli Seçin",
            "",
            "Görseller (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.psd);;Tüm Dosyalar (*.*)",
        )
        if f:
            self.receive_file(f, "browse")

    def run_analysis(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Meşgul", "Devam eden analiz bitmeden yenisi başlatılamaz.")
            return

        use_ps = self.rb_ps.isChecked() and self._active_doc is not None
        file_path = self.txt_path.text().strip()
        if not use_ps and (not file_path or not os.path.exists(file_path)):
            QMessageBox.warning(self, "Görsel Seçilmedi", "Lütfen bir görsel seçin veya açık PS belgesi kullanın.")
            return

        cfg = load_settings()
        out_dir = cfg.get("output_dir") or os.getcwd()
        api_key = get_env_key("GEMINI_API_KEY")
        overrides = {
            "source_condition": "auto",
            "render_style": "auto",
            "scope": "auto",
            "bg": "auto",
            "text_mode": "auto",
            "custom_text": "",
            "extra_note": "",
            "strict_single_output": True,
            "strict_zero_hallucination": True,
            "treat_text_as_shape": False,
            "license_safe": False,
            "pre_rectify": True,
        }
        exp_name = self._active_doc["name"] if (use_ps and self._active_doc) else None

        def _job(progress_cb, cancel_event, **_kw):
            src_img = file_path
            if use_ps and exp_name:
                os.makedirs(out_dir, exist_ok=True)
                src_img = os.path.join(out_dir, "_mechi_duzelt_src.png")
                progress_cb(8, "Photoshop belgesi dışa aktarılıyor...")
                ok_ex, msg_ex = self.ps_bridge.export_active_doc_png(src_img, expected_doc_name=exp_name)
                if not ok_ex:
                    raise RuntimeError(msg_ex)
            res = analyze_and_build_repair_prompt(
                src_img,
                overrides=overrides,
                api_key=api_key,
                rectify_out_dir=out_dir,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )
            return src_img, res

        self.btn_analyze.setEnabled(False)
        self.job_started.emit("Görsel analiz ediliyor...")
        self.mascot_event.emit("busy")
        w = TaskWorker(_job)
        w.progress.connect(self.job_progress.emit)
        w.succeeded.connect(self._on_analysis_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        self._worker = w
        w.start()

    def _clear_palette_row(self) -> None:
        while self.palette_row.count():
            item = self.palette_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_analysis_done(self, payload: tuple) -> None:
        self.btn_analyze.setEnabled(True)
        src_img, res = payload
        self._last_src_path = src_img
        self._last_result = res

        self.apply_result_to_card(res)
        self.mascot_event.emit("success")
        self.job_finished.emit(True, "Görsel analizi ve düzeltme promptu hazır.")

    def apply_result_to_card(self, res: RepairPromptResult) -> None:
        self.lbl_res_title.setText(f"✓ Yeniden yaratım promptu güncellendi ({time.strftime('%H:%M:%S')})")
        is_vis = res.analysis_mode == "gemini_vision"
        self.lbl_mode_badge.setText("Gemini Vision Derin Analiz" if is_vis else "Yerel Python Analizi")
        self.lbl_mode_badge.setObjectName("BadgeOk" if is_vis else "BadgeWarn")
        self.lbl_mode_badge.style().unpolish(self.lbl_mode_badge)
        self.lbl_mode_badge.style().polish(self.lbl_mode_badge)

        summary = res.summary_tr
        if res.fallback_note:
            summary += f"\n({res.fallback_note})"
        self.lbl_res_summary.setText(summary)
        self.lbl_mods.setText(f"Modüller: {res.detected_modules}")
        self.txt_hidden_prompt.setPlainText(res.full_prompt)

        self._clear_palette_row()
        lbl_p = QLabel("Palet:")
        lbl_p.setObjectName("SubtleHint")
        self.palette_row.addWidget(lbl_p)
        for hx in res.palette_hexes:
            chip = QLabel(f"  {hx}  ")
            chip.setStyleSheet(
                f"background-color: {hx}; color: #FFFFFF; border: 1px solid #3E464D; "
                f"border-radius: 5px; padding: 2px 6px; font-size: 11px; font-weight: 700;"
            )
            # Acik renkte siyah yazi
            r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
            if (0.299 * r + 0.587 * g + 0.114 * b) > 150:
                chip.setStyleSheet(
                    f"background-color: {hx}; color: #141618; border: 1px solid #3E464D; "
                    f"border-radius: 5px; padding: 2px 6px; font-size: 11px; font-weight: 700;"
                )
            self.palette_row.addWidget(chip)
        self.palette_row.addStretch(1)

        self.card_result.setVisible(True)
        has_rect_ref = bool(res.rectified_ref_path and os.path.exists(res.rectified_ref_path))
        self.btn_download_ref.setVisible(has_rect_ref)
        self.refresh_api_state()

    def _download_rectified_ref(self) -> None:
        if not self._last_result or not self._last_result.rectified_ref_path:
            return
        ok, res = save_to_downloads(self._last_result.rectified_ref_path, reveal_in_explorer=True)
        if not ok:
            QMessageBox.warning(self, "İndirme Hatası", res)
            return
        self.btn_download_ref.setText("Ön-Düzeltilmiş Ref İndirildi ✓")
        QTimer.singleShot(2000, lambda: self.btn_download_ref.setText("Ön-Düzeltilmiş Referansı İndir"))

    def _toggle_preview(self, checked: bool) -> None:
        self.txt_hidden_prompt.setVisible(checked)
        self.btn_toggle_preview.setText("Gizle" if checked else "Önizle")

    def copy_prompt(self) -> None:
        if not self._last_result:
            return
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self._last_result.full_prompt)
        self.btn_copy.setText("Kopyalandı")
        QTimer.singleShot(1800, lambda: self.btn_copy.setText("Promptu Kopyala"))

    def generate_repaired_image(self) -> None:
        if not self._last_result:
            return
        cfg = load_settings()
        ok, msg = self.api_bridge.is_configured(cfg)
        if not ok:
            QMessageBox.information(self, "API Yapılandırılmadı", msg)
            return

        out_dir = cfg.get("output_dir") or os.getcwd()
        prompt_text = self._last_result.full_prompt
        sys_inst = self._last_result.system_instruction
        aspect = self._last_result.metrics.recommended_aspect or self._last_result.metrics.aspect_label
        rect_p = self._last_result.rectified_ref_path
        if rect_p and os.path.exists(rect_p):
            ref_path = rect_p
        else:
            ref_path = self._last_src_path if os.path.exists(self._last_src_path) else None

        def _job(progress_cb, **_kw):
            progress_cb(30, "Referans görselle API yeniden yaratımı çalıştırılıyor...")
            g_ok, g_res = self.api_bridge.generate_image(
                prompt=prompt_text,
                out_dir=out_dir,
                aspect=aspect,
                filename_prefix="mechi_yeniden_yaratim",
                settings=cfg,
                reference_image_path=ref_path,
                system_instruction=sys_inst,
            )
            if not g_ok:
                raise RuntimeError(g_res)
            progress_cb(100, "Düzeltilmiş görsel hazır.")
            return g_res

        self.btn_analyze.setEnabled(False)
        self.job_started.emit("API ile görsel düzeltiliyor...")
        self.mascot_event.emit("busy")
        w = TaskWorker(_job)
        w.progress.connect(self.job_progress.emit)
        w.succeeded.connect(self._on_img_generated)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        self._worker = w
        w.start()

    def _on_img_generated(self, out_path: str) -> None:
        self.btn_analyze.setEnabled(True)
        self._last_generated_path = str(out_path)
        pix = QPixmap(self._last_generated_path)
        if not pix.isNull():
            self.lbl_out_preview.setPixmap(
                pix.scaled(172, 172, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        self.lbl_out_path.setText(self._last_generated_path)
        self.card_img.setVisible(True)
        self.mascot_event.emit("success")
        self.job_finished.emit(True, "Düzeltilmiş görsel üretildi.")

    def _on_failed(self, err: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "Analiz / düzeltme başarısız.")
        QMessageBox.warning(self, "Görsel Düzeltme Hatası", err)

    def _on_cancelled(self) -> None:
        self.btn_analyze.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "İşlem iptal edildi.")

    def _save_to_downloads(self) -> None:
        ok, res = save_to_downloads(self._last_generated_path, reveal_in_explorer=True)
        if not ok:
            QMessageBox.warning(self, "İndirme Hatası", res)
            return
        self.btn_download.setText("İndirilenlerde Gösterildi ✓")
        QTimer.singleShot(2000, lambda: self.btn_download.setText("İndirilenlere İndir"))

    def _open_in_ps(self) -> None:
        if not self._last_generated_path:
            return
        ok, msg = self.ps_bridge.open_file(self._last_generated_path)
        if not ok:
            QMessageBox.warning(self, "Photoshop Hatası", msg)

    def cancel_active_job(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
