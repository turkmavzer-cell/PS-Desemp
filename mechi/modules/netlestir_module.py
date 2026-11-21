"""Netlestir Modulu — Sade kaynak secimi, temizlik seviyesi ve opsiyonel ince ayarlar."""
from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mechi.config import load_settings, save_to_downloads
from mechi.core.enhancer import PRESETS, enhance_psd_via_photoshop, enhance_single_file
from mechi.core.module_registry import BaseToolModule
from mechi.core.worker import TaskWorker


class NetlestirModule(BaseToolModule):
    """Tek gorsel veya cok katmanli PSD icin puruz, noise ve hale giderici."""

    module_id = "netlestir"
    title = "Netleştir"

    def __init__(self, context: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(context, parent)
        self.ps_bridge = context["ps_bridge"]
        self._active_doc: Optional[dict] = None
        self._last_output_path: str = ""
        self._worker: Optional[TaskWorker] = None
        self._preset_chips: Dict[str, QPushButton] = {}
        self.setAcceptDrops(True)
        self._build_ui()
        self._update_mode_banner()

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

        lbl_title = QLabel("Görsel ve Katman Netleştirme")
        lbl_title.setObjectName("SectionTitle")
        lay.addWidget(lbl_title)

        # 1) Kaynak Karti
        sec_src = QFrame()
        sec_src.setObjectName("CleanSection")
        s_lay = QVBoxLayout(sec_src)
        s_lay.setContentsMargins(18, 16, 18, 16)
        s_lay.setSpacing(10)

        lbl_s = QLabel("Kaynak Dosya")
        lbl_s.setObjectName("SectionLabel")
        s_lay.addWidget(lbl_s)

        self.rb_ps = QRadioButton("Photoshop'ta açık belge (Açık belge yok)")
        self.rb_file = QRadioButton("Bilgisayardan seç (JPEG, PNG, TIF, PSD)")
        self.rb_file.setChecked(True)
        self.rb_ps.setEnabled(False)
        self.rb_ps.toggled.connect(self._update_mode_banner)
        self.rb_file.toggled.connect(self._update_mode_banner)
        s_lay.addWidget(self.rb_ps)
        s_lay.addWidget(self.rb_file)

        f_row = QHBoxLayout()
        f_row.setSpacing(8)
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Dosyayı sürükleyin veya seçin...")
        self.txt_path.textChanged.connect(self._update_mode_banner)
        btn_browse = QPushButton("Seç...")
        btn_browse.clicked.connect(self._browse_file)
        f_row.addWidget(self.txt_path, 1)
        f_row.addWidget(btn_browse)
        s_lay.addLayout(f_row)

        self.lbl_mode_info = QLabel("")
        self.lbl_mode_info.setObjectName("SubtleHint")
        self.lbl_mode_info.setWordWrap(True)
        s_lay.addWidget(self.lbl_mode_info)

        lay.addWidget(sec_src)

        # 2) Temizlik Seviyesi ve Baski Hedefi Karti
        sec_opt = QFrame()
        sec_opt.setObjectName("CleanSection")
        o_lay = QVBoxLayout(sec_opt)
        o_lay.setContentsMargins(18, 16, 18, 16)
        o_lay.setSpacing(12)

        top_o = QHBoxLayout()
        lbl_o = QLabel("Temizlik Seviyesi ve Baskı Hedefi")
        lbl_o.setObjectName("SectionLabel")
        top_o.addWidget(lbl_o)
        top_o.addStretch(1)
        self.btn_adv_toggle = QPushButton("İnce Ayarları Göster")
        self.btn_adv_toggle.setCheckable(True)
        self.btn_adv_toggle.toggled.connect(self._toggle_advanced)
        top_o.addWidget(self.btn_adv_toggle)
        o_lay.addLayout(top_o)

        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)

        chip_row1 = QHBoxLayout()
        chip_row1.setSpacing(10)
        for pid, text in (
            ("hafif", "Hafif (Ham Doku & Suluboya)"),
            ("dengeli", "Detay Korumalı 4K Netleştirme"),
            ("guclu", "Bozuk / JPEG Piksel Onarım"),
        ):
            chip = QPushButton(text)
            chip.setObjectName("PresetChip")
            chip.setCheckable(True)
            chip.clicked.connect(lambda _=False, p=pid: self._apply_preset(p))
            self._chip_group.addButton(chip)
            self._preset_chips[pid] = chip
            chip_row1.addWidget(chip, 1)
        o_lay.addLayout(chip_row1)

        chip_row2 = QHBoxLayout()
        chip_row2.setSpacing(10)
        for pid, text in (
            ("vektor_yakin", "Yeniden Çizim & Vektör Netliği (Önerilen)"),
            ("vektor_keskin", "Düz Grafik / Spot Baskı"),
        ):
            chip = QPushButton(text)
            chip.setObjectName("PresetChip")
            chip.setCheckable(True)
            if pid == "vektor_yakin":
                chip.setChecked(True)
            chip.clicked.connect(lambda _=False, p=pid: self._apply_preset(p))
            self._chip_group.addButton(chip)
            self._preset_chips[pid] = chip
            chip_row2.addWidget(chip, 1)
        o_lay.addLayout(chip_row2)

        # Dogrudan gorunur 4K Cozunurluk ve 254 DPI Baski Hedefi secicileri
        target_row = QHBoxLayout()
        target_row.setSpacing(14)

        up_col = QVBoxLayout()
        up_col.setSpacing(4)
        up_col.addWidget(QLabel("Çıktı Ölçeği:"))
        self.cmb_upscale = QComboBox()
        self.cmb_upscale.addItem("4K Baskı Hedefi (3840 px)", "4k")
        self.cmb_upscale.addItem("4K Ultra (4096 px)", "4k_ultra")
        self.cmb_upscale.addItem("2x Süper Çözünürlük", "2x")
        self.cmb_upscale.addItem("Orijinal Ölçek (Büyütme Yok)", "none")
        self.cmb_upscale.setCurrentIndex(0)
        up_col.addWidget(self.cmb_upscale)
        target_row.addLayout(up_col, 1)

        dpi_col = QVBoxLayout()
        dpi_col.setSpacing(4)
        dpi_col.addWidget(QLabel("Baskı Çözünürlüğü (DPI):"))
        self.cmb_dpi = QComboBox()
        self.cmb_dpi.addItem("254 DPI (100 px/cm — Tekstil Standart)", 254)
        self.cmb_dpi.addItem("508 DPI (200 px/cm — Emprime Kalıp)", 508)
        self.cmb_dpi.addItem("300 DPI (Dijital / Ofset)", 300)
        self.cmb_dpi.addItem("Orijinal DPI Koru", 0)
        self.cmb_dpi.setCurrentIndex(0)
        dpi_col.addWidget(self.cmb_dpi)
        target_row.addLayout(dpi_col, 1)

        o_lay.addLayout(target_row)

        # Varsayilan gizli ince ayarlar
        self.adv_panel = QWidget()
        self.adv_panel.setVisible(False)
        adv_l = QVBoxLayout(self.adv_panel)
        adv_l.setContentsMargins(0, 8, 0, 0)
        adv_l.setSpacing(10)

        dn_row = QHBoxLayout()
        dn_row.addWidget(QLabel("Gürültü Giderme:"))
        self.sld_denoise = QSlider(Qt.Orientation.Horizontal)
        self.sld_denoise.setRange(0, 100)
        self.sld_denoise.setValue(55)
        self.lbl_denoise_val = QLabel("55")
        self.lbl_denoise_val.setFixedWidth(30)
        self.sld_denoise.valueChanged.connect(lambda v: self.lbl_denoise_val.setText(str(v)))
        dn_row.addWidget(self.sld_denoise, 1)
        dn_row.addWidget(self.lbl_denoise_val)
        adv_l.addLayout(dn_row)

        vc_row = QHBoxLayout()
        vc_row.addWidget(QLabel("Düz Alan Pürüzsüzleştirme (Detay Korumalı):"))
        self.sld_vector_clean = QSlider(Qt.Orientation.Horizontal)
        self.sld_vector_clean.setRange(0, 100)
        self.sld_vector_clean.setValue(35)
        self.lbl_vector_clean_val = QLabel("35")
        self.lbl_vector_clean_val.setFixedWidth(30)
        self.sld_vector_clean.valueChanged.connect(lambda v: self.lbl_vector_clean_val.setText(str(v)))
        vc_row.addWidget(self.sld_vector_clean, 1)
        vc_row.addWidget(self.lbl_vector_clean_val)
        adv_l.addLayout(vc_row)

        ec_row = QHBoxLayout()
        ec_row.addWidget(QLabel("Kontur Sıkılaştırma (Piksel Kırığı Topla):"))
        self.sld_edge_crisp = QSlider(Qt.Orientation.Horizontal)
        self.sld_edge_crisp.setRange(0, 100)
        self.sld_edge_crisp.setValue(80)
        self.lbl_edge_crisp_val = QLabel("80")
        self.lbl_edge_crisp_val.setFixedWidth(30)
        self.sld_edge_crisp.valueChanged.connect(lambda v: self.lbl_edge_crisp_val.setText(str(v)))
        ec_row.addWidget(self.sld_edge_crisp, 1)
        ec_row.addWidget(self.lbl_edge_crisp_val)
        adv_l.addLayout(ec_row)

        sh_row = QHBoxLayout()
        sh_row.addWidget(QLabel("Kenar Netleştirme:"))
        self.sld_sharpen = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharpen.setRange(0, 100)
        self.sld_sharpen.setValue(65)
        self.lbl_sharpen_val = QLabel("65")
        self.lbl_sharpen_val.setFixedWidth(30)
        self.sld_sharpen.valueChanged.connect(lambda v: self.lbl_sharpen_val.setText(str(v)))
        sh_row.addWidget(self.sld_sharpen, 1)
        sh_row.addWidget(self.lbl_sharpen_val)
        adv_l.addLayout(sh_row)

        self.chk_despeckle = QCheckBox("İzole mikro leke ve kırık pikselleri temizle")
        self.chk_despeckle.setChecked(True)
        adv_l.addWidget(self.chk_despeckle)

        self.chk_deblock = QCheckBox("JPEG blok izlerini ve kenar halelerini (halo) önle")
        self.chk_deblock.setChecked(True)
        adv_l.addWidget(self.chk_deblock)

        o_lay.addWidget(self.adv_panel)
        lay.addWidget(sec_opt)

        self._apply_preset("vektor_yakin")

        # Baslat
        self.btn_run = QPushButton("Netleştirmeyi Başlat")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setMinimumHeight(46)
        self.btn_run.clicked.connect(self.run_enhance)
        lay.addWidget(self.btn_run)

        # Sonuc Karti
        self.card_done = QFrame()
        self.card_done.setObjectName("CleanSection")
        self.card_done.setVisible(False)
        d_lay = QVBoxLayout(self.card_done)
        d_lay.setContentsMargins(18, 16, 18, 16)
        d_lay.setSpacing(10)

        self.lbl_done_title = QLabel("✓ Netleştirme tamamlandı")
        self.lbl_done_title.setObjectName("CardTitle")
        d_lay.addWidget(self.lbl_done_title)

        self.lbl_done_info = QLabel("")
        self.lbl_done_info.setObjectName("MutedText")
        self.lbl_done_info.setWordWrap(True)
        d_lay.addWidget(self.lbl_done_info)

        h_act = QHBoxLayout()
        h_act.setSpacing(10)
        self.btn_download = QPushButton("İndirilenlere İndir")
        self.btn_download.clicked.connect(self._save_to_downloads)
        btn_ps = QPushButton("Photoshop'ta Aç")
        btn_ps.clicked.connect(self._open_in_ps)
        btn_emp = QPushButton("Emprimeye Gönder")
        btn_emp.setObjectName("PrimaryButton")
        btn_emp.clicked.connect(lambda: self.handoff_requested.emit("emprime", self._last_output_path))
        h_act.addWidget(self.btn_download)
        h_act.addWidget(btn_ps)
        h_act.addWidget(btn_emp)
        h_act.addStretch(1)
        d_lay.addLayout(h_act)

        lay.addWidget(self.card_done)
        lay.addStretch(1)

        wrap_h.addWidget(col, 10)
        wrap_h.addStretch(1)

    def _toggle_advanced(self, checked: bool) -> None:
        self.adv_panel.setVisible(checked)
        self.btn_adv_toggle.setText("İnce Ayarları Gizle" if checked else "İnce Ayarları Göster")

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
            self._update_mode_banner()

    def on_ps_context_changed(self, doc_info: Optional[dict]) -> None:
        self._active_doc = doc_info
        if doc_info:
            lyr_txt = f"{doc_info.get('layer_count', 1)} katman"
            self.rb_ps.setEnabled(True)
            self.rb_ps.setText(f"Photoshop'ta açık belge: {doc_info['name']} ({lyr_txt})")
        else:
            self.rb_ps.setEnabled(False)
            self.rb_ps.setText("Photoshop'ta açık belge (Açık belge yok)")
            if self.rb_ps.isChecked():
                self.rb_file.setChecked(True)
        self._update_mode_banner()

    def _is_layered_mode(self) -> bool:
        if self.rb_ps.isChecked() and self._active_doc:
            name_low = (self._active_doc.get("name") or "").lower()
            return bool(self._active_doc.get("has_layers")) or name_low.endswith((".psd", ".psb"))
        p = self.txt_path.text().strip().lower()
        return p.endswith((".psd", ".psb"))

    def _update_mode_banner(self) -> None:
        layered = self._is_layered_mode()
        self.cmb_upscale.setEnabled(not layered)
        self.cmb_dpi.setEnabled(not layered)
        if layered:
            self.lbl_mode_info.setText(
                "Katmanlı PSD Modu — Her piksel katmanı ayrı temizlenir; yazı ve vektör katmanları korunur."
            )
        else:
            self.lbl_mode_info.setText(
                "Tek Dosya Modu — Orijinal dosya korunarak 4K / 254 DPI vektör temizliğinde yeni kopya üretilir."
            )

    def _apply_preset(self, preset_id: str) -> None:
        spec = PRESETS.get(preset_id)
        if not spec:
            return
        self.sld_denoise.setValue(int(spec.get("denoise", 55)))
        self.sld_sharpen.setValue(int(spec.get("sharpen", 65)))
        self.sld_vector_clean.setValue(int(spec.get("vector_clean", 75)))
        self.sld_edge_crisp.setValue(int(spec.get("edge_crisp", 80)))
        self.chk_despeckle.setChecked(bool(spec.get("despeckle", True)))
        self.chk_deblock.setChecked(bool(spec.get("deblock", True)))

    def _browse_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Netleştirilecek Dosyayı Seçin",
            "",
            "Görseller ve PSD (*.jpg *.jpeg *.png *.tif *.tiff *.psd *.psb);;Tüm Dosyalar (*.*)",
        )
        if f:
            self.rb_file.setChecked(True)
            self.txt_path.setText(f)

    def run_enhance(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Meşgul", "Devam eden netleştirme tamamlanmadan yenisi başlatılamaz.")
            return

        use_ps = self.rb_ps.isChecked() and self._active_doc is not None
        file_path = self.txt_path.text().strip()
        if not use_ps and (not file_path or not os.path.exists(file_path)):
            QMessageBox.warning(self, "Dosya Seçilmedi", "Lütfen geçerli bir dosya seçin veya açık PS belgesi kullanın.")
            return

        dpi_data = self.cmb_dpi.currentData()
        opts = {
            "denoise": self.sld_denoise.value(),
            "sharpen": self.sld_sharpen.value(),
            "vector_clean": self.sld_vector_clean.value(),
            "edge_crisp": self.sld_edge_crisp.value(),
            "despeckle": self.chk_despeckle.isChecked(),
            "deblock": self.chk_deblock.isChecked(),
            "upscale": self.cmb_upscale.currentData() or "4k",
            "target_dpi": int(dpi_data if dpi_data is not None else 254),
        }
        layered = self._is_layered_mode()
        cfg = load_settings()
        default_out_dir = cfg.get("output_dir") or os.getcwd()

        if layered:
            if use_ps and self._active_doc:
                src_psd = None
                exp_name = self._active_doc["name"]
                src_dir = os.path.dirname(self._active_doc.get("path") or "") or default_out_dir
                stem = os.path.splitext(exp_name)[0]
                out_psd = os.path.join(src_dir, f"{stem}_net.psd")
            else:
                src_psd = file_path
                exp_name = None
                stem = os.path.splitext(os.path.basename(file_path))[0]
                out_psd = os.path.join(os.path.dirname(file_path), f"{stem}_net.psd")

            def _job_psd(progress_cb, cancel_event, **_kw):
                return enhance_psd_via_photoshop(
                    self.ps_bridge,
                    src_psd_or_none=src_psd,
                    out_psd_path=out_psd,
                    options=opts,
                    expected_doc_name=exp_name,
                    progress_cb=progress_cb,
                    cancel_event=cancel_event,
                )

            self._launch(_job_psd, "Katmanlı PSD netleştiriliyor...")
            return

        def _job_single(progress_cb, cancel_event, **_kw):
            tmp_export = None
            in_path = file_path
            if use_ps and self._active_doc:
                exp_name = self._active_doc["name"]
                src_dir = os.path.dirname(self._active_doc.get("path") or "") or default_out_dir
                stem = os.path.splitext(exp_name)[0]
                tmp_export = os.path.join(default_out_dir, f"_tmp_ps_{stem}.png")
                progress_cb(5, "Photoshop belgesi dışa aktarılıyor...")
                ok_ex, msg_ex = self.ps_bridge.export_active_doc_png(tmp_export, expected_doc_name=exp_name)
                if not ok_ex:
                    raise RuntimeError(msg_ex)
                in_path = tmp_export
                out_path = os.path.join(src_dir, f"{stem}_net.png")
            else:
                stem, ext = os.path.splitext(os.path.basename(file_path))
                out_path = os.path.join(os.path.dirname(file_path), f"{stem}_net{ext}")

            try:
                return enhance_single_file(
                    in_path,
                    out_path,
                    options=opts,
                    progress_cb=progress_cb,
                    cancel_event=cancel_event,
                )
            finally:
                if tmp_export and os.path.exists(tmp_export):
                    try:
                        os.remove(tmp_export)
                    except Exception:
                        pass

        self._launch(_job_single, "Görsel netleştiriliyor...")

    def _launch(self, fn, title: str) -> None:
        self.btn_run.setEnabled(False)
        self.job_started.emit(title)
        self.mascot_event.emit("busy")
        w = TaskWorker(fn)
        w.progress.connect(self.job_progress.emit)
        w.succeeded.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        self._worker = w
        w.start()

    def _on_done(self, res: dict) -> None:
        self.btn_run.setEnabled(True)
        self._last_output_path = res.get("output_path", "")
        if res.get("mode") == "psd_layers":
            self.lbl_done_info.setText(
                f"Çıktı: {self._last_output_path}\n"
                f"Netleştirilen katman: {res.get('enhanced_layers', 0)}  /  "
                f"Korunan vektör/yazı: {res.get('skipped_vector_layers', 0)}"
            )
        else:
            self.lbl_done_info.setText(
                f"Çıktı: {self._last_output_path} "
                f"({res.get('width')}x{res.get('height')} px · {res.get('dpi', 254)} DPI · "
                f"{res.get('width_cm', 0)}x{res.get('height_cm', 0)} cm)"
            )
        self.card_done.setVisible(True)
        self.mascot_event.emit("success")
        self.job_finished.emit(True, "Netleştirme tamamlandı.")

    def _on_failed(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "Netleştirme başarısız.")
        QMessageBox.warning(self, "Netleştirme Hatası", err)

    def _on_cancelled(self) -> None:
        self.btn_run.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "Netleştirme iptal edildi.")

    def _save_to_downloads(self) -> None:
        ok, res = save_to_downloads(self._last_output_path, reveal_in_explorer=True)
        if not ok:
            QMessageBox.warning(self, "İndirme Hatası", res)
            return
        self.btn_download.setText("İndirilenlerde Gösterildi ✓")
        QTimer.singleShot(2000, lambda: self.btn_download.setText("İndirilenlere İndir"))

    def _open_in_ps(self) -> None:
        if not self._last_output_path:
            return
        ok, msg = self.ps_bridge.open_file(self._last_output_path)
        if not ok:
            QMessageBox.warning(self, "Photoshop Hatası", msg)

    def cancel_active_job(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
