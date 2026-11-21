"""Emprime Ayir Modulu — Ferah 3 sekmeli LAP + Dot-on-Dot serigrafi arayuzu."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mechi.config import TOOLS_DIR, load_settings, save_to_downloads
from mechi.core.module_registry import BaseToolModule
from mechi.core.worker import TaskCancelled, TaskWorker


def _derive_short_job_name(raw_name: str) -> str:
    if not raw_name:
        return "desen"
    stem = os.path.splitext(os.path.basename(raw_name))[0].lower()
    stem = re.sub(r"(calisma|çalışma|tasarim|tasarım|baski|baskı|desen)", "", stem)
    stem = stem.translate(str.maketrans("çğıöşü", "cgiosu"))
    clean = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return clean or "desen"


def _tr_ascii(s: str) -> str:
    return (s or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")).replace('"', "")


class EmprimeModule(BaseToolModule):
    """Mevcut LAP + Dot-on-Dot serigrafi spot kanal ayirma motorunun modern arayuzu."""

    module_id = "emprime"
    title = "Emprime Ayır"

    def __init__(self, context: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(context, parent)
        self.ps_bridge = context["ps_bridge"]
        self._active_doc: Optional[dict] = None
        self._src_dims: Optional[tuple[int, int]] = None
        self._last_output_path: str = ""
        self._worker: Optional[TaskWorker] = None
        self.setAcceptDrops(True)
        self._build_ui()
        self._update_size_preview()

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
        col.setMinimumWidth(680)
        col.setMaximumWidth(840)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 4, 0, 12)
        lay.setSpacing(14)

        lbl_title = QLabel("Emprime Renk Ayrımı")
        lbl_title.setObjectName("SectionTitle")
        lay.addWidget(lbl_title)

        tabs = QTabWidget()

        # --- TAB 1: Kaynak ve Renk ---
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.setContentsMargins(14, 14, 14, 14)
        t1.setSpacing(14)

        self.rb_ps = QRadioButton("Photoshop'ta açık belge (Açık belge yok)")
        self.rb_file = QRadioButton("Bilgisayardan seç (JPEG, PNG, TIF, PSD)")
        self.rb_ps.setEnabled(False)
        self.rb_file.setChecked(True)
        self.rb_ps.toggled.connect(self._on_source_changed)
        self.rb_file.toggled.connect(self._on_source_changed)
        t1.addWidget(self.rb_ps)
        t1.addWidget(self.rb_file)

        f_row = QHBoxLayout()
        f_row.setSpacing(8)
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Dosya seçin veya sürükleyin...")
        self.txt_path.textChanged.connect(self._on_file_text_changed)
        btn_br = QPushButton("Seç...")
        btn_br.clicked.connect(self._browse_file)
        f_row.addWidget(self.txt_path, 1)
        f_row.addWidget(btn_br)
        t1.addLayout(f_row)

        gf = QFormLayout()
        gf.setSpacing(10)

        self.cmb_kumas = QComboBox()
        self.cmb_kumas.addItem("İkisi Birlikte (Koyu + Açık Tişört)", "ikisi")
        self.cmb_kumas.addItem("Yalnızca Koyu Tişört (Beyaz LAP fon + tram)", "koyu")
        self.cmb_kumas.addItem("Yalnızca Açık Tişört (Fonsuz üst tram)", "acik")
        gf.addRow("Tişört Zemini:", self.cmb_kumas)

        prof_row = QHBoxLayout()
        prof_row.setSpacing(8)
        self.cmb_profil = QComboBox()
        self.cmb_profil.addItem("Profesyonel Tam Ayrım (Önerilen)", "profesyonel")
        self.cmb_profil.addItem("Ekonomik 6 Kafa", "ekonomik")
        self.cmb_profil.addItem("Özel Kafa Sayısı (5–10)", "ozel")
        self.cmb_profil.currentIndexChanged.connect(
            lambda: self.spn_head.setEnabled(self.cmb_profil.currentData() == "ozel")
        )
        self.spn_head = QSpinBox()
        self.spn_head.setRange(5, 10)
        self.spn_head.setValue(8)
        self.spn_head.setEnabled(False)
        self.spn_head.setSuffix(" kafa")
        prof_row.addWidget(self.cmb_profil, 1)
        prof_row.addWidget(self.spn_head)
        gf.addRow("Ayrım Profili:", prof_row)

        self.cmb_siyah = QComboBox()
        self.cmb_siyah.addItem("CMYK K Kanalı (FOGRA39)", "cmyk")
        self.cmb_siyah.addItem("Lab L Eğrisi", "lab")
        gf.addRow("Siyah Kanalı:", self.cmb_siyah)

        merge_row = QHBoxLayout()
        merge_row.setSpacing(6)
        for pct in (3, 5, 7, 10, 15):
            b = QPushButton(f"%{pct}")
            b.setFixedWidth(54)
            b.setStyleSheet("padding: 5px 6px;")
            b.clicked.connect(lambda _=False, v=pct: self.spn_merge.setValue(float(v)))
            merge_row.addWidget(b)
        self.spn_merge = QDoubleSpinBox()
        self.spn_merge.setRange(0.0, 30.0)
        self.spn_merge.setSingleStep(1.0)
        self.spn_merge.setValue(5.0)
        self.spn_merge.setSuffix(" %")
        merge_row.addWidget(self.spn_merge)
        merge_row.addStretch(1)
        gf.addRow("Yakın Renk Birleştirme:", merge_row)
        t1.addLayout(gf)
        t1.addStretch(1)
        tabs.addTab(tab1, "Kaynak ve Renk")

        # --- TAB 2: Olcu ve Tram ---
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.setContentsMargins(14, 14, 14, 14)
        t2.setSpacing(12)

        gsz = QFormLayout()
        gsz.setSpacing(10)

        dim_row = QHBoxLayout()
        dim_row.setSpacing(6)
        self.spn_width = QDoubleSpinBox()
        self.spn_width.setRange(2.0, 120.0)
        self.spn_width.setValue(28.0)
        self.spn_width.setSuffix(" cm")
        self.spn_width.valueChanged.connect(self._update_size_preview)
        for wcm in (24, 28, 32):
            bw = QPushButton(f"{wcm} cm")
            bw.setFixedWidth(60)
            bw.setStyleSheet("padding: 5px 6px;")
            bw.clicked.connect(lambda _=False, val=wcm: self.spn_width.setValue(float(val)))
            dim_row.addWidget(bw)
        dim_row.addWidget(self.spn_width)
        dim_row.addWidget(QLabel("PPI:"))
        self.spn_ppi = QSpinBox()
        self.spn_ppi.setRange(150, 1200)
        self.spn_ppi.setValue(508)
        self.spn_ppi.valueChanged.connect(self._update_size_preview)
        dim_row.addWidget(self.spn_ppi)
        gsz.addRow("En ve PPI:", dim_row)

        self.lbl_size_calc = QLabel("")
        self.lbl_size_calc.setObjectName("MutedText")
        gsz.addRow("Boy Hesabı:", self.lbl_size_calc)

        self.cmb_bg_mask = QComboBox()
        self.cmb_bg_mask.addItem("Otomatik (Metrajda tüm tuval, tekilde maskeli)", "oto")
        self.cmb_bg_mask.addItem("Dış beyaz zemin maskeli", "maskeli")
        self.cmb_bg_mask.addItem("Tüm tuval basılsın (maske yok)", "tumu")
        gsz.addRow("Zemin Maskesi:", self.cmb_bg_mask)

        tr_row = QHBoxLayout()
        tr_row.setSpacing(8)
        self.cmb_lpi = QComboBox()
        self.cmb_lpi.addItem("45 LPI (Spot Emprime)", 45.0)
        self.cmb_lpi.addItem("55 LPI (CMYK Fotoğraf)", 55.0)
        self.cmb_aci = QComboBox()
        self.cmb_aci.addItem("15° (Standart)", 15.0)
        self.cmb_aci.addItem("45°", 45.0)
        self.cmb_aci.addItem("72° (Usta Tercihi)", 72.0)
        self.cmb_choke = QComboBox()
        self.cmb_choke.addItem("2 px Choke", 2)
        self.cmb_choke.addItem("3 px Choke", 3)
        tr_row.addWidget(self.cmb_lpi)
        tr_row.addWidget(self.cmb_aci)
        tr_row.addWidget(self.cmb_choke)
        gsz.addRow("Tram ve Choke:", tr_row)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(14)
        self.chk_acma = QCheckBox("%5 Baskı Açma")
        self.chk_acma.setChecked(True)
        self.chk_degrade = QCheckBox("%85 Degrade Alt Bindirme")
        self.chk_degrade.setChecked(True)
        chk_row.addWidget(self.chk_acma)
        chk_row.addWidget(self.chk_degrade)
        chk_row.addStretch(1)
        gsz.addRow("Telafi:", chk_row)

        t2.addLayout(gsz)
        t2.addStretch(1)
        tabs.addTab(tab2, "Ölçü ve Tram")

        # --- TAB 3: Etiket ve Cikti ---
        tab3 = QWidget()
        t3 = QVBoxLayout(tab3)
        t3.setContentsMargins(14, 14, 14, 14)
        t3.setSpacing(12)

        go = QFormLayout()
        go.setSpacing(10)

        self.chk_kros = QCheckBox("+ Kayıt Krosu ve 4 Alanlı Etiket Ekle")
        self.chk_kros.setChecked(True)
        self.chk_kros.toggled.connect(self._on_kros_toggled)
        go.addRow("", self.chk_kros)

        lbl_row = QHBoxLayout()
        lbl_row.setSpacing(8)
        self.txt_beden = QLineEdit("s-xl")
        self.txt_yuz = QLineEdit("on baski")
        self.txt_isno = QLineEdit("703")
        self.txt_isad = QLineEdit("desen")
        for lbl, edit in (("Beden", self.txt_beden), ("Yüz", self.txt_yuz), ("İş No", self.txt_isno), ("İş Adı", self.txt_isad)):
            lbl_row.addWidget(QLabel(lbl + ":"))
            lbl_row.addWidget(edit)
        go.addRow("Etiket Bilgisi:", lbl_row)

        self.cmb_fmt = QComboBox()
        self.cmb_fmt.addItem("Tek PSD Dosyası (3 Katman + Spot Kanallar)", "psd")
        self.cmb_fmt.addItem("Tek TIF Dosyası (Spot Kanallı TIFF)", "tif")
        self.cmb_fmt.addItem("Yalnızca Photoshop'ta Açık Çalışma (Geçici)", "memory")
        self.cmb_fmt.currentIndexChanged.connect(
            lambda: self.txt_out_dir.setEnabled(self.cmb_fmt.currentData() != "memory")
        )
        go.addRow("Çıktı Biçimi:", self.cmb_fmt)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        cfg = load_settings()
        self.txt_out_dir = QLineEdit(cfg.get("output_dir") or "")
        btn_out = QPushButton("Seç...")
        btn_out.clicked.connect(self._browse_out_dir)
        out_row.addWidget(self.txt_out_dir, 1)
        out_row.addWidget(btn_out)
        go.addRow("Hedef Klasör:", out_row)

        t3.addLayout(go)
        t3.addStretch(1)
        tabs.addTab(tab3, "Etiket ve Çıktı")

        lay.addWidget(tabs)

        self.btn_run = QPushButton("Emprime Ayrımını Başlat")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setMinimumHeight(46)
        self.btn_run.clicked.connect(self.run_separation)
        lay.addWidget(self.btn_run)

        self.card_done = QFrame()
        self.card_done.setObjectName("CleanSection")
        self.card_done.setVisible(False)
        dl = QHBoxLayout(self.card_done)
        dl.setContentsMargins(16, 14, 16, 14)
        self.lbl_done = QLabel("")
        self.lbl_done.setWordWrap(True)
        dl.addWidget(self.lbl_done, 1)
        self.btn_download = QPushButton("İndirilenlere İndir")
        self.btn_download.clicked.connect(self._save_to_downloads)
        dl.addWidget(self.btn_download)
        btn_open_ps = QPushButton("Photoshop'ta Aç")
        btn_open_ps.clicked.connect(self._open_in_ps)
        dl.addWidget(btn_open_ps)
        lay.addWidget(self.card_done)
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
            self.txt_isad.setText(_derive_short_job_name(file_path))
            parent = os.path.dirname(file_path)
            if parent and os.path.isdir(parent):
                self.txt_out_dir.setText(parent)

    def on_ps_context_changed(self, doc_info: Optional[dict]) -> None:
        self._active_doc = doc_info
        if doc_info:
            self.rb_ps.setEnabled(True)
            self.rb_ps.setText(
                f"Photoshop'ta açık belge: {doc_info['name']} ({doc_info.get('width_px', 0)}x{doc_info.get('height_px', 0)} px)"
            )
        else:
            self.rb_ps.setEnabled(False)
            self.rb_ps.setText("Photoshop'ta açık belge (Açık belge yok)")
            if self.rb_ps.isChecked():
                self.rb_file.setChecked(True)
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        if self.rb_ps.isChecked() and self._active_doc:
            w = int(self._active_doc.get("width_px") or 0)
            h = int(self._active_doc.get("height_px") or 0)
            self._src_dims = (w, h) if w > 0 and h > 0 else None
            self.txt_isad.setText(_derive_short_job_name(self._active_doc.get("name", "desen")))
            doc_path = self._active_doc.get("path") or ""
            if doc_path and os.path.isdir(os.path.dirname(doc_path)):
                self.txt_out_dir.setText(os.path.dirname(doc_path))
        else:
            self._on_file_text_changed()
        self._update_size_preview()

    def _on_file_text_changed(self) -> None:
        if not self.rb_file.isChecked():
            return
        p = self.txt_path.text().strip()
        if p and os.path.exists(p):
            try:
                with Image.open(p) as im:
                    self._src_dims = im.size
            except Exception:
                self._src_dims = None
        else:
            self._src_dims = None
        self._update_size_preview()

    def _on_kros_toggled(self, checked: bool) -> None:
        for w in (self.txt_beden, self.txt_yuz, self.txt_isno, self.txt_isad):
            w.setEnabled(checked)
        self._update_size_preview()

    def _update_size_preview(self) -> None:
        en_cm = self.spn_width.value()
        ppi = self.spn_ppi.value()
        target_w = int(round(en_cm / 2.54 * ppi))
        if self._src_dims and self._src_dims[0] > 0:
            sw, sh = self._src_dims
            target_h = int(round(sh * (target_w / float(sw))))
            boy_cm = en_cm * (sh / float(sw))
            strip_txt = f" (+3.0 cm etiket = {boy_cm + 3.0:.1f} cm)" if self.chk_kros.isChecked() else ""
            self.lbl_size_calc.setText(f"{boy_cm:.1f} cm ({target_w} x {target_h} px @ {ppi} PPI){strip_txt}")
        else:
            self.lbl_size_calc.setText(f"Görsel oranına göre hesaplanacak (En: {target_w} px @ {ppi} PPI)")

    def _browse_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Emprime Ayrımı Yapılacak Dosya",
            "",
            "Görseller ve PSD (*.jpg *.jpeg *.jfif *.png *.tif *.tiff *.psd);;Tüm Dosyalar (*.*)",
        )
        if f:
            self.receive_file(f, "browse")

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seçin", self.txt_out_dir.text())
        if d:
            self.txt_out_dir.setText(d)

    def run_separation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Meşgul", "Devam eden emprime ayrımı bitmeden yenisi başlatılamaz.")
            return

        py_script = os.path.join(TOOLS_DIR, "lap_aile_ayir.py")
        if not os.path.exists(py_script):
            QMessageBox.critical(self, "Eksik Dosya", f"Ayrım motoru bulunamadı: {py_script}")
            return

        use_ps = self.rb_ps.isChecked() and self._active_doc is not None
        in_file = self.txt_path.text().strip()
        if not use_ps and (not in_file or not os.path.exists(in_file)):
            QMessageBox.warning(self, "Kaynak Eksik", "Lütfen geçerli bir görsel seçin veya açık PS belgesi kullanın.")
            return

        out_mode = self.cmb_fmt.currentData()
        out_fmt = "tif" if out_mode == "tif" else "psd"
        kumas = self.cmb_kumas.currentData()
        prof_sel = self.cmb_profil.currentData()
        profil = "ekonomik" if prof_sel == "ekonomik" else "profesyonel"
        renk_param = None
        if prof_sel == "ekonomik":
            renk_param = 6 if kumas == "acik" else 5
        elif prof_sel == "ozel":
            heads = self.spn_head.value()
            renk_param = heads if kumas == "acik" else max(2, heads - 1)

        is_ad = _derive_short_job_name(self.txt_isad.text())
        out_dir = self.txt_out_dir.text().strip() or (load_settings().get("output_dir") or os.getcwd())
        exp_doc_name = self._active_doc["name"] if (use_ps and self._active_doc) else None

        params = {
            "py_script": py_script,
            "use_ps": use_ps,
            "exp_doc_name": exp_doc_name,
            "in_file": in_file,
            "out_mode": out_mode,
            "out_fmt": out_fmt,
            "out_dir": out_dir,
            "kumas": kumas,
            "profil": profil,
            "renk": renk_param,
            "siyah_mod": self.cmb_siyah.currentData(),
            "genislik_cm": self.spn_width.value(),
            "ppi": self.spn_ppi.value(),
            "zemin_maske": self.cmb_bg_mask.currentData(),
            "lpi": self.cmb_lpi.currentData(),
            "aci": self.cmb_aci.currentData(),
            "choke": self.cmb_choke.currentData(),
            "baski_aciklama": 0.05 if self.chk_acma.isChecked() else 0.0,
            "degrade_bindirme": 0.85 if self.chk_degrade.isChecked() else 0.0,
            "renk_birlestirme": self.spn_merge.value(),
            "kros_etiket": self.chk_kros.isChecked(),
            "beden": _tr_ascii(self.txt_beden.text()) or "s-xl",
            "yuz": _tr_ascii(self.txt_yuz.text()) or "on baski",
            "is_no": _tr_ascii(self.txt_isno.text()) or "703",
            "is_ad": is_ad,
        }

        def _job(progress_cb, log_cb, cancel_event, attach_process, **_kw):
            tmp_dir = tempfile.mkdtemp(prefix="mechi_emp_")
            try:
                if params["use_ps"]:
                    progress_cb(8, "Photoshop'tan açık belge dışa aktarılıyor...")
                    src_img = os.path.join(tmp_dir, "kaynak.png")
                    ok_ex, msg_ex = self.ps_bridge.export_active_doc_png(
                        src_img, expected_doc_name=params["exp_doc_name"]
                    )
                    if not ok_ex:
                        raise RuntimeError(msg_ex)
                else:
                    src_img = params["in_file"]

                tmp_out = os.path.join(tmp_dir, f"cikti.{params['out_fmt']}")
                cmd = [
                    sys.executable,
                    "-u",
                    params["py_script"],
                    src_img,
                    tmp_dir,
                    "--tek-dosya",
                    "--format",
                    params["out_fmt"],
                    "--cikti-dosya",
                    tmp_out,
                    "--kumas",
                    params["kumas"],
                    "--profil",
                    params["profil"],
                    "--siyah-mod",
                    params["siyah_mod"],
                    "--genislik-cm",
                    str(params["genislik_cm"]),
                    "--ppi",
                    str(params["ppi"]),
                    "--zemin-maske",
                    params["zemin_maske"],
                    "--lpi",
                    str(params["lpi"]),
                    "--aci",
                    str(params["aci"]),
                    "--choke",
                    str(params["choke"]),
                    "--baski-aciklama",
                    str(params["baski_aciklama"]),
                    "--degrade-bindirme",
                    str(params["degrade_bindirme"]),
                    "--renk-birlestirme",
                    str(params["renk_birlestirme"]),
                    "--kros-etiket" if params["kros_etiket"] else "--kros-yok",
                    "--beden",
                    params["beden"],
                    "--yuz",
                    params["yuz"],
                    "--is-no",
                    params["is_no"],
                    "--is-ad",
                    params["is_ad"],
                ]
                if params["renk"] is not None:
                    cmd.extend(["--renk", str(params["renk"])])

                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                progress_cb(15, "Emprime ayrım motoru başlatıldı...")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                attach_process(proc)
                logs = []
                pct = 20
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if cancel_event.is_set():
                        proc.terminate()
                        raise TaskCancelled()
                    line = raw_line.strip()
                    if line:
                        logs.append(line)
                        log_cb(line)
                        pct = min(92, pct + 3)
                        progress_cb(pct, line[:95])
                rc = proc.wait()
                attach_process(None)
                if cancel_event.is_set():
                    raise TaskCancelled()
                if rc != 0 or not os.path.exists(tmp_out):
                    tail = "\n".join(logs[-15:])
                    raise RuntimeError(f"Ayrım motoru hata verdi (kod {rc}):\n{tail}")

                os.makedirs(params["out_dir"], exist_ok=True)
                final_path = os.path.join(params["out_dir"], f"{params['is_ad']}_emprime.{params['out_fmt']}")
                if params["out_mode"] == "memory":
                    progress_cb(95, "Photoshop'ta açılıyor...")
                    self.ps_bridge.open_file(tmp_out)
                    return tmp_out

                shutil.copy2(tmp_out, final_path)
                progress_cb(96, "Photoshop'ta açılıyor...")
                self.ps_bridge.open_file(final_path)
                progress_cb(100, "Emprime dosyası hazır.")
                return final_path
            finally:
                if params["out_mode"] != "memory":
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        self.btn_run.setEnabled(False)
        self.job_started.emit("Emprime ayrımı çalışıyor...")
        self.mascot_event.emit("busy")
        w = TaskWorker(fn=_job)
        w.progress.connect(self.job_progress.emit)
        w.succeeded.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        self._worker = w
        w.start()

    def _on_done(self, out_path: str) -> None:
        self.btn_run.setEnabled(True)
        self._last_output_path = str(out_path)
        self.lbl_done.setText(f"✓ Emprime dosyası hazır: {self._last_output_path}")
        self.card_done.setVisible(True)
        self.mascot_event.emit("success")
        self.job_finished.emit(True, "Emprime ayrımı tamamlandı.")

    def _on_failed(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "Emprime ayrımı başarısız.")
        QMessageBox.warning(self, "Emprime Hatası", err)

    def _on_cancelled(self) -> None:
        self.btn_run.setEnabled(True)
        self.mascot_event.emit("idle")
        self.job_finished.emit(False, "Emprime ayrımı iptal edildi.")

    def _save_to_downloads(self) -> None:
        ok, res = save_to_downloads(self._last_output_path, reveal_in_explorer=True)
        if not ok:
            QMessageBox.warning(self, "İndirme Hatası", res)
            return
        self.btn_download.setText("İndirilenlerde Gösterildi ✓")
        QTimer.singleShot(2000, lambda: self.btn_download.setText("İndirilenlere İndir"))

    def _open_in_ps(self) -> None:
        if not self._last_output_path or not os.path.exists(self._last_output_path):
            return
        ok, msg = self.ps_bridge.open_file(self._last_output_path)
        if not ok:
            QMessageBox.warning(self, "Photoshop Hatası", msg)

    def cancel_active_job(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
