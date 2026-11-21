"""MeChi Studio v1.1 Ayarlar penceresi — Sade 2 sekmeli duzen."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mechi.config import ENV_KEYS, load_env, load_settings, save_env_values, save_settings
from mechi.core.api_bridge import PROVIDERS


class SettingsDialog(QDialog):
    """API koprusu (yalniz secili saglayici anahtarlari) ve Genel ayarlar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ayarlar — MeChi Studio")
        self.setMinimumWidth(560)
        self._settings = load_settings()
        self._env_vals = load_env()
        self._key_inputs: Dict[str, QLineEdit] = {}
        self._key_rows: Dict[str, QWidget] = {}
        self._build_ui()
        self._load_values()

    def _make_key_row(self, key: str) -> QWidget:
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(8)
        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        edit.setPlaceholderText("API anahtarını girin...")
        btn_eye = QPushButton("Göster")
        btn_eye.setCheckable(True)
        btn_eye.setFixedWidth(74)
        btn_eye.toggled.connect(
            lambda checked, e=edit, b=btn_eye: (
                e.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password),
                b.setText("Gizle" if checked else "Göster"),
            )
        )
        row_l.addWidget(edit, 1)
        row_l.addWidget(btn_eye)
        self._key_inputs[key] = edit
        return row_w

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(18, 18, 18, 18)

        tabs = QTabWidget()

        # --- TAB 1: Gorsel API ---
        tab_api = QWidget()
        t_api = QVBoxLayout(tab_api)
        t_api.setContentsMargins(12, 12, 12, 12)
        t_api.setSpacing(14)

        form_api = QFormLayout()
        form_api.setSpacing(10)

        self.cmb_provider = QComboBox()
        for pid, spec in PROVIDERS.items():
            self.cmb_provider.addItem(spec.display_name, pid)
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        form_api.addRow("Görsel Sağlayıcı:", self.cmb_provider)

        self.lbl_provider_note = QLabel("")
        self.lbl_provider_note.setObjectName("MutedText")
        self.lbl_provider_note.setWordWrap(True)
        form_api.addRow("", self.lbl_provider_note)

        self.lbl_model_row = QLabel("Model Adı:")
        self.txt_model = QLineEdit()
        self.txt_model.setPlaceholderText("Varsayılan model için boş bırakın")
        form_api.addRow(self.lbl_model_row, self.txt_model)

        key_labels = {
            "CF_ACCOUNT_ID": "Cloudflare Account ID:",
            "CF_API_TOKEN": "Cloudflare API Token:",
            "HF_TOKEN": "Hugging Face Token:",
            "FAL_KEY": "fal.ai API Key:",
            "OPENAI_API_KEY": "OpenAI API Key:",
        }
        self._provider_key_labels: Dict[str, QLabel] = {}
        for key in ("CF_ACCOUNT_ID", "CF_API_TOKEN", "HF_TOKEN", "FAL_KEY", "OPENAI_API_KEY"):
            lbl = QLabel(key_labels[key])
            row_w = self._make_key_row(key)
            self._provider_key_labels[key] = lbl
            self._key_rows[key] = row_w
            form_api.addRow(lbl, row_w)

        t_api.addLayout(form_api)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2B3136;")
        t_api.addWidget(sep)

        lbl_gemini_title = QLabel("Stil Öğrenme ve Gemini Görsel Anahtarı")
        lbl_gemini_title.setObjectName("SectionLabel")
        t_api.addWidget(lbl_gemini_title)

        lbl_gemini_hint = QLabel(
            "Klasörden Tasarım Dili Öğrenme özelliği ücretsiz Google AI Studio Gemini anahtarıyla çalışır."
        )
        lbl_gemini_hint.setObjectName("MutedText")
        lbl_gemini_hint.setWordWrap(True)
        t_api.addWidget(lbl_gemini_hint)

        form_gem = QFormLayout()
        form_gem.addRow("GEMINI_API_KEY:", self._make_key_row("GEMINI_API_KEY"))
        t_api.addLayout(form_gem)
        t_api.addStretch(1)

        tabs.addTab(tab_api, "Görsel API")

        # --- TAB 2: Genel ---
        tab_gen = QWidget()
        t_gen = QVBoxLayout(tab_gen)
        t_gen.setContentsMargins(12, 12, 12, 12)
        t_gen.setSpacing(12)

        form_gen = QFormLayout()
        form_gen.setSpacing(12)

        self.cmb_ps_ver = QComboBox()
        self.cmb_ps_ver.addItem("Otomatik (Açık Photoshop'a bağlan)", "auto")
        self.cmb_ps_ver.addItem("Adobe Photoshop 2026 Öncelikli", "2026")
        self.cmb_ps_ver.addItem("Adobe Photoshop 2021 Öncelikli", "2021")
        form_gen.addRow("Photoshop Sürümü:", self.cmb_ps_ver)

        out_row = QWidget()
        out_l = QHBoxLayout(out_row)
        out_l.setContentsMargins(0, 0, 0, 0)
        out_l.setSpacing(8)
        self.txt_out_dir = QLineEdit()
        btn_out = QPushButton("Seç...")
        btn_out.clicked.connect(self._browse_out_dir)
        out_l.addWidget(self.txt_out_dir, 1)
        out_l.addWidget(btn_out)
        form_gen.addRow("Çıktı Klasörü:", out_row)

        self.chk_mascot = QCheckBox("Canlı Tekir ve Bombay maskot sahnesini göster")
        form_gen.addRow("", self.chk_mascot)

        self.spn_pause = QDoubleSpinBox()
        self.spn_pause.setRange(0.5, 15.0)
        self.spn_pause.setSingleStep(0.5)
        self.spn_pause.setSuffix(" sn")
        form_gen.addRow("Sahneler Arası Bekleme:", self.spn_pause)

        t_gen.addLayout(form_gen)
        t_gen.addStretch(1)
        tabs.addTab(tab_gen, "Genel")

        root.addWidget(tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("İptal")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Ayarları Kaydet")
        btn_save.setObjectName("PrimaryButton")
        btn_save.clicked.connect(self._save_and_close)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _load_values(self) -> None:
        pid = self._settings.get("api_provider", "none")
        idx = self.cmb_provider.findData(pid)
        self.cmb_provider.setCurrentIndex(idx if idx >= 0 else 0)
        self.txt_model.setText(self._settings.get("api_model", ""))
        self._on_provider_changed()

        for k, edit in self._key_inputs.items():
            edit.setText(self._env_vals.get(k, ""))

        ps_idx = self.cmb_ps_ver.findData(self._settings.get("ps_preferred_version", "auto"))
        self.cmb_ps_ver.setCurrentIndex(ps_idx if ps_idx >= 0 else 0)
        self.txt_out_dir.setText(self._settings.get("output_dir", ""))
        self.chk_mascot.setChecked(bool(self._settings.get("mascot_enabled", True)))
        self.spn_pause.setValue(float(self._settings.get("mascot_pause_between_sec", 2.0)))

    def _on_provider_changed(self) -> None:
        pid = self.cmb_provider.currentData()
        spec = PROVIDERS.get(pid)
        if not spec:
            return
        self.lbl_provider_note.setText(spec.note)

        show_model = pid != "none"
        self.lbl_model_row.setVisible(show_model)
        self.txt_model.setVisible(show_model)
        self.txt_model.setPlaceholderText(spec.default_model or "")

        needed = set(spec.env_keys)
        for k, row_w in self._key_rows.items():
            vis = k in needed
            row_w.setVisible(vis)
            self._provider_key_labels[k].setVisible(vis)

        self.adjustSize()

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Varsayılan Çıktı Klasörü Seçin", self.txt_out_dir.text())
        if d:
            self.txt_out_dir.setText(d)

    def _save_and_close(self) -> None:
        env_updates = {k: self._key_inputs[k].text().strip() for k in ENV_KEYS if k in self._key_inputs}
        save_env_values(env_updates)

        self._settings["api_provider"] = self.cmb_provider.currentData()
        self._settings["api_model"] = self.txt_model.text().strip()
        self._settings["ps_preferred_version"] = self.cmb_ps_ver.currentData()
        self._settings["output_dir"] = self.txt_out_dir.text().strip()
        self._settings["mascot_enabled"] = self.chk_mascot.isChecked()
        self._settings["mascot_pause_between_sec"] = float(self.spn_pause.value())
        save_settings(self._settings)
        self.accept()
