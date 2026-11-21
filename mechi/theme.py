"""MeChi Studio v1.1 — Sade, ferah ve yuksek okunurluklu koyu tasarim sistemi."""
from __future__ import annotations

from typing import Dict

TOKENS: Dict[str, str] = {
    "bg_base": "#141618",
    "bg_nav": "#1A1D20",
    "bg_surface": "#1F2326",
    "bg_surface_hover": "#272C30",
    "bg_stage": "#262828",  # 4 maskot WebP klibinin kenar medyani [38, 40, 40] ile eslesir
    "bg_input": "#121416",
    "border_subtle": "#2B3136",
    "border_strong": "#3E464D",
    "text_primary": "#F5F7F6",
    "text_secondary": "#B8C1BE",
    "text_muted": "#889390",
    "accent": "#E0A34D",
    "accent_hover": "#ECBA6B",
    "accent_pressed": "#C98C36",
    "accent_text": "#141618",
    "ok": "#56C28E",
    "warn": "#E0A34D",
    "err": "#E06C6C",
}


def build_stylesheet() -> str:
    t = TOKENS
    return f"""
    QWidget {{
        background-color: {t['bg_base']};
        color: {t['text_primary']};
        font-family: "Segoe UI", "Inter", sans-serif;
        font-size: 14px;
    }}

    QLabel {{
        background-color: transparent;
    }}

    QMainWindow, QDialog {{
        background-color: {t['bg_base']};
    }}

    QFrame#TopBar {{
        background-color: {t['bg_nav']};
        border-bottom: 1px solid {t['border_subtle']};
    }}

    QFrame#NavRail {{
        background-color: {t['bg_nav']};
        border-right: 1px solid {t['border_subtle']};
    }}

    QFrame#StatusFooter {{
        background-color: {t['bg_nav']};
        border-top: 1px solid {t['border_subtle']};
    }}

    QFrame#MascotStage {{
        background-color: {t['bg_stage']};
        border: none;
        border-radius: 14px;
    }}

    QLabel#MascotCanvas {{
        background-color: {t['bg_stage']};
        border: none;
    }}

    QFrame#CleanSection, QFrame#SurfaceCard {{
        background-color: {t['bg_surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: 12px;
    }}

    QFrame#ModuleCard {{
        background-color: {t['bg_surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: 12px;
    }}

    QFrame#ModuleCard:hover {{
        background-color: {t['bg_surface_hover']};
        border: 1px solid {t['accent']};
    }}

    QFrame#PromptResultCard {{
        background-color: {t['bg_surface']};
        border: 1px solid {t['accent']};
        border-radius: 12px;
    }}

    QLabel#HeroTitle {{
        color: {t['text_primary']};
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.4px;
    }}

    QLabel#SectionTitle {{
        color: {t['text_primary']};
        font-size: 19px;
        font-weight: 700;
    }}

    QLabel#CardTitle {{
        color: {t['text_primary']};
        font-size: 16px;
        font-weight: 600;
    }}

    QLabel#SectionLabel {{
        color: {t['text_primary']};
        font-size: 14px;
        font-weight: 600;
    }}

    QLabel#MutedText {{
        color: {t['text_secondary']};
        font-size: 13px;
    }}

    QLabel#SubtleHint {{
        color: {t['text_muted']};
        font-size: 12px;
    }}

    QPushButton#PsStatusPill {{
        background-color: {t['bg_surface']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_subtle']};
        border-radius: 14px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 500;
    }}

    QPushButton#PsStatusPill:hover {{
        background-color: {t['bg_surface_hover']};
        border-color: {t['border_strong']};
        color: {t['text_primary']};
    }}

    QPushButton#PsStatusPill[connected="true"] {{
        background-color: rgba(86, 194, 142, 0.14);
        color: {t['ok']};
        border: 1px solid rgba(86, 194, 142, 0.35);
    }}

    QLabel#BadgeOk {{
        background-color: rgba(86, 194, 142, 0.14);
        color: {t['ok']};
        border: 1px solid rgba(86, 194, 142, 0.35);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: 600;
    }}

    QLabel#BadgeWarn {{
        background-color: rgba(224, 163, 77, 0.12);
        color: {t['warn']};
        border: 1px solid rgba(224, 163, 77, 0.35);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: 600;
    }}

    QPushButton {{
        background-color: {t['bg_surface_hover']};
        color: {t['text_primary']};
        border: 1px solid {t['border_strong']};
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
    }}

    QPushButton:hover {{
        background-color: #30373C;
        border-color: {t['text_muted']};
    }}

    QPushButton:pressed {{
        background-color: {t['bg_surface']};
    }}

    QPushButton:disabled {{
        color: {t['text_muted']};
        background-color: {t['bg_surface']};
        border-color: {t['border_subtle']};
    }}

    QPushButton#PrimaryButton {{
        background-color: {t['accent']};
        color: {t['accent_text']};
        border: 1px solid {t['accent']};
        border-radius: 9px;
        font-size: 14px;
        font-weight: 700;
        padding: 11px 22px;
    }}

    QPushButton#PrimaryButton:hover {{
        background-color: {t['accent_hover']};
        border-color: {t['accent_hover']};
    }}

    QPushButton#PrimaryButton:pressed {{
        background-color: {t['accent_pressed']};
    }}

    QPushButton#PrimaryButton:disabled {{
        background-color: {t['bg_surface_hover']};
        color: {t['text_muted']};
        border-color: {t['border_subtle']};
    }}

    QPushButton#PresetChip {{
        background-color: {t['bg_input']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_subtle']};
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
    }}

    QPushButton#PresetChip:hover {{
        background-color: {t['bg_surface_hover']};
        color: {t['text_primary']};
    }}

    QPushButton#PresetChip:checked {{
        background-color: rgba(224, 163, 77, 0.16);
        color: {t['accent_hover']};
        border: 1px solid {t['accent']};
    }}

    QPushButton#NavButton {{
        background-color: transparent;
        color: {t['text_secondary']};
        border: 1px solid transparent;
        border-radius: 9px;
        text-align: left;
        padding: 11px 14px;
        font-size: 14px;
        font-weight: 500;
    }}

    QPushButton#NavButton:hover {{
        background-color: {t['bg_surface']};
        color: {t['text_primary']};
    }}

    QPushButton#NavButton[active="true"] {{
        background-color: {t['bg_surface_hover']};
        color: {t['text_primary']};
        border: 1px solid {t['border_strong']};
        border-left: 3px solid {t['accent']};
        font-weight: 700;
    }}

    QPushButton#LockToggle {{
        background-color: {t['bg_input']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_subtle']};
        border-radius: 16px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 500;
    }}

    QPushButton#LockToggle:checked {{
        background-color: rgba(224, 163, 77, 0.16);
        color: {t['accent_hover']};
        border: 1px solid {t['accent']};
        font-weight: 600;
    }}

    QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
        border: 1px solid {t['border_strong']};
        border-radius: 8px;
        padding: 8px 12px;
        min-height: 22px;
        font-size: 14px;
        selection-background-color: {t['accent']};
        selection-color: {t['accent_text']};
    }}

    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {t['accent']};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 28px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {t['bg_surface']};
        color: {t['text_primary']};
        border: 1px solid {t['border_strong']};
        selection-background-color: {t['bg_surface_hover']};
        padding: 4px;
    }}

    QCheckBox, QRadioButton {{
        background: transparent;
        color: {t['text_primary']};
        spacing: 8px;
        font-size: 14px;
    }}

    QCheckBox:disabled, QRadioButton:disabled {{
        color: {t['text_muted']};
    }}

    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 10px;
        border: 2px solid {t['border_strong']};
        background-color: {t['bg_input']};
    }}

    QRadioButton::indicator:checked {{
        width: 12px;
        height: 12px;
        border-radius: 10px;
        border: 4px solid {t['accent']};
        background-color: {t['bg_base']};
    }}

    QRadioButton::indicator:disabled {{
        width: 16px;
        height: 16px;
        border-radius: 10px;
        border: 2px solid {t['border_subtle']};
        background-color: {t['bg_surface']};
    }}

    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 2px solid {t['border_strong']};
        background-color: {t['bg_input']};
    }}

    QCheckBox::indicator:checked {{
        border: 2px solid {t['accent']};
        background-color: {t['accent']};
    }}

    QCheckBox::indicator:disabled {{
        border: 2px solid {t['border_subtle']};
        background-color: {t['bg_surface']};
    }}

    QSlider::groove:horizontal {{
        height: 6px;
        background: {t['bg_input']};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        background: {t['accent']};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}

    QProgressBar {{
        background-color: {t['bg_input']};
        border: 1px solid {t['border_subtle']};
        border-radius: 5px;
        height: 10px;
        text-align: center;
        color: {t['text_secondary']};
        font-size: 11px;
    }}

    QProgressBar::chunk {{
        background-color: {t['accent']};
        border-radius: 4px;
    }}

    QTabWidget::pane {{
        border: 1px solid {t['border_subtle']};
        border-radius: 12px;
        background-color: {t['bg_surface']};
        padding: 8px;
    }}

    QTabBar::tab {{
        background-color: {t['bg_nav']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_subtle']};
        padding: 9px 18px;
        font-size: 13px;
        font-weight: 600;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
    }}

    QTabBar::tab:selected {{
        background-color: {t['bg_surface']};
        color: {t['text_primary']};
        border-bottom-color: {t['bg_surface']};
    }}

    QScrollArea {{
        border: none;
        background-color: transparent;
    }}

    QScrollBar:vertical {{
        background: {t['bg_base']};
        width: 9px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {t['border_strong']};
        min-height: 28px;
        border-radius: 4px;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
