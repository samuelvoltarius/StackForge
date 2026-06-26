#!/usr/bin/env python3
"""ui/theme.py — globales Qt-Stylesheet (Anthrazit + Chili-Grün) für ForgePix."""
THEME = """
/* ForgePix — Anthrazit + Chili-Grün (GreenChili). Statusfarben: grün=gut, gelb=Warnung,
   rot=Problem, blau=Info. Akzent #4caf50-Familie. */
QWidget { background:#16171a; color:#e8eae6; font-size:13px; }
QMainWindow, QDialog { background:#16171a; }

/* Karten/Gruppen — sichtbar abgehobene Flächen auf Anthrazit */
QGroupBox {
    background:#202227; border:1px solid #30343a; border-radius:12px;
    margin-top:20px; padding:16px 12px 12px 12px; font-weight:600; }
QGroupBox::title {
    subcontrol-origin:margin; subcontrol-position:top left; left:14px; padding:3px 8px;
    color:#7bd36a; font-size:12px; font-weight:700; background:#16171a; }
QGroupBox::indicator { width:18px; height:18px; }

QLabel { background:transparent; }
/* Wiederverwendbare Label-Rollen (statt verstreuter Inline-Styles) */
QLabel#sectionHeader { color:#7bd36a; font-weight:bold; margin-top:8px; }
QLabel#hint { color:#9aa09a; }

/* Eingaben — flach, mit Grün-Fokus */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background:#26282e; border:1px solid #34383f; border-radius:8px; padding:6px 8px; color:#e8eae6;
    selection-background-color:#4caf50; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border:1px solid #5cc85c; background:#282b30; }
QPlainTextEdit { background:#101113; border:1px solid #26282e; }
QComboBox::drop-down { border:none; width:22px; }
QComboBox QAbstractItemView {
    background:#202227; border:1px solid #34383f; border-radius:8px;
    selection-background-color:#4caf50; outline:none; padding:4px; }

/* Standard-Buttons — flach, weicher Rand */
QPushButton {
    background:#2a2d33; border:1px solid #3a3e45; border-radius:9px; padding:8px 14px; color:#e8eae6; }
QPushButton:hover { background:#33373e; border-color:#4a4f57; }
QPushButton:pressed { background:#3c4149; }
QPushButton:disabled { color:#6a6e73; background:#1d1f23; border-color:#26282e; }

/* Primär-Aktion (objectName 'primary') — gefülltes Chili-Grün */
QPushButton#primary {
    background:#4caf50; border:1px solid #4caf50; color:#0d1f0e; font-weight:700; }
QPushButton#primary:hover { background:#5cc85c; border-color:#5cc85c; }
QPushButton#primary:pressed { background:#3f9942; }
QPushButton#primary:disabled { background:#2f4630; border-color:#2f4630; color:#8aa88c; }

/* Modul-Karten auf dem Startbildschirm */
QPushButton#card {
    background:#202227; border:1px solid #34383f; border-radius:16px; text-align:center; }
QPushButton#card:hover { background:#23282a; border:2px solid #4caf50; }
QPushButton#card:pressed { background:#1c2a1c; }

/* Schnell-Export-Chips im Entscheidungs-Panel */
QPushButton#chip {
    background:#23252c; border:1px solid #3a3d47; border-radius:13px;
    padding:4px 11px; font-size:12px; font-weight:600; color:#cfd2cd; }
QPushButton#chip:hover { background:#2b3a2b; border-color:#4caf50; color:#dff3df; }
QPushButton#chip:pressed { background:#1c2a1c; }
QPushButton#chip:disabled { background:#1b1c21; border-color:#26282f; color:#5a5d63; }

QCheckBox { spacing:7px; }
QCheckBox::indicator {
    width:18px; height:18px; border-radius:5px; border:1px solid #3c4047; background:#26282e; }
QCheckBox::indicator:hover { border-color:#5cc85c; }
QCheckBox::indicator:checked { background:#4caf50; border-color:#4caf50; }

QProgressBar {
    border:none; border-radius:7px; background:#26282e; text-align:center; height:16px; color:#cfd2cd; }
QProgressBar::chunk { background:#4caf50; border-radius:7px; }

/* Schieberegler — grüner Verlauf, heller Griff (statt Qt-Standard-Blau) */
QSlider::groove:horizontal { height:5px; background:#34383f; border-radius:3px; }
QSlider::sub-page:horizontal { background:#4caf50; border-radius:3px; }
QSlider::add-page:horizontal { background:#34383f; border-radius:3px; }
QSlider::handle:horizontal {
    background:#e8eae6; width:15px; height:15px; margin:-6px 0; border-radius:8px; }
QSlider::handle:horizontal:hover { background:#ffffff; }
QSlider::groove:vertical { width:5px; background:#34383f; border-radius:3px; }
QSlider::handle:vertical {
    background:#e8eae6; width:15px; height:15px; margin:0 -6px; border-radius:8px; }

QSplitter::handle { background:transparent; width:10px; }

QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:#34383f; border-radius:5px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:#474c54; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; }
QScrollBar:horizontal { background:transparent; height:10px; margin:2px; }
QScrollBar::handle:horizontal { background:#34383f; border-radius:5px; min-width:30px; }

QToolTip {
    background:#26282e; color:#e8eae6; border:1px solid #4a7d4a; border-radius:8px; padding:6px 8px; }
QMenu { background:#202227; border:1px solid #312f40; border-radius:8px; padding:4px; }
QMenu::item:selected { background:#4caf50; border-radius:6px; }
"""
