#!/usr/bin/env python3
"""
focus_stack_gui.py — Starter für ForgePix.

Die Oberfläche liegt modular im Paket `ui/` (ui/main_window.py + ui/components.py).
Diese Datei ist nur der Einstiegspunkt und re-exportiert die öffentlichen Namen,
damit `python3 focus_stack_gui.py`, das .app-Bundle und bestehende Skripte/Tests
(`import focus_stack_gui as g; g.MainWindow / g.THEME / g.AdjustDialog …`) weiter funktionieren.

Start:  python3 focus_stack_gui.py
"""
import os
import sys

# Projekt-Root (dieses Verzeichnis) auf den Importpfad — falls von woanders gestartet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow, THEME, main, APP_NAME, ICON, ICON_PNG  # noqa: F401
from ui.components import (  # noqa: F401  (Rück-Export für bestehende Skripte/Tests)
    CompareSlider, CurveWidget, AdjustDialog, RetouchDialog, _Canvas,
    _bgr_to_pixmap, histogram_pixmap, adjust_image, HSL_BANDS,
    help_btn, _row, reveal_in_files, open_path, notify,
)

if __name__ == "__main__":
    main()
