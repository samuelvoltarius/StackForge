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

# Projekt-Root auf den Importpfad — NUR im Quellcode-Modus. Im gebündelten Binary
# (PyInstaller) würde das den Pfad verschmutzen und cv2 doppelt auflösen (Rekursionsfehler).
if not getattr(sys, "frozen", False):
    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _root)
    sys.path.insert(0, os.path.join(_root, "core"))   # Engine-Module liegen jetzt in core/

from ui.main_window import MainWindow, THEME, main, APP_NAME, ICON, ICON_PNG  # noqa: F401
from ui.components import (  # noqa: F401  (Rück-Export für bestehende Skripte/Tests)
    CompareSlider, CurveWidget, AdjustDialog, RetouchDialog, _Canvas,
    _bgr_to_pixmap, histogram_pixmap, adjust_image, HSL_BANDS,
    help_btn, _row, reveal_in_files, open_path, notify,
)

if __name__ == "__main__":
    # Im gebündelten Binary (PyInstaller) ist `sys.executable` das Binary selbst, nicht python.
    # Damit der GUI-Subprozess die Pipeline starten kann, dient `--cli` als zweiter Einstiegspunkt:
    #   forgepix --cli --input … → ruft focus_cull_stack.main() statt der GUI.
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import focus_cull_stack
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        focus_cull_stack.main()
    else:
        main()
