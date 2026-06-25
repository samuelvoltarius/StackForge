#!/usr/bin/env python3
"""ui/appinfo.py — geteilte Pfad- und Namens-Konstanten für ForgePix (von mehreren ui-Modulen genutzt)."""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Projekt-Root (ui/ liegt darunter)
SCRIPT = os.path.join(HERE, "focus_cull_stack.py")
ICON = os.path.join(HERE, "assets", "ForgePix.icns")
ICON_PNG = os.path.join(HERE, "assets", "forgepix_512.png")
APP_NAME = "ForgePix"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
