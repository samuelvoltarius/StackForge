#!/usr/bin/env bash
# StackForge-Tests (Standardbibliothek, kein pytest nötig)
cd "$(dirname "$0")"
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
