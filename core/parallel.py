#!/usr/bin/env python3
"""
parallel.py — kleiner, geteilter Parallel-Helfer für ForgePix.

Die schweren Schleifen (RAW entwickeln, Schärfe messen) sind „peinlich parallel": jedes Bild
ist unabhängig. rawpy und OpenCV geben den Python-GIL während der Rechenarbeit frei, darum
reicht ein ThreadPool (kein Prozess-Overhead, kein Pickling). Ergebnis-Reihenfolge bleibt
erhalten (wichtig für die Nachbar-Logik des Cullings).
"""
import os
from concurrent.futures import ThreadPoolExecutor


def cpu_workers(memory_heavy=False):
    """Sinnvolle Worker-Zahl. memory_heavy=True (volle RAW-Entwicklung) wird gedeckelt,
    damit nicht zu viele große Bilder gleichzeitig im RAM liegen."""
    n = os.cpu_count() or 4
    if memory_heavy:
        return max(1, min(n, 6))
    return max(1, min(n, 12))


def pmap(fn, items, max_workers=None, memory_heavy=False):
    """fn auf jedes item anwenden, geordnete Ergebnisliste zurückgeben.
    Bei 0/1 Items oder max_workers=1 läuft es seriell (kein Thread-Overhead)."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    workers = max_workers if max_workers is not None else cpu_workers(memory_heavy)
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))
