"""Tiny YAML config loader. Keeps every experiment reproducible from one file."""
from __future__ import annotations
import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)
