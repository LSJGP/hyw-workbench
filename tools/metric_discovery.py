"""Discover grading metrics from REGISTER_METRIC and metrics_default.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(WORKBENCH_ROOT))
from hyw_paths import DEFAULT_METRICS, HYW_GRADING  # noqa: E402

_REGISTER_RE = re.compile(
    r'REGISTER_METRIC\s*\(\s*[^,]+,\s*"([^"]+)"\s*\)',
)
_RUNNABLE_RE = re.compile(r'\bn\s*==\s*"([^"]+)"')

_GRADING_SRC = HYW_GRADING / "src"
_GRADING_MAIN = HYW_GRADING / "src" / "entry" / "grading_main.cc"

_cache_key: Optional[Tuple[float, float]] = None
_cache_catalog: Optional[Dict[str, Dict[str, Any]]] = None
_cache_runnable: Optional[Set[str]] = None


def _dir_mtime(root: Path) -> float:
    if not root.is_dir():
        return 0.0
    latest = 0.0
    for p in root.rglob("*.cc"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            pass
    return latest


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _cache_valid() -> bool:
    global _cache_key, _cache_catalog
    if _cache_catalog is None or _cache_key is None:
        return False
    key = (_dir_mtime(_GRADING_SRC), _file_mtime(_GRADING_MAIN), _file_mtime(DEFAULT_METRICS))
    return key == _cache_key


def scan_registered_metric_names() -> List[str]:
    names: Set[str] = set()
    if not _GRADING_SRC.is_dir():
        return []
    for cc in _GRADING_SRC.rglob("*.cc"):
        try:
            text = cc.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _REGISTER_RE.finditer(text):
            names.add(m.group(1))
    return sorted(names)


def scan_runnable_metric_names() -> Set[str]:
    if not _GRADING_MAIN.is_file():
        return set()
    try:
        text = _GRADING_MAIN.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(_RUNNABLE_RE.findall(text))


def _load_default_params() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    if not DEFAULT_METRICS.is_file():
        return out
    try:
        data = json.loads(DEFAULT_METRICS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for entry in data.get("metrics") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or not isinstance(name, str):
            continue
        pj = entry.get("paramsJson")
        if pj is None:
            out[name] = None
        else:
            out[name] = pj if isinstance(pj, str) else json.dumps(pj, ensure_ascii=False)
    return out


def default_metric_names() -> List[str]:
    if not DEFAULT_METRICS.is_file():
        return scan_registered_metric_names()
    try:
        data = json.loads(DEFAULT_METRICS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return scan_registered_metric_names()
    names: List[str] = []
    for entry in data.get("metrics") or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return names


def discover_metric_catalog() -> Dict[str, Dict[str, Any]]:
    global _cache_key, _cache_catalog, _cache_runnable
    if _cache_valid():
        return dict(_cache_catalog or {})

    registered = scan_registered_metric_names()
    params = _load_default_params()
    runnable = scan_runnable_metric_names()
    _cache_runnable = runnable

    catalog: Dict[str, Dict[str, Any]] = {}
    for name in registered:
        catalog[name] = {
            "paramsJson": params.get(name),
            "registered": True,
            "runnable": name in runnable,
        }

    _cache_key = (
        _dir_mtime(_GRADING_SRC),
        _file_mtime(_GRADING_MAIN),
        _file_mtime(DEFAULT_METRICS),
    )
    _cache_catalog = catalog
    return dict(catalog)


def discover_metric_meta() -> List[Dict[str, Any]]:
    catalog = discover_metric_catalog()
    return [
        {
            "name": name,
            "registered": info.get("registered", True),
            "runnable": info.get("runnable", False),
            "has_default_params": info.get("paramsJson") is not None,
        }
        for name, info in sorted(catalog.items())
    ]


def get_metric_params_json(name: str) -> Optional[str]:
    catalog = discover_metric_catalog()
    entry = catalog.get(name)
    if not entry:
        return None
    pj = entry.get("paramsJson")
    return pj if isinstance(pj, str) else None
