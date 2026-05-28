#!/usr/bin/env python3
"""Merge sub-repo compile_commands.json for IDE at workbench root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
HYW_ROOT = WORKBENCH_ROOT.parent

_SOURCE_ROOTS = tuple(
    (HYW_ROOT / sub / subpath).resolve()
    for sub, subpath in (
        ("hyw-sim", "cpp"),
        ("hyw-planner", "cpp"),
        ("hyw-grading", "src"),
    )
)
_SOURCE_SUFFIXES = {".cc", ".cpp", ".c", ".h", ".hpp", ".hh"}


def _load(path: Path) -> list:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _prefix_entries(entries: list, prefix: str) -> list:
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        e = dict(e)
        for key in ("file", "directory"):
            if key in e and isinstance(e[key], str):
                p = Path(e[key])
                if not p.is_absolute():
                    e[key] = str((HYW_ROOT / prefix / p).resolve())
        out.append(e)
    return out


def _is_first_party_source(file_path: str) -> bool:
    try:
        path = Path(file_path).resolve()
    except OSError:
        return False
    if path.suffix not in _SOURCE_SUFFIXES:
        return False
    for root in _SOURCE_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _filter_first_party(entries: list) -> list:
    return [e for e in entries if _is_first_party_source(e.get("file", ""))]


def main() -> int:
    merged: list = []
    for sub, prefix in (
        ("hyw-sim", "hyw-sim"),
        ("hyw-planner", "hyw-planner"),
        ("hyw-grading", "hyw-grading"),
    ):
        cc = HYW_ROOT / sub / "compile_commands.json"
        entries = _load(cc)
        if entries:
            prefixed = _prefix_entries(entries, prefix)
            kept = _filter_first_party(prefixed)
            print(f"[merge] {sub}: {len(entries)} raw -> {len(kept)} first-party")
            merged.extend(kept)
        else:
            print(f"[merge] skip missing/empty: {cc}", file=sys.stderr)

    out = WORKBENCH_ROOT / "compile_commands.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[merge] wrote {len(merged)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
