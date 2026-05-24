#!/usr/bin/env python3
"""Merge hyw-sim/ and hyw-grading/ compile_commands.json for IDE at workbench root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
HYW_ROOT = WORKBENCH_ROOT.parent


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


def main() -> int:
    merged: list = []
    for sub, prefix in (("hyw-sim", "hyw-sim"), ("hyw-grading", "hyw-grading")):
        cc = HYW_ROOT / sub / "compile_commands.json"
        entries = _load(cc)
        if entries:
            print(f"[merge] {sub}: {len(entries)} entries")
            merged.extend(_prefix_entries(entries, prefix))
        else:
            print(f"[merge] skip missing/empty: {cc}", file=sys.stderr)

    out = WORKBENCH_ROOT / "compile_commands.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[merge] wrote {len(merged)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
