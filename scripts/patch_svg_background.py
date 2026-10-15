#!/usr/bin/env python3
"""Patch a literal white background rect into exported architecture SVGs.

GitHub's SVG sanitizer strips inline ``style`` attributes on the ``<svg>``
root, so ``style="background:#fff"`` never survives rendering there. The
only backdrop that survives is a real drawn shape: a ``<rect>`` covering the
full viewBox, inserted as the first child right after the outer
``<defs/><g>`` wrapper that draw.io's exporter emits.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
VIEWBOX_RE = re.compile(r'viewBox="0 0 (?P<w>[\d.]+) (?P<h>[\d.]+)"')


def patch_svg(path: Path) -> None:
    text = path.read_text()
    match = VIEWBOX_RE.search(text)
    if not match:
        raise ValueError(f"{path} has no viewBox; cannot size the background rect.")
    width, height = match.group("w"), match.group("h")
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'

    if rect in text:
        print(f"{path} already patched, skipping.")
        return

    # Insert right after the first "<defs/><g ...>" (draw.io's standard
    # export wrapper), so the rect becomes the first drawn element.
    pattern = re.compile(r"(<defs\s*/>\s*<g[^>]*>)")
    patched, count = pattern.subn(lambda m: m.group(1) + rect, text, count=1)
    if count == 0:
        raise ValueError(f"{path}: could not find '<defs/><g>' to patch after.")

    path.write_text(patched)
    print(f"patched white background into {path}")


def main() -> int:
    svgs = sorted(DOCS.glob("*.svg"))
    if not svgs:
        print(f"No SVGs found under {DOCS}", file=sys.stderr)
        return 1
    for svg in svgs:
        patch_svg(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
