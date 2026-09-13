"""Build 448px annotation thumbnails for the SHROOM image set.

Thumbnails exist only so the annotator can view every image cheaply. Nothing on
the evaluation path ever reads them: the models always open the full-resolution
original from `data/shroom/images`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SRC = Path("shroom-visions-images/shroom-vis-images")
MAX_SIDE = 448


def main(dest: str) -> int:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.iterdir() if p.is_file())
    made = skipped = 0
    for i, p in enumerate(files):
        target = out / f"{i:04d}.jpg"
        if target.exists():
            skipped += 1
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - a broken file is data, not a crash
            print(f"UNREADABLE {p.name}: {exc}")
            continue
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(target, quality=82)
        made += 1
    print(f"{len(files)} sources -> {made} written, {skipped} already present, in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
