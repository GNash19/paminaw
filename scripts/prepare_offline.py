"""Cache the clips and model weights in one folder for the workshop's external drive.

A thin wrapper around ``paminaw prepare-offline`` (see ``docs/OFFLINE.md``)::

    python scripts/prepare_offline.py --out D:/paminaw_offline --clips-zip paminaw_clips_v1.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paminaw.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-offline", *sys.argv[1:]]))
