"""Cache the clips and model weights in one folder for offline use.

A thin wrapper around ``paminaw prepare-offline`` (see ``docs/OFFLINE.md``)::

    python scripts/prepare_offline.py --out paminaw_offline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paminaw.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-offline", *sys.argv[1:]]))
