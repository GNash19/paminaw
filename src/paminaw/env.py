"""Environment checks, seeding and the one-call workshop setup used by cell 0."""

from __future__ import annotations

import platform
import random
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from .data import SEED, PaminawDataError, ensure_clips, load_manifest

__all__ = ["WorkshopSetup", "set_seed", "setup_workshop", "versions"]

#: Packages whose versions cell 0 prints.
REPORTED = ("torch", "torchaudio", "transformers", "datasets", "jiwer")


def versions(packages: tuple[str, ...] = REPORTED) -> dict[str, str]:
    """Installed versions of the workshop's key packages ("not installed" if missing)."""
    out = {"python": platform.python_version()}
    for name in packages:
        try:  # package metadata works even when a module has no __version__ (jiwer 4)
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def set_seed(seed: int = SEED) -> None:
    """Fix Python, NumPy and PyTorch random seeds (default: the paper's 20260814)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WorkshopSetup:
    """What cell 0 found: device, precision, data folder and how long it took."""

    device: str
    dtype: torch.dtype
    gpu_name: str | None
    clips_root: Path
    n_clips: int
    seconds: float
    smoke: bool


def setup_workshop(
    smoke: bool = False,
    cache_dir: str | Path | None = None,
    offline_dir: str | Path | None = None,
) -> WorkshopSetup:
    """Print versions, detect GPU/CPU, download the 200 clips and print a READY line.

    Stops with a one-sentence fix if anything is missing.
    """
    t0 = time.perf_counter()
    set_seed()
    v = versions()
    print("📦 Versions: " + " · ".join(f"{k} {val}" for k, val in v.items()))
    missing = [k for k in ("torchaudio", "transformers", "jiwer") if v[k] == "not installed"]
    if missing:
        raise SystemExit(
            f"❌ Missing packages: {', '.join(missing)}. "
            "Fix: run cell 0 again; if it still fails, use Runtime → Disconnect and delete runtime."
        )
    major, minor = (int(x) for x in v["transformers"].split(".")[:2])
    if (major, minor) < (4, 56):
        print(
            f"⚠️  transformers {v['transformers']} uses torch_dtype= instead of dtype=. "
            "Fix: run `pip install -U transformers` and restart the runtime."
        )

    if torch.cuda.is_available():
        device, dtype, gpu = "cuda", torch.float16, torch.cuda.get_device_name(0)
        print(f"🚀 GPU found: {gpu} → Lab 2b (MMS) will run live.")
    else:
        device, dtype, gpu = "cpu", torch.float32, None
        print(
            "🐢 No GPU: running on CPU. Everything works; Lab 2b will load precomputed MMS "
            "outputs. (Optional: Runtime → Change runtime type → T4 GPU.)"
        )

    try:
        root = ensure_clips(cache_dir=cache_dir, offline_dir=offline_dir)
    except PaminawDataError as err:
        raise SystemExit(f"❌ {err}") from None
    n = len(load_manifest())
    secs = time.perf_counter() - t0
    mode = " · SMOKE TEST (2 clips, whisper-tiny, no MMS)" if smoke else ""
    print(
        f"✅ READY: {n} clips ({n // 2} Tagalog + {n // 2} Cebuano, workshop subset) · "
        f"{device} ({str(dtype).replace('torch.', '')}) · {secs:.0f} s{mode}"
    )
    return WorkshopSetup(device, dtype, gpu, root, n, secs, smoke)
