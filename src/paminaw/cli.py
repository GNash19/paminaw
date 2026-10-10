"""Command line: ``paminaw check``, ``paminaw score`` and ``paminaw prepare-offline``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import (
    CLIPS_URL,
    CLIPS_ZIP,
    PaminawDataError,
    data_file,
    default_cache_dir,
    ensure_clips,
    load_manifest,
    load_precomputed,
)

#: Models the USB drive carries, with the files each one needs.
OFFLINE_MODELS: dict[str, dict[str, list[str]]] = {
    "openai/whisper-small": {"ignore": ["*.bin", "*.h5", "*.msgpack", "*.onnx"]},
    "openai/whisper-tiny": {"ignore": ["*.bin", "*.h5", "*.msgpack", "*.onnx"]},
    "facebook/mms-1b-all": {
        "allow": ["*.json", "*.txt", "model.safetensors", "adapter.ceb.safetensors",
                  "adapter.tgl.safetensors"],
    },
}  # fmt: skip


def cmd_check(args: argparse.Namespace) -> int:
    """Print versions, device, audio backend and data status; return 1 if something is wrong."""
    from .asr import pick_device_dtype
    from .audio import load_audio
    from .env import torchcodec_problem, versions

    ok = True
    print(f"paminaw {__version__}")
    for k, v in versions().items():
        print(f"  {k:<13}{v}")
    device, dtype = pick_device_dtype()
    print(f"  {'device':<13}{device} ({str(dtype).replace('torch.', '')})")
    problem = torchcodec_problem()
    if problem:
        ok = False
        print(f"  {'torchcodec':<13}BROKEN. {problem}")

    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        tone = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype("float32")
        sf.write(Path(tmp) / "tone.wav", tone, 16000)
        wav, sr = load_audio(Path(tmp) / "tone.wav")
        print(f"  {'audio':<13}OK, read a test tone as {tuple(wav.shape)} at {sr} Hz")

    print(f"  {'manifest':<13}{len(load_manifest())} clips")
    print(f"  {'precomputed':<13}{len(load_precomputed())} hypotheses")
    offline = os.environ.get("PAMINAW_OFFLINE_DIR")
    if offline:
        found = (Path(offline) / "clips").is_dir() or (Path(offline) / CLIPS_ZIP).is_file()
        print(f"  {'offline dir':<13}{offline} ({'clips found' if found else 'NO clips found'})")
        ok = ok and found
    cache = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    have = (cache / "clips").is_dir() or bool(offline)
    print(f"  {'clips cache':<13}{cache} ({'present' if (cache / 'clips').is_dir() else 'empty'})")
    if not args.offline and not have:
        try:
            req = urllib.request.Request(CLIPS_URL, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"  {'release zip':<13}reachable (HTTP {resp.status})")
        except Exception as err:
            ok = False
            print(
                f"  {'release zip':<13}NOT reachable ({err}). "
                "Fix: check internet or use the USB copy."
            )
    return 0 if ok else 1


def cmd_score(args: argparse.Namespace) -> int:
    """Score a CSV of hypotheses (columns lang, idx, hypothesis) against the subset references."""
    from .compare import convention_table, score_systems

    hyps = pd.read_csv(args.hypotheses, keep_default_na=False)
    hyps = hyps.rename(columns={"hypothesis_raw": "hypothesis"})
    if "system" not in hyps.columns:
        hyps["system"] = args.system or Path(args.hypotheses).stem
    refs = load_manifest()
    n = hyps.groupby("lang")["idx"].nunique().to_dict()
    label = "workshop subset (" + ", ".join(f"{k} {v} clips" for k, v in n.items()) + ")"
    scores = score_systems(refs, hyps, subset_label=label)
    print(f"Scores on the {label}; these are not the paper's full-split numbers.\n")
    for metric in ("wer", "cer"):
        print(f"{metric.upper()} (%) by convention")
        print(convention_table(scores, metric).to_string(), end="\n\n")
    if args.out:
        scores.to_csv(args.out, index=False)
        print(f"Saved tidy scores to {args.out}")
    return 0


def cmd_prepare_offline(args: argparse.Namespace) -> int:
    """Copy the clips zip, the clips and model weights into one folder for the USB drive."""
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = ensure_clips(cache_dir=args.cache_dir)
    if (root / "clips").resolve() != (out / "clips").resolve():
        shutil.copytree(root / "clips", out / "clips", dirs_exist_ok=True)
    shutil.copy2(data_file("subset_manifest.csv"), out / "subset_manifest.csv")
    zip_src = Path(args.clips_zip) if args.clips_zip else None
    if zip_src and zip_src.is_file():
        shutil.copy2(zip_src, out / CLIPS_ZIP)
    print(f"✅ clips → {out / 'clips'}")
    hub_cache = out / "hf_home" / "hub"  # a normal Hugging Face cache; point HF_HOME at hf_home
    for name in args.models:
        spec = OFFLINE_MODELS.get(name, {})
        print(f"⬇️  {name} ...")
        snapshot_download(
            name,
            cache_dir=hub_cache,
            allow_patterns=spec.get("allow"),
            ignore_patterns=spec.get("ignore"),
        )
        print(f"✅ {name} → {hub_cache}")
    print(
        f"\nDone. Before starting Jupyter set PAMINAW_OFFLINE_DIR={out}, "
        f"HF_HOME={out / 'hf_home'} and HF_HUB_OFFLINE=1 (see docs/OFFLINE.md)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for the ``paminaw`` command."""
    p = argparse.ArgumentParser(prog="paminaw", description="Paminaw workshop tools.")
    p.add_argument("--version", action="version", version=f"paminaw {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="check versions, device, audio decoding and data")
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--offline", action="store_true", help="skip the network check")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("score", help="score a hypotheses CSV under all four conventions")
    s.add_argument("hypotheses", help="CSV with columns lang, idx, hypothesis")
    s.add_argument("--system", default=None, help="name to show for this system")
    s.add_argument("--out", default=None, help="optional CSV for the tidy scores")
    s.set_defaults(func=cmd_score)

    o = sub.add_parser("prepare-offline", help="cache clips and model weights for the USB drive")
    o.add_argument("--out", required=True, help="folder to fill, e.g. E:/paminaw_offline")
    o.add_argument("--models", nargs="*", default=list(OFFLINE_MODELS), help="Hub model names")
    o.add_argument("--clips-zip", default=None, help=f"path to {CLIPS_ZIP} to copy as well")
    o.add_argument("--cache-dir", default=None)
    o.set_defaults(func=cmd_prepare_offline)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``paminaw`` console script."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PaminawDataError as err:
        print(f"❌ {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
