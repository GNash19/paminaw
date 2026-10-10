"""Build the workshop subset: 100 Tagalog + 100 Cebuano FLEURS test clips.

Selection (deterministic, seed 20260814, the paper's seed):

1. For each language, list the distinct sentence IDs (the FLEURS ``id`` column)
   of the test split, sorted ascending.
2. ``random.Random(seed).sample(ids, 100)`` draws 100 of them; the order of the
   draw is kept as ``draw_order``.
3. For each drawn sentence, the same generator picks one of its recordings
   with ``rng.choice`` over that sentence's rows sorted by row index.

No duration cap. Each language uses its own ``Random(seed)``.

Outputs ``data/subset_manifest.csv`` and, with ``--bundle``, the release zip
``paminaw_clips_v1.zip`` holding the original FLEURS WAV bytes (unchanged),
the manifest and the attribution notice.

Usage::

    python scripts/make_subset.py --parquet-dir PATH/TO/fleurs --bundle dist/paminaw_clips_v1.zip

``--parquet-dir`` must hold ``fil_ph-test.parquet`` and ``ceb_ph-test.parquet``;
without it the files are downloaded from the Hub at the pinned revision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paminaw.data import FLEURS_REPO, FLEURS_REVISION, LANGUAGES, SEED  # noqa: E402

N_PER_LANG = 100
META_COLS = ["id", "num_samples", "path", "transcription", "raw_transcription", "gender"]
ZIP_DATE = (2026, 10, 17, 13, 10, 0)  # fixed timestamp so the zip is byte-reproducible


def parquet_path(config: str, parquet_dir: Path | None) -> Path:
    """Local path of a test-split Parquet file, downloading it if needed."""
    if parquet_dir is not None:
        p = parquet_dir / f"{config}-test.parquet"
        if p.is_file():
            return p
        raise SystemExit(f"{p} not found. Fix: download it or drop --parquet-dir to fetch it.")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        FLEURS_REPO, f"parquet-data/{config}/test-00000-of-00001.parquet",
        repo_type="dataset", revision=FLEURS_REVISION,
    ))  # fmt: skip


def select(meta: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` distinct sentences, one recording each, as described in the module docstring."""
    rng = random.Random(seed)
    sentence_ids = sorted(int(x) for x in meta["id"].unique())
    if len(sentence_ids) < n:
        raise SystemExit(f"Only {len(sentence_ids)} sentences available, need {n}.")
    chosen = rng.sample(sentence_ids, n)
    picks = []
    for order, sid in enumerate(chosen):
        candidates = sorted(meta.index[meta["id"] == sid].tolist())
        idx = rng.choice(candidates)
        picks.append((order, idx))
    out = meta.loc[[i for _, i in picks]].copy()
    out.insert(0, "draw_order", [o for o, _ in picks])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_PER_LANG)
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "subset_manifest.csv")
    ap.add_argument("--bundle", type=Path, default=None, help="write the release zip here")
    args = ap.parse_args()

    frames, audio = [], {}
    for lang, info in LANGUAGES.items():
        config = str(info["fleurs_config"])
        path = parquet_path(config, args.parquet_dir)
        meta = pq.read_table(path, columns=META_COLS).to_pandas()
        meta.index.name = "idx"
        sel = select(meta, args.n, args.seed).reset_index()
        sel["lang"] = lang
        sel["fleurs_config"] = config
        sel["split"] = "test"
        sel["filename"] = [Path(p).name for p in sel["path"]]
        sel["duration_s"] = (sel["num_samples"] / 16000).round(3)
        sel = sel.rename(columns={"id": "fleurs_id"})
        frames.append(sel)
        print(f"{lang}: {len(meta)} rows, {meta['id'].nunique()} sentences → picked {len(sel)}; "
              f"{sel['duration_s'].sum() / 60:.1f} min of audio")  # fmt: skip
        if args.bundle:
            col = pq.read_table(path, columns=["audio"]).column("audio")
            for idx, fname in zip(sel["idx"], sel["filename"], strict=True):
                raw = col[int(idx)].as_py()["bytes"]
                info_ = sf.info(io.BytesIO(raw))
                if info_.samplerate != 16000 or info_.channels != 1:
                    raise SystemExit(f"{fname}: expected 16 kHz mono, got {info_}")
                audio[f"clips/{config}/{fname}"] = raw

    cols = ["lang", "fleurs_config", "split", "draw_order", "idx", "fleurs_id", "filename",
            "transcription", "raw_transcription", "num_samples", "duration_s",
            "gender"]  # fmt: skip
    manifest = pd.concat(frames)[cols].sort_values(["lang", "draw_order"]).reset_index(drop=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False, lineterminator="\n")
    print(f"Wrote {args.manifest} ({len(manifest)} rows)")

    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        members = dict(sorted(audio.items()))
        members["subset_manifest.csv"] = args.manifest.read_bytes()
        members["ATTRIBUTION.md"] = (REPO_ROOT / "data" / "ATTRIBUTION.md").read_bytes()
        with zipfile.ZipFile(args.bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zi = zipfile.ZipInfo(name, date_time=ZIP_DATE)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                zf.writestr(zi, data)
        digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
        size = args.bundle.stat().st_size / 1e6
        print(f"Wrote {args.bundle} ({size:.1f} MB)\nsha256 {digest}")
        print("→ paste this sha256 into CLIPS_SHA256 in src/paminaw/data.py")


if __name__ == "__main__":
    main()
