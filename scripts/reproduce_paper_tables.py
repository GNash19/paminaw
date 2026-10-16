"""Re-derive the paper's Table 3 (WER under four conventions) and Table 4 (primary
WER/CER/MER) on the FULL FLEURS test splits with this package's scoring code.

The Zenodo release has no reference text, so references are read from the
FLEURS Parquet files: only the text columns, so no audio is decoded. Hypotheses
come from the release. Every row is joined on ``idx`` and its ``utt_id`` checked
against the FLEURS ``id``.

Usage::

    python scripts/reproduce_paper_tables.py --archive NJ105_release_v2.zip \
        [--parquet-dir DIR_WITH_fil_ph-test.parquet_AND_ceb_ph-test.parquet]

Without ``--parquet-dir`` the text columns are streamed from the Hub
(a few MB of range requests).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paminaw.compare import (  # noqa: E402
    PAPER_TABLE3_WER,
    PAPER_TABLE4_PRIMARY,
    score_conventions,
)
from paminaw.data import FLEURS_REVISION  # noqa: E402
from paminaw.metrics import cer_jiwer_default  # noqa: E402
from paminaw.normalize import normalize  # noqa: E402

RUNS = {
    "Whisper large-v3": "whisper__{lang}__committed",
    "MMS-1B-all": "mms__{lang}__matched",
    "SeamlessM4T v2 Large": "seamless__{lang}__matched",
}
LANGS = {"tagalog": ("fil_ph", "filipino"), "cebuano": ("ceb_ph", "cebuano")}
TOL = 0.005  # the paper rounds to two decimals


def references(config: str, parquet_dir: Path | None) -> pd.DataFrame:
    """FLEURS test-split ``id`` and ``transcription`` in Hub row order."""
    cols = ["id", "transcription"]
    if parquet_dir:
        return pq.read_table(parquet_dir / f"{config}-test.parquet", columns=cols).to_pandas()
    from huggingface_hub import HfFileSystem

    path = (f"datasets/google/fleurs@{FLEURS_REVISION}/parquet-data/{config}/"
            "test-00000-of-00001.parquet")  # fmt: skip
    with HfFileSystem().open(path, "rb", block_size=1 << 20) as f:
        return pq.read_table(f, columns=cols).to_pandas()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--parquet-dir", type=Path, default=None)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.archive)
    rows, mismatches = [], 0
    for lang, (config, archive_lang) in LANGS.items():
        refs = references(config, args.parquet_dir)
        for system, run in RUNS.items():
            with zf.open(f"hypotheses/{run.format(lang=archive_lang)}.jsonl") as f:
                hyps = pd.read_json(f, lines=True, dtype={"hypothesis_raw": str})
            if len(hyps) != len(refs) or (refs["id"].to_numpy() != hyps["utt_id"].to_numpy()).any():
                print(f"Error: {system}/{lang}: idx ↔ FLEURS id join failed")
                return 1
            ref, hyp = refs["transcription"].tolist(), hyps["hypothesis_raw"].fillna("").tolist()
            df = score_conventions(ref, hyp)
            primary = df[df["convention"] == "primary"].iloc[0]
            cer_spaces = 100 * cer_jiwer_default(
                [normalize(r) for r in ref], [normalize(h) for h in hyp]
            )
            for conv, w in zip(df["convention"], df["wer"], strict=True):
                paper = PAPER_TABLE3_WER[(system, lang)][conv]
                ok = abs(round(w, 2) - paper) <= TOL
                mismatches += not ok
                match = "yes" if ok else "NO"
                rows.append([system, lang, f"WER {conv}", round(w, 2), paper, match])
            for metric in ("cer", "mer"):
                paper = PAPER_TABLE4_PRIMARY[(system, lang)][metric]
                val = round(float(primary[metric]), 2)
                ok = abs(val - paper) <= TOL
                mismatches += not ok
                rows.append(
                    [system, lang, f"{metric.upper()} primary", val, paper, "yes" if ok else "NO"]
                )
            rows.append([system, lang, "CER primary, jiwer default (spaces)", round(cer_spaces, 2),
                         None, "(not in paper)"])  # fmt: skip
    out = pd.DataFrame(rows, columns=["system", "lang", "metric", "this code", "paper", "match"])
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(out.to_string(index=False))
    n = (out["match"] == "yes").sum()
    print(f"\n{n} of {n + mismatches} paper values reproduced exactly (to two decimals).")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
