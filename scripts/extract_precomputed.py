"""Pull the subset's hypotheses out of the paper's Zenodo release.

The release (https://doi.org/10.5281/zenodo.23010849) stores one JSONL file per
run with **unnormalised** hypotheses for the full FLEURS test splits. Rows are
keyed by ``idx`` (row order of the Hugging Face test split) and ``utt_id``
(the FLEURS sentence id, shared by up to three recordings of one sentence).

For each of the 200 subset clips in ``data/subset_manifest.csv`` this script
takes the hypothesis whose ``idx`` matches, checks that its ``utt_id`` and
duration agree with the manifest, and writes one CSV per system to
``data/precomputed/`` plus ``models.json`` (revisions, precision, seed, source).
Hypotheses are copied verbatim, never normalised. If any clip is missing or
disagrees, it is reported and the script exits with an error instead of
filling the gap.

Usage::

    python scripts/extract_precomputed.py                      # downloads the release (1.1 MB)
    python scripts/extract_precomputed.py --archive NJ105_release_v2.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "precomputed"
MANIFEST = REPO_ROOT / "data" / "subset_manifest.csv"

CONCEPT_DOI = "10.5281/zenodo.23010849"
RECORD_API = "https://zenodo.org/api/records/23010849"  # redirects to the latest version
ARCHIVE_LANG = {"tagalog": "filipino", "cebuano": "cebuano"}

#: system → (output file, run name in the archive, HF model id, precision)
SYSTEMS = {
    "Whisper large-v3": ("whisper-large-v3.csv", "whisper__{lang}__committed",
                         "openai/whisper-large-v3", "float16"),
    "MMS-1B-all": ("mms-1b-all.csv", "mms__{lang}__matched", "facebook/mms-1b-all", "float32"),
    "SeamlessM4T v2 Large": ("seamless-m4t-v2-large.csv", "seamless__{lang}__matched",
                             "facebook/seamless-m4t-v2-large", "float16"),
}  # fmt: skip

CONDITIONS = {
    "Whisper large-v3": "language-committed: language='tl' for both splits "
    "(Whisper has no Cebuano token); byte-identical to the language-agnostic run",
    "MMS-1B-all": "matched adapter: 'tgl' for Tagalog, 'ceb' for Cebuano",
    "SeamlessM4T v2 Large": "matched target language: tgt_lang 'tgl' / 'ceb'",
}


def fetch_archive(dest: Path) -> Path:
    """Download the latest version of the Zenodo release into ``dest``."""
    with urllib.request.urlopen(RECORD_API, timeout=60) as resp:
        record = json.load(resp)
    (file,) = [f for f in record["files"] if f["key"].endswith(".zip")]
    out = dest / file["key"]
    print(f"Downloading {file['key']} from Zenodo record {record['id']} ...")
    urllib.request.urlretrieve(file["links"]["self"], out)
    return out


def read_jsonl(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """One run's hypotheses as a DataFrame indexed by ``idx``."""
    with zf.open(f"hypotheses/{name}.jsonl") as f:
        df = pd.read_json(f, lines=True, dtype={"hypothesis_raw": str})
    return df.set_index("idx", verify_integrity=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--archive", type=Path, default=None, help="local NJ105_release_v*.zip")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    archive = args.archive or fetch_archive(Path(tmp.name))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = pd.read_csv(MANIFEST, keep_default_na=False, dtype={"filename": str})
    zf = zipfile.ZipFile(archive)
    revisions = json.loads(zf.read("model_revisions.json"))
    environment = json.loads(zf.read("environment.json"))

    problems: list[str] = []
    args.out.mkdir(parents=True, exist_ok=True)
    sidecar: dict = {
        "source": {
            "doi": f"https://doi.org/{CONCEPT_DOI}",
            "archive_file": archive.name,
            "archive_sha256": sha,
            "licence": "CC BY-NC 4.0",
        },
        "note": "Hypotheses are unnormalised, exactly as stored in the release. "
        "Rows follow data/subset_manifest.csv (lang, draw_order).",
        "inference": {
            "transformers": environment["packages"]["transformers"],
            "torch": environment["torch"],
            "gpu": environment.get("gpu_name"),
            "seed": 20260814,
            "max_new_tokens_autoregressive": 256,
            "batch": "one utterance per forward pass",
        },
        "systems": {},
    }

    for system, (filename, run, model_id, dtype) in SYSTEMS.items():
        rows = []
        for lang, grp in manifest.groupby("lang", sort=False):
            run_name = run.format(lang=ARCHIVE_LANG[lang])
            hyps = read_jsonl(zf, run_name)
            if system == "Whisper large-v3":
                agnostic = read_jsonl(zf, run_name.replace("committed", "agnostic"))
                if not agnostic["hypothesis_raw"].equals(hyps["hypothesis_raw"]):
                    problems.append(f"{run_name}: committed and agnostic runs differ")
            for r in grp.itertuples():
                if r.idx not in hyps.index:
                    problems.append(
                        f"{system} / {lang}: idx {r.idx} (FLEURS id {r.fleurs_id}) missing"
                    )
                    continue
                h = hyps.loc[r.idx]
                if int(h["utt_id"]) != int(r.fleurs_id):
                    problems.append(f"{run_name} idx {r.idx}: utt_id {h['utt_id']} ≠ {r.fleurs_id}")
                    continue
                if abs(float(h["duration_s"]) - r.num_samples / 16000) > 0.01:
                    problems.append(
                        f"{run_name} idx {r.idx}: duration {h['duration_s']} ≠ manifest"
                    )
                    continue
                if h.get("error") not in (None, "") and not pd.isna(h.get("error")):
                    problems.append(f"{run_name} idx {r.idx}: inference error {h['error']!r}")
                row = {"lang": lang, "idx": r.idx, "fleurs_id": r.fleurs_id,
                       "hypothesis_raw": h["hypothesis_raw"]}  # fmt: skip
                if system == "Whisper large-v3":
                    row["detected_language"] = agnostic.loc[r.idx, "extras"]["detected_language"]
                rows.append(row)
        pd.DataFrame(rows).to_csv(args.out / filename, index=False, lineterminator="\n")
        rev = revisions[model_id]
        sidecar["systems"][system] = {
            "file": filename,
            "hf_model": model_id,
            "revision": rev["revision_at_run"],
            "revision_date": rev["revision_date"],
            "precision": dtype,
            "condition": CONDITIONS[system],
            "archive_runs": [run.format(lang=ARCHIVE_LANG[x]) for x in ("tagalog", "cebuano")],
            "rows": len(rows),
        }
        print(f"{system}: {len(rows)} rows → {args.out / filename}")

    zf.close()
    tmp.cleanup()
    (args.out / "models.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(
            f"\nError: {len(problems)} problem(s); nothing was invented to fill them:",
            file=sys.stderr,
        )
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print("OK: All 200 clips found for all three systems; utt_id and duration agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
