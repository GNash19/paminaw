"""Regenerate model outputs for the 200 subset clips (optional; meant for an A100).

The workshop ships the paper's own outputs in ``data/precomputed/``. Use this
script to regenerate them, or to add a system, with the paper's settings
(Sect. 3.3): fixed Hub revisions, Whisper and SeamlessM4T in float16, MMS in
float32, one utterance per forward pass, ``max_new_tokens=256``, seed 20260814.
Outputs are written unnormalised, in the same CSV format.

Your outputs may differ slightly from the paper's even with the same revisions,
because the paper ran transformers 4.44.2 and calls ``generate`` directly.

Usage::

    python scripts/precompute_large.py --system whisper-large-v3 --out whisper-large-v3.csv
    python scripts/precompute_large.py --system mms-1b-all --out mms.csv
    python scripts/precompute_large.py --system seamless-m4t-v2-large --out seamless.csv
    python scripts/precompute_large.py --system whisper-small --device cpu \\
        --out data/precomputed/whisper-small.csv      # the Section 3 fallback file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paminaw.asr import (  # noqa: E402
    MMSTranscriber,
    WhisperTranscriber,
    dtype_kwargs,
    pick_device_dtype,
    to_array,
)
from paminaw.data import LANGUAGES, load_subset  # noqa: E402
from paminaw.env import set_seed, versions  # noqa: E402

#: Hub ids and the revisions of the paper (Sect. 3.3) or of the workshop run.
SYSTEMS = {
    "whisper-large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "mms-1b-all": ("facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"),
    "seamless-m4t-v2-large": (
        "facebook/seamless-m4t-v2-large",
        "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
    ),
    "whisper-small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),
    "whisper-tiny": ("openai/whisper-tiny", None),
}


def seamless_runner(model_id: str, revision: str, device: str, dtype: torch.dtype):
    """A function that transcribes one clip with SeamlessM4T v2 (speech → text)."""
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_id, revision=revision, **dtype_kwargs(dtype)
    ).to(device)
    model.eval()

    def run(path: str, lang: str) -> str:
        inputs = processor(audios=to_array(path), sampling_rate=16_000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, tgt_lang=LANGUAGES[lang]["seamless"], max_new_tokens=256)
        return processor.decode(ids[0], skip_special_tokens=True)

    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--n", type=int, default=100, help="clips per language")
    args = ap.parse_args()

    set_seed()
    model_id, revision = SYSTEMS[args.system]
    device = args.device or pick_device_dtype()[0]
    half = device == "cuda" and not args.system.startswith("mms")
    dtype = torch.float16 if half else torch.float32

    if args.system.startswith("whisper"):
        w = WhisperTranscriber(model_id, device=device, dtype=dtype, revision=revision)

        def run(path: str, lang: str) -> str:
            return w.transcribe(path, language="tagalog")  # no Cebuano token in Whisper

    elif args.system.startswith("mms"):
        m = MMSTranscriber(model_id, device=device, revision=revision)

        def run(path: str, lang: str) -> str:
            if m.lang != LANGUAGES[lang]["mms"]:
                m.set_language(str(LANGUAGES[lang]["mms"]))
            return m.transcribe(path)

    else:
        run = seamless_runner(model_id, revision, device, dtype)

    rows, t0, audio_s = [], time.perf_counter(), 0.0
    for lang in ("cebuano", "tagalog"):
        clips = load_subset(lang, n=args.n)
        for i, c in enumerate(clips.itertuples(), 1):
            rows.append({"lang": lang, "idx": c.idx, "fleurs_id": c.fleurs_id,
                         "hypothesis_raw": run(c.path, lang)})  # fmt: skip
            audio_s += c.duration_s
            print(f"\r{lang}: {i}/{len(clips)}", end="", flush=True)
        print()
    secs = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, lineterminator="\n")
    meta = {
        "hf_model": model_id,
        "revision": revision,
        "precision": str(dtype).replace("torch.", ""),
        "device": device if device == "cpu" else torch.cuda.get_device_name(0),
        "language_hint": "tagalog for both languages" if "whisper" in args.system else "matched",
        "max_new_tokens": 256,
        "seed": 20260814,
        "versions": versions(),
        "seconds": round(secs, 1),
        "real_time_factor": round(secs / audio_s, 3),
        "note": "Unnormalised hypotheses for the workshop subset, generated by "
        "scripts/precompute_large.py.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out} in {secs:.0f} s (RTF {meta['real_time_factor']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
