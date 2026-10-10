"""Score several systems under all four conventions and compare their rankings.

The same stored hypotheses are re-scored under each convention of
:mod:`paminaw.normalize`; nothing is re-run. This is the analysis of the
paper's Sect. 4.2 (Table 3), here applied to whatever clips you pass in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import jiwer
import pandas as pd

from .metrics import char_counts, word_counts
from .normalize import CONVENTIONS, normalize

__all__ = [
    "PAPER_TABLE3_WER",
    "PAPER_TABLE4_PRIMARY",
    "convention_table",
    "ranking_table",
    "score_conventions",
    "score_systems",
    "top_substitutions",
]

#: Paper Table 3: WER (%) on the FULL FLEURS test splits (964 Tagalog, 541 Cebuano
#: utterances) under each convention. Not comparable with workshop-subset scores.
PAPER_TABLE3_WER: dict[tuple[str, str], dict[str, float]] = {
    ("Whisper large-v3", "tagalog"): {
        "none": 27.11,
        "casefold": 19.12,
        "primary": 12.23,
        "strict": 12.14,
    },
    ("Whisper large-v3", "cebuano"): {
        "none": 56.69,
        "casefold": 49.56,
        "primary": 43.32,
        "strict": 43.47,
    },
    ("MMS-1B-all", "tagalog"): {
        "none": 13.00,
        "casefold": 13.00,
        "primary": 12.50,
        "strict": 11.87,
    },
    ("MMS-1B-all", "cebuano"): {
        "none": 14.47,
        "casefold": 14.47,
        "primary": 13.62,
        "strict": 13.46,
    },
    ("SeamlessM4T v2 Large", "tagalog"): {
        "none": 25.26,
        "casefold": 21.09,
        "primary": 20.51,
        "strict": 20.41,
    },
    ("SeamlessM4T v2 Large", "cebuano"): {
        "none": 41.68,
        "casefold": 37.59,
        "primary": 37.44,
        "strict": 37.76,
    },
}

#: Paper Table 4: primary-convention WER, CER (non-whitespace) and MER (%) on the full splits.
PAPER_TABLE4_PRIMARY: dict[tuple[str, str], dict[str, float]] = {
    ("Whisper large-v3", "tagalog"): {"wer": 12.23, "cer": 4.73, "mer": 12.06},
    ("Whisper large-v3", "cebuano"): {"wer": 43.32, "cer": 13.04, "mer": 40.99},
    ("MMS-1B-all", "tagalog"): {"wer": 12.50, "cer": 3.47, "mer": 12.34},
    ("MMS-1B-all", "cebuano"): {"wer": 13.62, "cer": 4.32, "mer": 13.50},
    ("SeamlessM4T v2 Large", "tagalog"): {"wer": 20.51, "cer": 10.46, "mer": 19.67},
    ("SeamlessM4T v2 Large", "cebuano"): {"wer": 37.44, "cer": 15.95, "mer": 35.58},
}  # fmt: skip


def score_conventions(
    references: Sequence[str],
    hypotheses: Sequence[str],
    conventions: Sequence[str] = CONVENTIONS,
) -> pd.DataFrame:
    """Corpus WER, CER and MER (in %) plus S/D/I/C counts for one system under each convention."""
    rows = []
    for conv in conventions:
        refs = [normalize(r, conv) for r in references]
        hyps = [normalize(h, conv) for h in hypotheses]
        w, c = word_counts(refs, hyps), char_counts(refs, hyps)
        rows.append({
            "convention": conv,
            "n_utts": len(refs),
            "wer": 100 * w.error_rate,
            "cer": 100 * c.error_rate,
            "mer": 100 * w.mer,
            "S": w.S, "D": w.D, "I": w.I, "C": w.C,
            "n_ref_words": w.n_ref,
        })  # fmt: skip
    return pd.DataFrame(rows)


def score_systems(
    references: pd.DataFrame,
    hypotheses: pd.DataFrame,
    conventions: Sequence[str] = CONVENTIONS,
    subset_label: str = "workshop subset (100 clips)",
) -> pd.DataFrame:
    """Score every (system, language) pair under every convention.

    ``references`` needs columns ``lang``, ``idx``, ``transcription``.
    ``hypotheses`` needs columns ``system``, ``lang``, ``idx``, ``hypothesis``
    (as returned by :func:`paminaw.data.load_precomputed`). Rows are matched on
    ``(lang, idx)``; a hypothesis with no matching reference is an error.
    Returns one tidy row per (system, lang, convention).
    """
    for name, df, cols in [
        ("references", references, ["lang", "idx", "transcription"]),
        ("hypotheses", hypotheses, ["system", "lang", "idx", "hypothesis"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}. Fix: add them before scoring.")
    merged = hypotheses.merge(
        references[["lang", "idx", "transcription"]], on=["lang", "idx"], how="left"
    )
    if merged["transcription"].isna().any():
        bad = merged[merged["transcription"].isna()][["system", "lang", "idx"]].head()
        raise ValueError(
            f"Some hypotheses have no reference, e.g.\n{bad}\n"
            "Fix: score only clips that are in the references table."
        )
    frames = []
    for (system, lang), grp in merged.groupby(["system", "lang"], sort=False):
        df = score_conventions(
            grp["transcription"].tolist(), grp["hypothesis"].tolist(), conventions
        )
        df.insert(0, "lang", lang)
        df.insert(0, "system", system)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["subset"] = subset_label
    return out


def convention_table(scores: pd.DataFrame, metric: str = "wer") -> pd.DataFrame:
    """Rows = (system, lang), columns = conventions, plus sensitivity (max − min), as in Table 3."""
    table = scores.pivot_table(index=["system", "lang"], columns="convention", values=metric)
    table = table[[c for c in CONVENTIONS if c in table.columns]]
    table["sensitivity"] = table.max(axis=1) - table.min(axis=1)
    return table.round(2)


def ranking_table(scores: pd.DataFrame, metric: str = "wer") -> pd.DataFrame:
    """For each language and convention, the systems ordered best to worst (1 = lowest error)."""
    ranks = scores.copy()
    ranks["rank"] = ranks.groupby(["lang", "convention"])[metric].rank(method="min").astype(int)
    table = ranks.pivot_table(index=["lang", "convention"], columns="system", values="rank")
    order = [c for c in CONVENTIONS if c in scores["convention"].unique()]
    table = table.reindex(
        pd.MultiIndex.from_product([sorted(scores["lang"].unique()), order]), axis=0
    )
    table.index.names = ["lang", "convention"]
    return table.astype(int)


def top_substitutions(
    references: Sequence[str],
    hypotheses: Sequence[str],
    convention: str = "primary",
    n: int = 10,
) -> pd.DataFrame:
    """Most frequent (reference word → hypothesis word) substitutions after normalising.

    Uses the word alignment, as in the paper's Sect. 4.6 (e.g. Cebuano "nga"
    written as Tagalog "na" by Whisper).
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for ref, hyp in zip(references, hypotheses, strict=True):
        r, h = normalize(ref, convention), normalize(hyp, convention)
        if not r.split() or not h.split():
            continue
        out = jiwer.process_words(r, h)
        rt, ht = out.references[0], out.hypotheses[0]
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    pairs[(rt[chunk.ref_start_idx + i], ht[chunk.hyp_start_idx + i])] += 1
    rows = [{"reference": a, "hypothesis": b, "count": k} for (a, b), k in pairs.most_common(n)]
    return pd.DataFrame(rows, columns=["reference", "hypothesis", "count"])
