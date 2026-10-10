"""Error rates for speech recognition: WER, CER and MER from edit counts.

Definitions follow the paper (Golosino, Morales & Adlaon 2026, Sect. 3.5):

* WER = (S + D + I) / N, where N is the number of reference words.
  It can exceed 100% when the system inserts many extra words.
* CER is the same computation over **non-whitespace characters**.
* MER = (S + D + I) / (S + D + I + C), bounded by 100% (Morris et al. 2004).
* Corpus rates are computed from edit counts **summed over all utterances**,
  never by averaging per-utterance rates; the two are different estimators.
* An empty hypothesis is scored as all deletions; it is never dropped.

Alignments come from ``jiwer``. Normalise the text first (see
:mod:`paminaw.normalize`); these functions score the strings they are given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer

__all__ = [
    "EditCounts",
    "cer",
    "cer_jiwer_default",
    "char_counts",
    "corpus_counts",
    "mer",
    "wer",
    "word_counts",
]


@dataclass(frozen=True)
class EditCounts:
    """Substitutions (S), deletions (D), insertions (I) and correct tokens (C)."""

    S: int = 0
    D: int = 0
    I: int = 0  # noqa: E741  (the standard name for insertions)
    C: int = 0

    @property
    def n_ref(self) -> int:
        """Number of reference tokens, S + D + C."""
        return self.S + self.D + self.C

    @property
    def errors(self) -> int:
        """Total edit operations, S + D + I."""
        return self.S + self.D + self.I

    @property
    def error_rate(self) -> float:
        """Errors divided by reference tokens (WER for words, CER for characters).

        Returns NaN when the reference is empty, because the rate is undefined.
        """
        return float("nan") if self.n_ref == 0 else self.errors / self.n_ref

    @property
    def mer(self) -> float:
        """Match error rate: errors / (errors + correct). NaN if there is nothing to align."""
        denom = self.errors + self.C
        return float("nan") if denom == 0 else self.errors / denom

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(self.S + other.S, self.D + other.D, self.I + other.I, self.C + other.C)


def _as_list(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return ["" if t is None else str(t) for t in texts]


def _check_pairs(references: list[str], hypotheses: list[str]) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Got {len(references)} references but {len(hypotheses)} hypotheses. "
            "Fix: pass one hypothesis per reference, in the same order."
        )


def _token_counts(ref_tokens: list[list[str]], hyp_tokens: list[list[str]]) -> EditCounts:
    """Sum edit counts over token sequences, handling empty sides explicitly."""
    total = EditCounts()
    keep_ref: list[str] = []
    keep_hyp: list[str] = []
    for r, h in zip(ref_tokens, hyp_tokens, strict=True):
        if not r:
            total += EditCounts(I=len(h))  # nothing to recognise: all output is inserted
        elif not h:
            total += EditCounts(D=len(r))  # empty hypothesis: every reference token deleted
        else:
            keep_ref.append(" ".join(r))
            keep_hyp.append(" ".join(h))
    if keep_ref:
        out = jiwer.process_words(keep_ref, keep_hyp)
        total += EditCounts(out.substitutions, out.deletions, out.insertions, out.hits)
    return total


def word_counts(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> EditCounts:
    """Word-level edit counts summed over the corpus. Words are split on whitespace."""
    refs, hyps = _as_list(references), _as_list(hypotheses)
    _check_pairs(refs, hyps)
    return _token_counts([r.split() for r in refs], [h.split() for h in hyps])


def char_counts(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> EditCounts:
    """Character-level edit counts over non-whitespace characters (paper Sect. 3.5).

    Spaces are removed before aligning, so a merged or split word costs nothing
    at the character level: ``"ka na"`` vs ``"kana"`` has zero character errors.
    """
    refs, hyps = _as_list(references), _as_list(hypotheses)
    _check_pairs(refs, hyps)
    return _token_counts(
        [[c for c in r if not c.isspace()] for r in refs],
        [[c for c in h if not c.isspace()] for h in hyps],
    )


def corpus_counts(
    references: str | Sequence[str], hypotheses: str | Sequence[str]
) -> dict[str, EditCounts]:
    """Word and character edit counts for a corpus, as ``{"word": ..., "char": ...}``."""
    return {
        "word": word_counts(references, hypotheses),
        "char": char_counts(references, hypotheses),
    }


def wer(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> float:
    """Corpus word error rate as a fraction (0.25 means 25%)."""
    return word_counts(references, hypotheses).error_rate


def cer(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> float:
    """Corpus character error rate over non-whitespace characters, as a fraction."""
    return char_counts(references, hypotheses).error_rate


def mer(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> float:
    """Corpus match error rate (word level), as a fraction between 0 and 1."""
    return word_counts(references, hypotheses).mer


def cer_jiwer_default(references: str | Sequence[str], hypotheses: str | Sequence[str]) -> float:
    """CER the way ``jiwer.cer`` computes it by default: spaces count as characters.

    Shown in the workshop next to :func:`cer` because the two definitions give
    different numbers (``"kumusta ka na"`` vs ``"kumusta kana"``: 1/13 here, 0/11
    in the paper). The paper reports the non-whitespace version.
    """
    refs, hyps = _as_list(references), _as_list(hypotheses)
    _check_pairs(refs, hyps)
    if any(not r.strip() for r in refs):
        raise ValueError(
            "jiwer's default CER cannot score an empty reference. "
            "Fix: remove empty references, or use paminaw.metrics.cer instead."
        )
    return float(jiwer.cer(refs, hyps))
