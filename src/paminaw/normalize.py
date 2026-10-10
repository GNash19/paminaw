"""Text normalisation for scoring: the four conventions of the paper.

Golosino, Morales & Adlaon (2026), *Benchmarking Multilingual ASR for Tagalog
and Cebuano across Scoring Conventions and Language Coverage*, Sect. 3.4:

    none      both strings are scored exactly as produced
    casefold  Unicode case folding is applied and whitespace collapsed
    primary   NFC normalization and case folding are applied, characters
              outside the word and whitespace classes are replaced by a
              space, and whitespace is collapsed
    strict    the primary convention is applied and any token containing a
              digit is then removed

Every convention is applied identically to the reference and the hypothesis.
"primary" is the convention behind all of the paper's headline figures.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

__all__ = [
    "CONVENTIONS",
    "normalize",
    "normalize_casefold",
    "normalize_none",
    "normalize_primary",
    "normalize_strict",
]

#: The four conventions, ordered from least to most aggressive (paper Sect. 3.4).
CONVENTIONS: tuple[str, ...] = ("none", "casefold", "primary", "strict")

_NOT_WORD_OR_SPACE = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize_none(text: str | None) -> str:
    """Return the text unchanged (the paper's ``none`` convention, Sect. 3.4).

    ``None`` becomes an empty string so that a missing hypothesis is still scored.
    """
    return "" if text is None else str(text)


def normalize_casefold(text: str | None) -> str:
    """Case-fold the text and collapse whitespace (``casefold``, Sect. 3.4).

    Punctuation is kept and no Unicode normalisation is applied.
    """
    return _WHITESPACE.sub(" ", normalize_none(text).casefold()).strip()


def normalize_primary(text: str | None) -> str:
    """Apply the paper's primary convention (Sect. 3.4).

    NFC, then case folding, then every character that is neither a word
    character nor whitespace becomes a space, then whitespace is collapsed.
    NFC runs first so that a decomposed ``n`` + combining tilde becomes ``ñ``
    (a word character) instead of being split by the punctuation rule.
    """
    text = unicodedata.normalize("NFC", normalize_none(text)).casefold()
    text = _NOT_WORD_OR_SPACE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_strict(text: str | None) -> str:
    """Apply ``primary``, then drop every token that contains a digit (Sect. 3.4).

    FLEURS writes numerals as digits while the systems spell them out, so this
    convention bounds how much of an error rate comes from numerals alone.
    """
    return " ".join(tok for tok in normalize_primary(text).split() if not _DIGIT.search(tok))


_FUNCTIONS: dict[str, Callable[[str | None], str]] = {
    "none": normalize_none,
    "casefold": normalize_casefold,
    "primary": normalize_primary,
    "strict": normalize_strict,
}


def normalize(text: str | None, convention: str = "primary") -> str:
    """Normalise one transcript under a named convention (default: ``primary``).

    >>> normalize("Maayong buntag, Doc!")
    'maayong buntag doc'
    """
    try:
        fn = _FUNCTIONS[convention]
    except KeyError:
        raise ValueError(
            f"Unknown convention {convention!r}. Fix: use one of {', '.join(CONVENTIONS)}."
        ) from None
    return fn(text)
