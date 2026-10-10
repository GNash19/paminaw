"""WER/CER/MER against hand-computed worked examples (paper Sects. 3.5 and 3.7).

Each expected value is worked out by hand in the comment beside it, not taken
from the code under test.
"""

import math

import pytest

from paminaw.metrics import (
    EditCounts,
    cer,
    cer_jiwer_default,
    char_counts,
    mer,
    wer,
    word_counts,
)
from paminaw.normalize import normalize


def test_perfect_match():
    assert wer("the cat sat on the mat", "the cat sat on the mat") == 0.0


def test_substitution_and_insertion():
    # cat→dog (S), +down (I) over 3 words; chars: c,a,t→d,o,g (3 S) + d,o,w,n (4 I) over 9
    c = word_counts("the cat sat", "the dog sat down")
    assert (c.S, c.D, c.I, c.C) == (1, 0, 1, 2)
    assert c.error_rate == pytest.approx(2 / 3)
    assert c.mer == pytest.approx(2 / 4)
    assert cer("the cat sat", "the dog sat down") == pytest.approx(7 / 9)


def test_deletions_only():
    assert wer("a b c d", "a c") == pytest.approx(2 / 4)


def test_empty_hypothesis_is_all_deletions_and_is_kept():
    c = word_counts("ang bata ay natutulog", "")
    assert (c.S, c.D, c.I, c.C) == (0, 4, 0, 0)
    assert c.error_rate == 1.0 and c.mer == 1.0
    assert cer("ang bata ay natutulog", "") == 1.0
    # still counted in a corpus: 4 deletions + 0 errors over 4 + 2 words
    assert wer(["ang bata ay natutulog", "oo nga"], ["", "oo nga"]) == pytest.approx(4 / 6)


def test_no_correct_token():
    c = word_counts("unsa ang imong pangalan", "xxx yyy zzz www")
    assert (c.S, c.C) == (4, 0)
    assert c.error_rate == 1.0 and c.mer == 1.0


def test_single_token_right_and_wrong():
    assert wer("oo", "oo") == 0.0
    assert wer("oo", "dili") == 1.0
    # o,o → d,i (2 S) + l,i (2 I) over 2 characters
    assert cer("oo", "dili") == pytest.approx(2.0)


def test_wer_above_100_percent_from_insertions():
    # 4 inserted words over 2 reference words; MER = 4 / (4 + 2)
    ref, hyp = "salamat kaayo", "salamat kaayo nimo bai kuya maam"
    assert wer(ref, hyp) == pytest.approx(2.0)
    assert mer(ref, hyp) == pytest.approx(4 / 6)
    assert cer(ref, hyp) == pytest.approx(15 / 12)  # "nimobaikuyamaam" over "salamatkaayo"


def test_empty_reference_is_undefined_not_zero():
    c = word_counts("", "something here")
    assert c.I == 2 and c.n_ref == 0
    assert math.isnan(c.error_rate)
    assert c.mer == 1.0


def test_corpus_rate_sums_counts_instead_of_averaging():
    refs = ["a", "a b c d e f g h i j"]
    hyps = ["b", "a b c d e f g h i j"]
    # summed: 1 error / 11 words. Averaging per-utterance rates would give (1 + 0) / 2.
    assert wer(refs, hyps) == pytest.approx(1 / 11)


def test_kumusta_wer_and_both_cer_definitions():
    ref, hyp = "kumusta ka na", "kumusta kana"
    assert wer(ref, hyp) == pytest.approx(2 / 3)  # ka→kana (S), na deleted (D), over 3
    c = char_counts(ref, hyp)
    assert (c.errors, c.n_ref) == (0, 11)  # the paper: no characters wrong out of 11
    assert cer(ref, hyp) == 0.0
    assert cer_jiwer_default(ref, hyp) == pytest.approx(1 / 13)  # jiwer: one lost space of 13


def test_maayong_buntag_none_vs_primary():
    ref, hyp = "Maayong buntag, Doc!", "maayong buntag doc"
    assert wer(normalize(ref, "none"), normalize(hyp, "none")) == 1.0
    assert wer(normalize(ref, "casefold"), normalize(hyp, "casefold")) == pytest.approx(2 / 3)
    assert wer(normalize(ref, "primary"), normalize(hyp, "primary")) == 0.0


def test_edit_counts_add():
    assert EditCounts(1, 2, 3, 4) + EditCounts(1, 1, 1, 1) == EditCounts(2, 3, 4, 5)


def test_mismatched_lengths_explain_fix():
    with pytest.raises(ValueError, match="Fix: pass one hypothesis per reference"):
        wer(["a", "b"], ["a"])


def test_jiwer_default_cer_rejects_empty_reference():
    with pytest.raises(ValueError, match="Fix"):
        cer_jiwer_default([""], ["a"])
