"""The four scoring conventions (paper Sect. 3.4)."""

import pytest

from paminaw.normalize import CONVENTIONS, normalize

DECOMPOSED = "Señor Ñino"  # 'n' + combining tilde, not the single character 'ñ'


def test_order_of_conventions():
    assert CONVENTIONS == ("none", "casefold", "primary", "strict")


def test_none_keeps_text_exactly():
    assert normalize("  Maayong buntag, Doc!  ", "none") == "  Maayong buntag, Doc!  "
    assert normalize(None, "none") == ""


def test_casefold_folds_case_and_collapses_whitespace_but_keeps_punctuation():
    assert normalize("Maayong   buntag,\tDoc!\n", "casefold") == "maayong buntag, doc!"


def test_casefold_is_not_lower():
    assert normalize("STRASSE Straße", "casefold") == "strasse strasse"


def test_primary_worked_example():
    assert normalize("Maayong buntag, Doc!") == "maayong buntag doc"
    assert normalize("Maayong buntag, Doc!", "primary") == normalize("maayong buntag doc")


def test_primary_replaces_punctuation_with_space():
    assert normalize("bag-ohon") == "bag ohon"
    assert normalize("800.kini") == "800 kini"


@pytest.mark.parametrize("text", ["Ñino", "ÑINO", "ñino"])
def test_casefold_handles_enye(text):
    assert normalize(text, "casefold") == "ñino"
    assert normalize(text, "primary") == "ñino"


def test_nfc_composes_enye_before_punctuation_removal():
    # Without NFC the combining tilde is "not a word character" and would split the word.
    assert normalize(DECOMPOSED, "primary") == "señor ñino"
    assert normalize("Señor Ñino", "primary") == normalize(DECOMPOSED, "primary")


def test_casefold_convention_does_not_apply_nfc():
    assert normalize(DECOMPOSED, "casefold") != normalize("Señor Ñino", "casefold")


def test_strict_drops_tokens_containing_digits():
    assert normalize("May 800 na tao sa 2nd floor.", "strict") == "may na tao sa floor"
    assert normalize("sa 2026", "primary") == "sa 2026"
    assert normalize("2026", "strict") == ""


@pytest.mark.parametrize("conv", CONVENTIONS)
def test_conventions_are_idempotent(conv):
    once = normalize("Kumusta, Ñino! 3 ka?", conv)
    assert normalize(once, conv) == once


def test_unknown_convention_explains_fix():
    with pytest.raises(ValueError, match="Fix: use one of"):
        normalize("x", "lowercase")
