"""Scoring several systems under all conventions, ranking flips, bundled data."""

import pandas as pd
import pytest

from paminaw.compare import convention_table, ranking_table, score_systems, top_substitutions
from paminaw.data import load_manifest, load_precomputed

REFS = pd.DataFrame({
    "lang": ["cebuano", "cebuano"],
    "idx": [0, 1],
    "transcription": ["maayong buntag doc", "salamat kaayo"],
})  # fmt: skip
HYPS = pd.DataFrame({
    "system": ["Polite", "Polite", "Plain", "Plain"],
    "lang": ["cebuano"] * 4,
    "idx": [0, 1, 0, 1],
    # Polite is right but capitalised and punctuated; Plain is lowercase with one real error.
    "hypothesis": ["Maayong buntag, Doc!", "Salamat kaayo.", "maayong buntag dok", "salamat kaayo"],
})  # fmt: skip


@pytest.fixture
def scores():
    return score_systems(REFS, HYPS, subset_label="toy subset (2 clips)")


def test_tidy_shape_and_label(scores):
    assert len(scores) == 2 * 4
    assert set(scores["convention"]) == {"none", "casefold", "primary", "strict"}
    assert (scores["subset"] == "toy subset (2 clips)").all()


def test_values(scores):
    get = scores.set_index(["system", "convention"])["wer"]
    assert get[("Polite", "none")] == pytest.approx(100 * 5 / 5)  # every token differs
    assert get[("Polite", "primary")] == 0.0
    assert get[("Plain", "none")] == pytest.approx(100 * 1 / 5)
    assert get[("Plain", "primary")] == pytest.approx(100 * 1 / 5)


def test_ranking_flips_between_conventions(scores):
    ranks = ranking_table(scores)
    assert ranks.loc[("cebuano", "none"), "Plain"] == 1
    assert ranks.loc[("cebuano", "primary"), "Polite"] == 1


def test_convention_table_sensitivity(scores):
    table = convention_table(scores)
    assert table.loc[("Polite", "cebuano"), "sensitivity"] == pytest.approx(100.0)
    assert table.loc[("Plain", "cebuano"), "sensitivity"] == pytest.approx(0.0)


def test_missing_reference_is_an_error():
    bad = HYPS.assign(idx=[0, 1, 0, 99])
    with pytest.raises(ValueError, match="Fix: score only clips"):
        score_systems(REFS, bad)


def test_missing_column_is_an_error():
    with pytest.raises(ValueError, match="missing columns"):
        score_systems(REFS.drop(columns="transcription"), HYPS)


def test_top_substitutions_finds_cebuano_to_tagalog_linker():
    refs = ["dako nga balay", "gamay nga iro"]
    hyps = ["dako na balay", "gamay na iro"]
    top = top_substitutions(refs, hyps)
    assert top.iloc[0].to_dict() == {"reference": "nga", "hypothesis": "na", "count": 2}


def test_manifest_is_100_distinct_sentences_per_language():
    m = load_manifest()
    assert m.groupby("lang").size().to_dict() == {"cebuano": 100, "tagalog": 100}
    assert m.groupby("lang")["fleurs_id"].nunique().to_dict() == {"cebuano": 100, "tagalog": 100}
    assert sorted(m[m.lang == "tagalog"]["draw_order"]) == list(range(100))


def test_precomputed_covers_the_subset_and_is_unnormalised():
    pre = load_precomputed()
    assert pre.groupby("system").size().to_dict() == {
        "MMS-1B-all": 200, "SeamlessM4T v2 Large": 200, "Whisper large-v3": 200,
    }  # fmt: skip
    keys = set(zip(load_manifest()["lang"], load_manifest()["idx"], strict=True))
    assert set(zip(pre["lang"], pre["idx"], strict=True)) == keys
    whisper = pre[pre.system == "Whisper large-v3"]["hypothesis"]
    assert whisper.str.contains(r"[A-Z]").any() and whisper.str.contains(r"[.,]").any()


def test_scoring_precomputed_runs_end_to_end():
    s = score_systems(load_manifest(), load_precomputed())
    assert len(s) == 3 * 2 * 4
    assert s["wer"].between(0, 200).all()
