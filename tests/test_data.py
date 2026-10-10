"""Data loading and the download fallback chain, with no network."""

import io
import sys
import zipfile

import pytest

from paminaw import data, env
from paminaw.data import PaminawDataError, ensure_clips, load_manifest, load_subset, resolve_lang


@pytest.mark.parametrize(
    ("alias", "lang"),
    [("tl", "tagalog"), ("fil_ph", "tagalog"), ("Filipino", "tagalog"), ("ceb", "cebuano"),
     ("ceb_ph", "cebuano"), ("Bisaya", "cebuano")],
)  # fmt: skip
def test_resolve_lang(alias, lang):
    assert resolve_lang(alias) == lang


def test_resolve_lang_unknown_explains_fix():
    with pytest.raises(ValueError, match="Fix: use 'tagalog' or 'cebuano'"):
        resolve_lang("ilocano")


@pytest.fixture
def offline_dir(tmp_path):
    """A fake USB folder with an empty WAV placeholder for every manifest clip."""
    for row in load_manifest().itertuples():
        p = tmp_path / "clips" / row.fleurs_config / row.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return tmp_path


def test_load_subset_from_offline_dir(offline_dir):
    df = load_subset("ceb", n=5, offline_dir=offline_dir)
    assert len(df) == 5 and (df["lang"] == "cebuano").all()
    assert list(df["draw_order"]) == [0, 1, 2, 3, 4]
    assert df["path"].iloc[0].startswith(str(offline_dir))


@pytest.mark.parametrize("kwargs", [{"seed": 1}, {"split": "train"}, {"n": 0}, {"n": 101}])
def test_load_subset_rejects_non_workshop_settings(kwargs):
    with pytest.raises(ValueError, match="Fix"):
        load_subset("tagalog", **kwargs)


def test_offline_zip_is_extracted_to_cache(offline_dir, tmp_path_factory):
    usb = tmp_path_factory.mktemp("usb")
    with zipfile.ZipFile(usb / data.CLIPS_ZIP, "w") as zf:
        for p in (offline_dir / "clips").rglob("*.wav"):
            zf.write(p, p.relative_to(offline_dir).as_posix())
    cache = tmp_path_factory.mktemp("cache")
    assert ensure_clips(cache_dir=cache, offline_dir=usb) == cache


def test_incomplete_offline_dir_explains_fix(tmp_path):
    with pytest.raises(PaminawDataError, match="Fix: run `paminaw prepare-offline`"):
        ensure_clips(offline_dir=tmp_path)


def test_safe_extract_blocks_path_traversal(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with zipfile.ZipFile(buf) as zf, pytest.raises(PaminawDataError, match="outside"):
        data._safe_extract(zf, tmp_path)


def test_online_chain_reports_both_failures_with_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("PAMINAW_OFFLINE_DIR", raising=False)

    def hub_fails(cache, manifest):
        raise OSError("no internet")

    monkeypatch.setattr(data, "_extract_from_fleurs_parquet", hub_fails)
    with pytest.raises(PaminawDataError) as err:
        ensure_clips(cache_dir=tmp_path)
    msg = str(err.value)
    assert "GitHub release" in msg and "Hugging Face Hub" in msg and "Fix:" in msg


def test_corrupt_download_is_rejected(tmp_path, monkeypatch):
    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: FakeResp(b"junk"))
    with pytest.raises(PaminawDataError, match="checksum mismatch"):
        data._download_release_zip(tmp_path)


def test_setup_workshop_prints_ready(offline_dir, capsys):
    s = env.setup_workshop(offline_dir=offline_dir)
    out = capsys.readouterr().out
    assert "✅ READY: 200 clips" in out and "workshop subset" in out
    assert s.n_clips == 200 and s.device in ("cpu", "cuda")


def test_setup_workshop_stops_with_fix(tmp_path):
    with pytest.raises(SystemExit, match="Fix"):
        env.setup_workshop(offline_dir=tmp_path)


def test_versions_reads_package_metadata():
    # jiwer 4 has no __version__ attribute; cell 0 must still see it as installed.
    assert env.versions()["jiwer"] == "4.0.0"


def test_broken_torchcodec_is_reported_with_fix(monkeypatch):
    monkeypatch.setattr(env.metadata, "version", lambda name: "0.11.0")
    monkeypatch.setitem(sys.modules, "torchcodec", None)  # makes import torchcodec fail
    assert "Fix: run `pip uninstall -y torchcodec`" in env.torchcodec_problem()
