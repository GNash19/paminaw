"""The workshop subset: 100 Tagalog + 100 Cebuano clips from FLEURS.

What is in the subset
---------------------
``data/subset_manifest.csv`` lists the clips. It was built by
``scripts/make_subset.py`` from the FLEURS **test** splits (``fil_ph`` for
Tagalog, ``ceb_ph`` for Cebuano) with seed 20260814: 100 distinct sentence IDs
per language are drawn at random, then one recording of each sentence. The
column ``draw_order`` keeps the order of the draw, so the first *n* rows are
themselves a seeded random sample of size *n*. Scores on these clips are
**workshop-subset (100 clips)** scores, not the paper's full-split numbers.

How the audio is obtained (first match wins)
--------------------------------------------
1. ``offline_dir`` (or the ``PAMINAW_OFFLINE_DIR`` environment variable): a
   folder that holds ``clips/`` or ``paminaw_clips_v1.zip`` (see docs/OFFLINE.md).
2. ``cache_dir`` (default ``~/.cache/paminaw`` or ``PAMINAW_CACHE``), if a
   previous run already put the clips there.
3. The release asset ``paminaw_clips_v1.zip`` on GitHub (172 MB): the
   original FLEURS WAV files (16 kHz, 32-bit float), unchanged, plus the manifest.
4. The FLEURS Parquet files on the Hugging Face Hub, read with ``pyarrow``.
   ``google/fleurs`` used to be a script-based dataset, which ``datasets>=4``
   no longer runs; the repository now ships Parquet under
   ``parquet-data/<config>/<split>-00000-of-00001.parquet``. We download that
   file at a pinned revision with ``huggingface_hub`` (no token needed, the
   dataset is not gated) and cut out only the subset's rows. This is slow:
   the Tagalog test file alone is 1.1 GB.

FLEURS is released under CC BY 4.0 by Conneau et al. (2023); see
``data/ATTRIBUTION.md``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

import pandas as pd

__all__ = [
    "CLIPS_SHA256",
    "CLIPS_URL",
    "FLEURS_REVISION",
    "LANGUAGES",
    "SEED",
    "PaminawDataError",
    "data_file",
    "default_cache_dir",
    "ensure_clips",
    "load_manifest",
    "load_precomputed",
    "load_subset",
    "resolve_lang",
]

#: The paper's seed (Sect. 3.3), reused for the subset draw and in the notebooks.
SEED = 20260814

GITHUB_REPO = "GNash19/paminaw"
CLIPS_ZIP = "paminaw_clips_v1.zip"
CLIPS_URL = f"https://github.com/{GITHUB_REPO}/releases/download/v1.0.0/{CLIPS_ZIP}"
#: SHA-256 of the release zip; checked after download.
CLIPS_SHA256 = "b79cd887e745beca2eb0048973ac762e42ed43335d0e3812ef04136710304e82"

FLEURS_REPO = "google/fleurs"
#: Hub commit of google/fleurs used to build the subset.
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"

#: Workshop languages: FLEURS config plus the codes each model uses.
LANGUAGES: dict[str, dict[str, str | None]] = {
    "tagalog": {"fleurs_config": "fil_ph", "whisper": "tagalog", "mms": "tgl", "seamless": "tgl"},
    "cebuano": {"fleurs_config": "ceb_ph", "whisper": None, "mms": "ceb", "seamless": "ceb"},
}

_ALIASES = {
    "tagalog": "tagalog", "tl": "tagalog", "tgl": "tagalog", "fil": "tagalog",
    "filipino": "tagalog", "fil_ph": "tagalog",
    "cebuano": "cebuano", "ceb": "cebuano", "ceb_ph": "cebuano", "bisaya": "cebuano",
    "binisaya": "cebuano",
}  # fmt: skip


class PaminawDataError(RuntimeError):
    """Raised when the workshop data cannot be found or downloaded."""


def resolve_lang(lang: str) -> str:
    """Map a language name or code ("tl", "ceb", "fil_ph", ...) to "tagalog" or "cebuano"."""
    key = str(lang).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown language {lang!r}. Fix: use 'tagalog' or 'cebuano' "
            "(codes like 'tl', 'fil_ph', 'ceb' and 'ceb_ph' also work)."
        )
    return _ALIASES[key]


def data_file(name: str) -> Path:
    """Path to a bundled data file (manifest, precomputed outputs), installed or in the repo."""
    bundled = resources.files("paminaw") / "_data" / name
    if bundled.is_file() or bundled.is_dir():
        return Path(str(bundled))
    repo = Path(__file__).resolve().parents[2] / "data" / name
    if repo.exists():
        return repo
    raise PaminawDataError(
        f"Bundled file {name!r} is missing. Fix: reinstall the package with "
        f"`pip install --force-reinstall git+https://github.com/{GITHUB_REPO}`."
    )


def default_cache_dir() -> Path:
    """Where downloaded clips live: ``$PAMINAW_CACHE`` or ``~/.cache/paminaw``."""
    return Path(os.environ.get("PAMINAW_CACHE", Path.home() / ".cache" / "paminaw"))


def load_manifest(lang: str | None = None) -> pd.DataFrame:
    """The subset manifest (no audio), optionally for one language, in draw order."""
    df = pd.read_csv(
        data_file("subset_manifest.csv"), dtype={"filename": str}, keep_default_na=False
    )
    if lang is not None:
        df = df[df["lang"] == resolve_lang(lang)]
    return df.sort_values(["lang", "draw_order"], kind="stable").reset_index(drop=True)


def _clips_complete(root: Path, manifest: pd.DataFrame) -> bool:
    return all(
        (root / "clips" / c / f).is_file()
        for c, f in zip(manifest["fleurs_config"], manifest["filename"], strict=True)
    )


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if dest not in target.parents and target != dest:
            raise PaminawDataError(
                f"Refusing to extract {member!r}: it points outside {dest}. "
                "Fix: re-download the zip from the GitHub release page."
            )
    zf.extractall(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_release_zip(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / (CLIPS_ZIP + ".part")
    print(f"Downloading the 200 workshop clips (172 MB) from {CLIPS_URL} ...")
    with urllib.request.urlopen(CLIPS_URL, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    if _sha256(tmp) != CLIPS_SHA256:
        tmp.unlink(missing_ok=True)
        raise PaminawDataError(
            "The downloaded clips zip is corrupted (checksum mismatch). "
            "Fix: run the cell again to re-download it."
        )
    with zipfile.ZipFile(tmp) as zf:
        _safe_extract(zf, cache_dir)
    tmp.unlink(missing_ok=True)


def _extract_from_fleurs_parquet(cache_dir: Path, manifest: pd.DataFrame) -> None:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise PaminawDataError(
            "The Hugging Face fallback needs pyarrow and huggingface_hub. "
            "Fix: run `pip install pyarrow huggingface_hub`."
        ) from err
    for config, rows in manifest.groupby("fleurs_config"):
        print(f"Fallback: downloading FLEURS {config} test Parquet (0.5–1.1 GB, be patient) ...")
        local = hf_hub_download(
            FLEURS_REPO,
            f"parquet-data/{config}/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=FLEURS_REVISION,
        )
        table = pq.read_table(local, columns=["id", "path", "audio"])
        out_dir = cache_dir / "clips" / str(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = table.column("id").to_pylist()
        for idx, fleurs_id, filename in zip(
            rows["idx"], rows["fleurs_id"], rows["filename"], strict=True
        ):
            if ids[int(idx)] != int(fleurs_id):
                raise PaminawDataError(
                    f"FLEURS row {idx} of {config} is no longer sentence {fleurs_id}. "
                    "Fix: use the GitHub release zip or an offline folder instead."
                )
            audio = table.column("audio")[int(idx)].as_py()
            (out_dir / str(filename)).write_bytes(audio["bytes"])


def ensure_clips(
    cache_dir: str | Path | None = None,
    offline_dir: str | Path | None = None,
    langs: tuple[str, ...] = ("tagalog", "cebuano"),
) -> Path:
    """Make sure the subset's WAV files exist locally; return the folder holding ``clips/``."""
    manifest = load_manifest()
    manifest = manifest[manifest["lang"].isin([resolve_lang(x) for x in langs])]

    offline = offline_dir or os.environ.get("PAMINAW_OFFLINE_DIR")
    if offline:
        offline = Path(offline)
        if _clips_complete(offline, manifest):
            return offline
        if (offline / CLIPS_ZIP).is_file():
            cache = Path(cache_dir) if cache_dir else default_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(offline / CLIPS_ZIP) as zf:
                _safe_extract(zf, cache)
            if _clips_complete(cache, manifest):
                return cache
        raise PaminawDataError(
            f"The offline folder {offline} has no complete clips/ folder or {CLIPS_ZIP}. "
            "Fix: run `paminaw prepare-offline` for that folder again (see docs/OFFLINE.md)."
        )

    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    if _clips_complete(cache, manifest):
        return cache

    errors = []
    try:
        _download_release_zip(cache)
        if _clips_complete(cache, manifest):
            return cache
        errors.append("release zip downloaded but some clips were missing")
    except Exception as err:  # network, HTTP 404, corrupt zip
        errors.append(f"GitHub release: {err}")
        print(f"Warning: GitHub download failed ({err}); trying the Hugging Face Hub instead.")
    try:
        _extract_from_fleurs_parquet(cache, manifest)
        if _clips_complete(cache, manifest):
            return cache
        errors.append("Hub fallback finished but some clips were missing")
    except Exception as err:
        errors.append(f"Hugging Face Hub: {err}")
    raise PaminawDataError(
        "Could not get the workshop clips (" + "; ".join(errors) + "). "
        "Fix: check this runtime's internet, or set PAMINAW_OFFLINE_DIR to an offline folder."
    )


def load_subset(
    lang: str,
    n: int = 100,
    split: str = "test",
    seed: int = SEED,
    cache_dir: str | Path | None = None,
    offline_dir: str | Path | None = None,
) -> pd.DataFrame:
    """The first ``n`` clips of the workshop subset for one language, with local file paths.

    Returns a DataFrame with columns ``idx`` (row in the FLEURS test split),
    ``fleurs_id``, ``lang``, ``transcription`` (the reference text, lowercased
    and unpunctuated by FLEURS), ``raw_transcription``, ``duration_s`` and
    ``path``. Clips are downloaded on first use (see the module docstring).
    """
    if split != "test" or seed != SEED:
        raise ValueError(
            f"The workshop subset is fixed to split='test' and seed={SEED}. "
            "Fix: keep the defaults, or build your own subset with scripts/make_subset.py --seed."
        )
    if not 1 <= n <= 100:
        raise ValueError(f"n must be between 1 and 100, got {n}. Fix: pick a smaller number.")
    lang = resolve_lang(lang)
    root = ensure_clips(cache_dir=cache_dir, offline_dir=offline_dir, langs=(lang,))
    df = load_manifest(lang).head(n).copy()
    df["path"] = [
        str(root / "clips" / c / f)
        for c, f in zip(df["fleurs_config"], df["filename"], strict=True)
    ]
    return df.reset_index(drop=True)


#: File names of the precomputed outputs under ``data/precomputed/``.
PRECOMPUTED = {
    "Whisper large-v3": "whisper-large-v3.csv",
    "MMS-1B-all": "mms-1b-all.csv",
    "SeamlessM4T v2 Large": "seamless-m4t-v2-large.csv",
}


def load_precomputed(system: str | None = None) -> pd.DataFrame:
    """Unnormalised hypotheses from the paper's runs for the 200 subset clips.

    Returns long format: ``system``, ``lang``, ``idx``, ``fleurs_id``,
    ``hypothesis``. Licensed CC BY-NC 4.0 (see ``data/ATTRIBUTION.md``).
    """
    names = PRECOMPUTED if system is None else {system: PRECOMPUTED[system]}
    frames = []
    for name, filename in names.items():
        df = pd.read_csv(data_file(f"precomputed/{filename}"), keep_default_na=False)
        df = df.rename(columns={"hypothesis_raw": "hypothesis"})
        df.insert(0, "system", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
