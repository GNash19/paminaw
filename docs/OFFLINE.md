# Running Paminaw without internet

Colab is the easiest way to follow the workshop. It needs internet, but the clips and models
download on Google's servers, not through your laptop. This page is for running everything on
your own computer **with no internet at all**, from an *offline folder* that holds the clips,
the model weights and the Python packages.

## What an offline folder contains

```
paminaw/                 the repository (git clone)
paminaw_offline/
├── paminaw_clips_v1.zip the 200 clips (172 MB), same file as the GitHub release asset
├── clips/fil_ph/*.wav   … already unzipped (100 Tagalog)
├── clips/ceb_ph/*.wav   … (100 Cebuano)
├── subset_manifest.csv
├── hf_home/hub/         a Hugging Face cache with the model weights:
│                          openai/whisper-small (~1 GB), openai/whisper-tiny (~150 MB),
│                          facebook/mms-1b-all (~3.9 GB, plus the ceb and tgl adapters)
└── wheels/              Python packages for an offline `pip install`
```

The precomputed outputs used in Labs 2b and 3 ship inside the `paminaw` package itself.

## Using it

You need **Python 3.10–3.13** already installed. Below, `OFFLINE` is the path of your
`paminaw_offline` folder.

1. Install from the folder's wheels, with no internet. Run this inside the `paminaw` folder:

   ```bash
   pip install --no-index --find-links OFFLINE/wheels -e ".[dev]" notebook
   ```

2. Tell `paminaw` and Hugging Face to use the folder (Windows PowerShell shown; on macOS/Linux
   use `export NAME=value`):

   ```powershell
   $env:PAMINAW_OFFLINE_DIR = "OFFLINE"
   $env:HF_HOME = "OFFLINE\hf_home"
   $env:HF_HUB_OFFLINE = "1"
   ```

3. Check, then start Jupyter in the same terminal:

   ```bash
   paminaw check --offline
   jupyter notebook notebooks/paminaw_workshop.ipynb
   ```

   Cell 0 skips the Colab-only install step, finds the clips in the folder and prints
   **READY**. Whisper and MMS load from `hf_home` without downloading anything.

**Do not install `torchcodec` unless FFmpeg is installed.** `transformers` imports TorchCodec
whenever it is present, and without FFmpeg that import crashes every Whisper call. `paminaw`
reads audio with `soundfile` instead, and `paminaw check` warns you if TorchCodec is broken.

## Preparing an offline folder yourself

On a computer with internet, from the repository folder:

```bash
pip install -e ".[dev]"
# 1. clips + model weights (about 5 GB in total)
paminaw prepare-offline --out OFFLINE
# 2. Python packages, once per (platform, Python version) you need, e.g. Windows + Python 3.13:
pip download -d OFFLINE/wheels -c constraints/colab-2026-10.txt torch torchaudio transformers \
    datasets huggingface_hub jiwer soundfile numpy pandas matplotlib pyarrow pytest nbclient \
    nbformat ipykernel ipywidgets notebook hatchling editables ruff==0.17.0 \
    --only-binary=:all: --platform win_amd64 --python-version 3.13
```

For other computers, change `--platform` (`macosx_14_0_arm64`, `manylinux2014_x86_64`, …) and
`--python-version`.
