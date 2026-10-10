# Running Paminaw with no internet (the external drive)

Colab is the main path: it downloads the clips and models on Google's servers, so slow venue
Wi-Fi rarely matters. But if the internet is down entirely, Colab cannot even load. For that
case the facilitators carry an external drive with everything needed to run the workshop on a
laptop that already has Python.

## What is on the drive

```
D:\
├── paminaw\                     the repository, cloned at tag v1.0.0
└── paminaw_offline\
    ├── paminaw_clips_v1.zip     the 200 clips (172 MB), same file as the GitHub release asset
    ├── clips\fil_ph\*.wav       … already unzipped (100 Tagalog)
    ├── clips\ceb_ph\*.wav       … (100 Cebuano)
    ├── subset_manifest.csv
    ├── hf_home\hub\             a Hugging Face cache with the model weights:
    │                              openai/whisper-small (~1 GB), openai/whisper-tiny (~150 MB),
    │                              facebook/mms-1b-all (~3.9 GB, plus the ceb and tgl adapters)
    └── wheels\                  Python packages for an offline `pip install` (see below)
```

Running the notebooks also needs no network: they read the precomputed outputs from the
`paminaw` package itself.

## Using it on a laptop (Windows, macOS or Linux)

You need **Python 3.10–3.13** already installed.

1. Copy `paminaw\` and `paminaw_offline\` from the drive to the laptop. Running straight from
   the drive works too, but it is slower.
2. Install from the drive's wheels, with no internet. Run this inside the `paminaw` folder:

   ```bash
   pip install --no-index --find-links ../paminaw_offline/wheels -e ".[dev]" notebook
   ```

3. Tell `paminaw` and Hugging Face to use the drive (Windows PowerShell shown; on macOS/Linux
   use `export NAME=value`):

   ```powershell
   $env:PAMINAW_OFFLINE_DIR = "D:\paminaw_offline"
   $env:HF_HOME = "D:\paminaw_offline\hf_home"
   $env:HF_HUB_OFFLINE = "1"
   ```

4. Check, then start Jupyter in the same terminal:

   ```bash
   paminaw check --offline
   jupyter notebook notebooks/paminaw_workshop.ipynb
   ```

   Cell 0 skips the Colab-only install step, finds the clips on the drive and prints
   **✅ READY**. Whisper and MMS load from `hf_home` without downloading anything.

**Do not install `torchcodec` unless FFmpeg is installed.** `transformers` imports TorchCodec
whenever it is present, and without FFmpeg that import crashes every Whisper call. `paminaw`
reads audio with `soundfile` instead, and `paminaw check` warns you if TorchCodec is broken.

## Making (or refreshing) the drive

On a computer with internet, from the repository folder:

```bash
pip install -e ".[dev]"
# 1. clips + model weights (about 5 GB in total)
paminaw prepare-offline --out D:/paminaw_offline --clips-zip path/to/paminaw_clips_v1.zip
# 2. wheels for the laptops you expect (one line per Python version / OS you need)
pip download -d D:/paminaw_offline/wheels -r constraints/colab-2026-10.txt jiwer==4.0.0 \
    soundfile pyarrow ruff==0.17.0 pytest nbclient nbformat ipykernel notebook hatchling \
    --extra-index-url https://download.pytorch.org/whl/cpu
# 3. the repository itself
git clone --branch v1.0.0 https://github.com/GNash19/paminaw.git D:/paminaw
```

`pip download` fetches wheels for the Python version and operating system it runs on. To cover
other laptops, add `--python-version 3.12 --platform win_amd64 --only-binary=:all:` (or
`macosx_11_0_arm64`, `manylinux2014_x86_64`) and run it once per combination.

The scripts `scripts/prepare_offline.py` and `paminaw prepare-offline` do the same job.
