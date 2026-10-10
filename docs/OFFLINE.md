# Running Paminaw with no internet (the external drive)

Colab is the main path: it downloads the clips and models on Google's servers, so slow venue
Wi-Fi rarely matters. But if the internet is down entirely, Colab cannot even load. For that
case the facilitators carry an external drive with everything needed to run the workshop on a
laptop that already has Python.

## What is on the drive

```
D:\
├── paminaw\                     a clone of the repository (git clone, main branch)
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
# 2. wheels: run once per (platform, Python version) you expect on attendee laptops
pip download -d D:/paminaw_offline/wheels -r constraints/offline-wheels.txt \
    --only-binary=:all: --platform win_amd64 --python-version 3.13
# 3. the repository itself
git clone https://github.com/GNash19/paminaw.git D:/paminaw
```

The drive prepared for PyCon Davao 2026 has wheels for **Windows (x86-64) and macOS (Apple
Silicon, macOS 12 or newer), each for Python 3.12 and 3.13** (macOS wheels were fetched with
`--platform macosx_14_0_arm64`). For other laptops, repeat step 2 with
`--platform manylinux2014_x86_64`, `macosx_10_15_x86_64`, or another `--python-version`.

**Git on an exFAT drive.** exFAT does not record file owners, so git refuses to work inside
`D:\paminaw` ("dubious ownership"). Copy the folder to the laptop first, or run
`git config --global --add safe.directory D:/paminaw` once on that laptop. The notebooks and
`pip install` work either way; only `git` commands are affected.

The scripts `scripts/prepare_offline.py` and `paminaw prepare-offline` do the same job.
