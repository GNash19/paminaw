# Paminaw! Teaching Machines to Listen in Cebuano and Tagalog

*Paminaw* is Cebuano for "listen". In two hours you will load Tagalog and Cebuano speech into PyTorch,
transcribe it with open speech-recognition models, and learn why the way you score a transcript
can change which model looks best.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GNash19/paminaw/blob/v1.0.0/notebooks/paminaw_workshop.ipynb)
[![CI](https://github.com/GNash19/paminaw/actions/workflows/ci.yml/badge.svg)](https://github.com/GNash19/paminaw/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23010849.svg)](https://doi.org/10.5281/zenodo.23010849)

**PyCon Davao 2026** · Workshop 2 (MinNa LProc) · Saturday 17 October 2026, 1:10–3:10 PM ·
Ateneo de Davao University · Facilitators: **Nash Golosino** and **Aurora Manseras**

## Start here in 3 steps

1. **Open the notebook in Colab:** click the **Open in Colab** badge above (or scan the QR code on
   the slides). Sign in with your Google account.
2. **Run cell 0:** click into it and press **Shift+Enter**. Wait for **✅ READY** (about a
   minute). If something goes wrong, the message tells you how to fix it.
3. **Follow the sections:** Lab 1 → Lab 2 → Lab 3, top to bottom. Each lab starts with its own
   setup cell, so you can jump ahead if you fall behind.

No installation is needed on your laptop, and a free Colab CPU runtime is enough.

## What you will learn

1. Explain how speech is represented as data (waveforms, sample rate, spectrograms) and the
   stages of an ASR pipeline.
2. Compare PyTorch and TensorFlow for speech processing.
3. Describe the challenges of Cebuano and Tagalog, including code-switching with English.
4. Load recordings as tensors, resample to 16 kHz and compute mel spectrograms with torchaudio
   in Colab.
5. Run pretrained multilingual ASR in PyTorch, with device, precision and inference-mode choices.
6. Score with WER and CER, and explain how normalisation changes scores and rankings.
7. Identify where models fall short and what fine-tuning involves.

## Workshop schedule

| Time | What |
|---|---|
| 1:10–1:40 | Overview talk |
| 1:40–1:55 | Q&A, while everyone opens the notebook and runs cell 0 |
| 1:55–2:15 | **Lab 1 · Load & see**: tensors, resampling, mel spectrograms |
| 2:15–2:45 | **Lab 2 · Transcribe**: Whisper on Tagalog and Cebuano (+ **Lab 2b**: MMS on a GPU runtime) |
| 2:45–3:05 | **Lab 3 · Score**: WER, CER and four scoring conventions |
| 3:05–3:10 | Wrap-up |

## What you need

- A laptop with a charged battery
- A web browser (Chrome, Edge, Firefox or Safari)
- A Google account, for Colab
- **Earphones**, for the listening exercise

Details for before the event: [docs/ATTENDEE_SETUP.md](docs/ATTENDEE_SETUP.md).

## Running it on your own computer

You need Python 3.10 or newer.

```bash
git clone https://github.com/GNash19/paminaw.git
cd paminaw
pip install -e ".[dev]"
paminaw check
jupyter notebook notebooks/paminaw_workshop.ipynb
```

On a laptop without FFmpeg, do **not** install `torchcodec`. `paminaw` reads audio with
`soundfile` instead, and `paminaw check` warns you if a broken `torchcodec` would stop Whisper
from running. To run with **no internet** from the workshop's external drive, see
[docs/OFFLINE.md](docs/OFFLINE.md).

## Repository map

```
notebooks/paminaw_workshop.ipynb    ← the notebook the QR code opens (Labs 0–3)
notebooks/01_… 02_… 03_…            standalone copy of each lab, for later study
notebooks/solutions/                the same notebooks with exercises filled in
src/paminaw/                        the small helper package the notebooks install
  audio.py      load, mono, 16 kHz, mel spectrogram   normalize.py  the four scoring conventions
  data.py       the 100+100 clip subset, downloads    metrics.py    WER, CER, MER from edit counts
  asr.py        Whisper and MMS wrappers              compare.py    four-convention tables, rankings
  plotting.py   waveform, spectrogram, bar charts     cli.py        paminaw check / score / prepare-offline
data/subset_manifest.csv            which FLEURS clips we use, with reference text (no audio)
data/precomputed/                   the paper's model outputs for those clips (CC BY-NC 4.0)
scripts/                            how the subset, the precomputed files and the notebooks are made
docs/                               attendee setup, facilitator guide, offline guide
tests/                              offline tests (pytest), run on every push
```

## The research behind it

The workshop is based on Golosino, Morales & Adlaon (2026), *Benchmarking Multilingual ASR for
Tagalog and Cebuano across Scoring Conventions and Language Coverage* (University of the
Immaculate Conception). The paper evaluates three open systems on the **full FLEURS test
splits** (964 Tagalog and 541 Cebuano utterances). Its word error rates (%) are:

| System | Scoring convention | Tagalog | Cebuano |
|---|---|---:|---:|
| Whisper large-v3 | primary | 12.23 | 43.32 |
| MMS-1B-all | primary | 12.50 | 13.62 |
| SeamlessM4T v2 Large | primary | 20.51 | 37.44 |
| Whisper large-v3 | none (raw text) | 27.11 | 56.69 |

The same Whisper transcripts score 27.11 or 12.23 on Tagalog depending only on how the text is
normalised before scoring (paper Table 3). That one choice is enough to move Whisper from last
place to first. On Cebuano, a language Whisper has no token for, MMS beats it under every
convention.

In the notebooks you score the same systems on the **workshop subset (100 clips per
language)**. Those numbers come out close to these, but they are not the paper's numbers.
`scripts/reproduce_paper_tables.py` re-derives every WER, CER and MER in the paper's Tables 3
and 4 from the released hypotheses with this package's scoring code, and all 36 values match
exactly.

## Data and model licences

| What | Licence |
|---|---|
| Code in this repository | [MIT](LICENSE) |
| FLEURS audio and transcriptions (`google/fleurs`, Conneau et al. 2023) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `data/precomputed/` paper outputs (from the [Zenodo release](https://doi.org/10.5281/zenodo.23010849)) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial research use |
| `openai/whisper-small`, `openai/whisper-large-v3` | Apache-2.0 |
| `facebook/mms-1b-all`, `facebook/seamless-m4t-v2-large` | CC BY-NC 4.0 |

Full attribution: [data/ATTRIBUTION.md](data/ATTRIBUTION.md). The workshop uses only FLEURS read
speech. It never uses patient or clinical recordings.

## How to cite

Use GitHub's **Cite this repository** button, which reads [CITATION.cff](CITATION.cff). For the
research, cite the paper and its data release:

> Golosino, N., Morales, M. N., & Adlaon, K. M. (2026). *Data and Code Release for "Benchmarking
> Multilingual ASR for Tagalog and Cebuano Across Scoring Conventions and Language Coverage"*
> (NLPIR 2026). Zenodo. https://doi.org/10.5281/zenodo.23010849

## Acknowledgements

- **DurianPy** and the **PyCon Davao 2026** organisers, volunteers and speaker-care team.
- The **FLEURS** authors (Conneau et al., 2023) for releasing read speech in 102 languages,
  Cebuano and Filipino among them, under an open licence.
- The teams behind Whisper, MMS, SeamlessM4T, PyTorch, torchaudio, Hugging Face `transformers`
  and `jiwer`.
- Ma. Nicole B. Morales and Kristine Mae Adlaon, co-authors of the paper, University of the
  Immaculate Conception.

## Contact

Questions, bugs or ideas? Please open an issue:
<https://github.com/GNash19/paminaw/issues>.
