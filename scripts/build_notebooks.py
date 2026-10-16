"""Generate every workshop notebook from one source, and optionally run them.

One definition of each cell feeds eight notebooks, so they cannot drift apart:

    notebooks/paminaw_workshop.ipynb      sections 0–3 (the QR-code notebook)
    notebooks/01_load_and_see.ipynb       setup + section 1
    notebooks/02_transcribe.ipynb         setup + sections 2 and 2b
    notebooks/03_score.ipynb              setup + section 3 + wrap-up
    notebooks/solutions/<same names>      exercises filled in

Usage::

    python scripts/build_notebooks.py                     # write notebooks without outputs
    python scripts/build_notebooks.py --execute           # write, run and save outputs
    python scripts/build_notebooks.py --execute --smoke   # 2 clips, whisper-tiny, no MMS
    python scripts/build_notebooks.py --keep-outputs      # new sources, saved outputs kept
    python scripts/build_notebooks.py --keep-outputs --graft-2b colab_t4_run.ipynb
        # copy Lab 2b outputs from a real GPU run (cells tagged "lab2b")
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = "GNash19/paminaw"
TAG = "v1.0.0"
ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
ZENODO = "https://doi.org/10.5281/zenodo.23010849"
MMS_BLOG = "https://huggingface.co/blog/mms_adapters"


def colab_url(relpath: str) -> str:
    """Colab link for a notebook path inside the repo, pinned to the release tag."""
    return f"https://colab.research.google.com/github/{REPO}/blob/{TAG}/notebooks/{relpath}"


class Cell:
    """One notebook cell; ``only`` limits it to the exercise or the solution version."""

    def __init__(self, kind: str, cid: str, src: str, only: str | None = None, tags=()):
        self.kind, self.cid, self.src, self.only, self.tags = (
            kind,
            cid,
            dedent(src).strip(),
            only,
            tags,
        )

    def build(self):
        make = new_code_cell if self.kind == "code" else new_markdown_cell
        cell = make(self.src, id=self.cid)
        if self.tags:
            cell.metadata["tags"] = list(self.tags)
        return cell


def md(cid, src, only=None, tags=()):
    return Cell("markdown", cid, src, only, tags)


def code(cid, src, only=None, tags=()):
    return Cell("code", cid, src, only, tags)


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

SCHEDULE = """
| Time | What |
|---|---|
| 1:10–1:40 | Overview talk |
| 1:40–1:55 | Q&A: open this notebook and run **cell 0** |
| 1:55–2:15 | **Lab 1** · Load & see |
| 2:15–2:45 | **Lab 2** · Transcribe (+ Lab 2b, MMS on a GPU runtime) |
| 2:45–3:05 | **Lab 3** · Score |
| 3:05–3:10 | Wrap-up |
"""


def title_main(relpath: str, solutions: bool) -> list[Cell]:
    sol = " (solutions)" if solutions else ""
    template = dedent(
        f"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(relpath)})

            # Paminaw! Teaching Machines to Listen in Cebuano and Tagalog{sol}

            **Speech Processing Using PyTorch** · PyCon Davao 2026 · Workshop 2 (MinNa LProc)<br>
            Saturday 17 October 2026, 1:10–3:10 PM · Ateneo de Davao University<br>
            Facilitators: **Nash Golosino** and **Aurora Manseras**
            @@SCHEDULE@@
            **What you'll do.** You will load real Tagalog and Cebuano recordings from the FLEURS
            dataset as PyTorch tensors, look at them as waveforms and mel spectrograms, and hear what
            is lost at telephone quality. Then you will run OpenAI's Whisper on them, find out what it
            does with Cebuano, a language it has no token for, and score the transcripts with WER and
            CER. Along the way you will discover that the way text is cleaned up before scoring can
            change which model "wins".

            > **How to use this notebook.** Click a code cell and press **Shift+Enter**. Run cells
            > from top to bottom. Every lab starts with its own *setup* cell, so if you fall behind,
            > jump to the next lab and start there. A free CPU runtime is enough; a GPU runtime
            > (Runtime → Change runtime type → T4 GPU) also runs Lab 2b live.

            > All scores here are on the **workshop subset (100 clips per language)** of the FLEURS
            > test split. They are **not** the paper's full-split numbers.
            """
    )
    return [md("title", template.replace("@@SCHEDULE@@", SCHEDULE.strip() + "\n"))]


def title_section(relpath: str, name: str, blurb: str, solutions: bool) -> list[Cell]:
    sol = " (solutions)" if solutions else ""
    return [
        md(
            "title",
            f"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(relpath)})

            # Paminaw! · {name}{sol}

            A standalone copy of one lab from the PyCon Davao 2026 workshop *Speech Processing Using
            PyTorch* (Nash Golosino and Aurora Manseras), for study after the event. {blurb}

            Run **cell 0** first, then go from top to bottom with **Shift+Enter**. Scores are on the
            **workshop subset (100 clips per language)**, not the paper's full-split numbers.
            """,
        )
    ]


# ---------------------------------------------------------------------------
# Section 0 · setup
# ---------------------------------------------------------------------------

SETUP = [
    md(
        "s0-md",
        """
        ## 0 · Setup: run me first

        This cell installs the workshop package `paminaw`, prints the versions of the libraries we
        use, checks for a GPU and downloads the 200 audio clips. Wait for the **READY** line
        (about a minute). If something fails, the message tells you how to fix it.
        """,
    ),
    code(
        "s0-setup",
        f"""
        # Cell 0 · setup
        import os
        import subprocess
        import sys

        PAMINAW_REF = "{TAG}"  # the workshop release on GitHub
        SMOKE_TEST = os.environ.get("PAMINAW_SMOKE") == "1"  # quick test mode: 2 clips, whisper-tiny

        if "google.colab" in sys.modules:
            print(f"Installing paminaw {{PAMINAW_REF}} ...")
            url = f"paminaw @ git+https://github.com/{REPO}@{{PAMINAW_REF}}"
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", url],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-1500:])
                raise SystemExit("Error: Could not install paminaw. Fix: check that Colab says "
                                 "'Connected' (top right), then run this cell again.")
        try:
            import paminaw
        except ModuleNotFoundError:
            raise SystemExit("Error: paminaw is not installed. Fix: outside Colab, run "
                             "`pip install -e .` in the repo folder (see docs/OFFLINE.md).") from None

        SETUP = paminaw.setup_workshop(smoke=SMOKE_TEST)
        """,
    ),
]

SECTION_GUARD = """
import os

try:
    import paminaw
except ModuleNotFoundError:
    raise SystemExit("Error: Run cell 0 (Setup) at the top of the notebook first.") from None
"""

# ---------------------------------------------------------------------------
# Section 1 · Load & see
# ---------------------------------------------------------------------------

S1 = [
    md(
        "s1-md",
        """
        ## 1 · Load & see (about 20 minutes)

        A microphone measures air pressure thousands of times per second. Each measurement is a
        **sample**, and the number per second is the **sample rate**. All our models expect
        **16,000 samples per second (16 kHz)**. In PyTorch, a recording is a **tensor** of shape
        `(channels, samples)`.

        A **mel spectrogram** shows which frequencies are loud at each moment, grouped into 80
        bands spaced the way human hearing spaces them. Taking the **log** makes quiet and loud
        parts visible together. Whisper does not read the raw waveform; it reads exactly this
        log-mel picture.
        """,
    ),
    code(
        "s1-setup",
        SECTION_GUARD
        + """
        # Lab 1 setup: safe to run even if you skipped the cells above
        import matplotlib.pyplot as plt
        import torch
        import torchaudio
        from IPython.display import Audio, display

        from paminaw.plotting import PALETTE, plot_mel, plot_waveform, set_style

        SMOKE_TEST = os.environ.get("PAMINAW_SMOKE") == "1"
        paminaw.set_seed()  # 20260814, the seed from the paper
        set_style()  # large fonts for the projector
        N = 2 if SMOKE_TEST else 100
        tl = paminaw.load_subset("tagalog", n=N)
        ceb = paminaw.load_subset("cebuano", n=N)
        print(f"{len(tl)} Tagalog + {len(ceb)} Cebuano clips ready.")
        tl[["fleurs_id", "duration_s", "transcription"]].head(3)
        """.replace("\n        ", "\n"),
    ),
    md(
        "s1-slide-md",
        """
        ### The slide code: from a file to a log-mel spectrogram

        Five lines take us from a file on disk to what the model sees.
        """,
    ),
    code(
        "s1-slide",
        """
        path = tl.path[0]
        print("Tagalog:", tl.transcription[0])

        try:
            wav, sr = torchaudio.load(path)  # file → tensor + sample rate
        except Exception:  # torchaudio.load needs TorchCodec + FFmpeg (Colab has both)
            wav, sr = paminaw.load_audio(path)  # same result, via soundfile

        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono, shape (1, samples)
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=16_000, n_fft=400, hop_length=160, n_mels=80)(wav)
        log_mel = torch.log(mel + 1e-6)

        print(f"sample rate: {sr} Hz · waveform {tuple(wav.shape)} · {wav.shape[1] / 16_000:.1f} s")
        print(f"mel spectrogram {tuple(mel.shape)} = (channels, mel bands, frames)")
        """,
    ),
    md(
        "s1-shapes-md",
        """
        **Reading the shapes.** `n_fft=400` looks at 400 samples (25 ms) at a time and
        `hop_length=160` moves 160 samples (10 ms) each step, so a 10-second clip becomes about
        1,000 frames of 80 numbers each. `+ 1e-6` keeps `log` away from `log(0)` in silent frames.
        """,
    ),
    md(
        "s1-ex1-md",
        """
        ### Exercise 1 · Plot one Tagalog and one Cebuano waveform

        Change `CLIP` to look at other recordings. **Question:** Can you see the pauses between phrases?
        Which recording is longer?
        """,
    ),
    code(
        "s1-ex1",
        """
        CLIP = 0  # try another number (0–99)

        wav_tl = paminaw.load_16k_mono(tl.path[CLIP])
        wav_ceb = paminaw.load_16k_mono(ceb.path[CLIP])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        plot_waveform(wav_tl, title=f"Tagalog: {tl.transcription[CLIP][:55]}…", ax=axes[0])
        plot_waveform(wav_ceb, title=f"Cebuano: {ceb.transcription[CLIP][:55]}…", ax=axes[1],
                      color=PALETTE[1])
        plt.tight_layout()
        plt.show()
        """,
    ),
    code(
        "s1-ex1-sol",
        """
        # Solution: durations, and where the pauses are (frames quieter than 2% of the peak)
        for name, w in [("Tagalog", wav_tl), ("Cebuano", wav_ceb)]:
            frames = w[0, : w.shape[1] // 160 * 160].reshape(-1, 160).abs().max(dim=1).values
            quiet = (frames < 0.02 * frames.max()).float().mean().item()
            print(f"{name}: {w.shape[1] / 16_000:.1f} s long, {quiet:.0%} of 10 ms frames are near-silent")
        """,
        only="solution",
    ),
    md(
        "s1-ex1-ans",
        """
        **Answer.** Pauses show up as flat stretches near zero, mostly between phrases and at the
        start and end of the recording. The length depends on the sentence and on the speaker;
        In our subset the clips run from about 6 to 37 seconds, most of them 10–25 seconds.
        """,
        only="solution",
    ),
    md(
        "s1-ex2-md",
        """
        ### Exercise 2 · Listen at 16 kHz and at 8 kHz

        Put your earphones on. Telephone audio is often 8 kHz: half as many samples, and nothing
        above 4 kHz survives. **Question:** Which sounds change the most?
        """,
    ),
    code(
        "s1-ex2",
        """
        SECONDS = 6  # play only the first few seconds
        clip16 = wav_ceb[:, : 16_000 * SECONDS]
        clip8 = torchaudio.functional.resample(clip16, 16_000, 8_000)
        print(f"16 kHz: {tuple(clip16.shape)} samples → 8 kHz: {tuple(clip8.shape)} samples")

        print("16 kHz (what the models expect):")
        display(Audio(clip16.numpy(), rate=16_000))
        print("8 kHz (telephone quality):")
        display(Audio(clip8.numpy(), rate=8_000))
        """,
    ),
    md(
        "s1-ex2-ans",
        """
        **Answer.** The 8 kHz version sounds muffled, like a phone call. Hissy consonants such as
        *s*, *sh*, *f* and *t* lose most of their energy, because much of it lies above 4 kHz.
        """,
        only="solution",
    ),
    md(
        "s1-ex3-md",
        """
        ### Exercise 3 · Compare their mel spectrograms

        We resample the 8 kHz clip back to 16 kHz, so both pictures have the same axes.
        **Question:** What happened to the top of the picture?
        """,
    ),
    code(
        "s1-ex3",
        """
        back16 = torchaudio.functional.resample(clip8, 8_000, 16_000)  # same length, less detail

        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        plot_mel(paminaw.log_mel(paminaw.mel_spectrogram(clip16)), title="Recorded at 16 kHz", ax=axes[0])
        plot_mel(paminaw.log_mel(paminaw.mel_spectrogram(back16)), title="Through 8 kHz", ax=axes[1])
        plt.tight_layout()
        plt.show()
        """,
    ),
    code(
        "s1-ex3-sol",
        """
        # Solution: how much energy is left above 4 kHz? (mel bands ≥ 60 of 80 ≈ above 4 kHz)
        hi16 = paminaw.mel_spectrogram(clip16)[0, 60:].sum() / paminaw.mel_spectrogram(clip16)[0].sum()
        hi8 = paminaw.mel_spectrogram(back16)[0, 60:].sum() / paminaw.mel_spectrogram(back16)[0].sum()
        print(f"Share of energy above ~4 kHz: 16 kHz {hi16:.2%} · through 8 kHz {hi8:.4%}")

        # Bonus: Tagalog vs Cebuano side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        plot_mel(paminaw.log_mel(paminaw.mel_spectrogram(wav_tl)), title="Tagalog", ax=axes[0])
        plot_mel(paminaw.log_mel(paminaw.mel_spectrogram(wav_ceb)), title="Cebuano", ax=axes[1])
        plt.tight_layout()
        plt.show()
        """,
        only="solution",
    ),
    md(
        "s1-ex3-ans",
        """
        **Answer.** Everything above about 4 kHz (mel band ≈ 60 of 80) goes dark. A sample rate
        of 8 kHz can only represent frequencies up to half that, 4 kHz (the *Nyquist* limit). A
        model trained on 16 kHz audio, like Whisper, has never seen pictures with that hole, which
        is one reason ASR does worse on phone recordings. Tagalog and Cebuano look alike as
        pictures: same speakers' voices, same kinds of sounds. The difference is in the words,
        which is why the language a model was trained on matters so much in Lab 2.
        """,
        only="solution",
    ),
]

# ---------------------------------------------------------------------------
# Section 2 · Transcribe
# ---------------------------------------------------------------------------

S2 = [
    md(
        "s2-md",
        """
        ## 2 · Transcribe (about 30 minutes)

        A speech recogniser (ASR) works in stages:
        **audio** → 16 kHz tensor → **log-mel features** → **encoder** (turns the picture into a
        sequence of vectors) → **decoder** (writes text one token at a time) → **text**.

        **Whisper** (OpenAI) is an encoder-decoder trained on 680,000 hours of audio from the web.
        Its vocabulary has a language token for Tagalog, `<|tl|>`, but **none for Cebuano**.

        **Why PyTorch rather than TensorFlow?** Both can run speech models. But the open speech
        models we use (Whisper, MMS and SeamlessM4T) are published as PyTorch weights. Hugging
        Face `transformers` v5 supports PyTorch only. And `torchaudio` gives us audio loading and
        features in the same tensor library.
        """,
    ),
    code(
        "s2-setup",
        SECTION_GUARD
        + """
        # Lab 2 setup: safe to run even if you skipped the cells above
        import time
        from pathlib import Path

        import pandas as pd
        import torch
        import transformers
        from transformers import pipeline

        from paminaw.asr import pick_device_dtype

        transformers.logging.set_verbosity_error()  # hide library warnings; real errors still show
        SMOKE_TEST = os.environ.get("PAMINAW_SMOKE") == "1"
        paminaw.set_seed()
        device, dtype = pick_device_dtype()  # ("cuda", float16) on a GPU, ("cpu", float32) otherwise
        WHISPER_MODEL = "openai/whisper-tiny" if SMOKE_TEST else "openai/whisper-small"

        # The one number to change: clips per language. 20 on CPU (about 10 minutes), 100 on a GPU.
        N_CLIPS = 2 if SMOKE_TEST else (100 if device == "cuda" else 20)

        tl = paminaw.load_subset("tagalog", n=N_CLIPS)
        ceb = paminaw.load_subset("cebuano", n=N_CLIPS)
        print(f"{WHISPER_MODEL} on {device} ({str(dtype).replace('torch.', '')}), "
              f"{N_CLIPS} clips per language")
        """.replace("\n        ", "\n"),
    ),
    code(
        "s2-load",
        """
        # Download Whisper (about 1 GB for whisper-small; Colab's network is fast) and build a pipeline.
        # chunk_length_s=30: Whisper hears at most 30 s at a time; 6 of our clips are a little longer.
        asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=device, dtype=dtype,
                       chunk_length_s=30)
        """,
    ),
    code(
        "s2-one",
        """
        wav = paminaw.load_16k_mono(tl.path[0])

        with torch.inference_mode():  # we only predict, so skip gradient bookkeeping
            out = asr({"raw": wav.squeeze(0).numpy(), "sampling_rate": 16_000},
                      generate_kwargs={"language": "tagalog", "task": "transcribe"})

        print("REFERENCE :", tl.transcription[0])
        print("WHISPER   :", out["text"])
        """,
    ),
    md(
        "s2-all-md",
        """
        ### Transcribe all clips

        We pass `language="tagalog"` for the Cebuano clips too. Whisper has no Cebuano option, so
        Tagalog is the nearest language it is allowed to write; the paper calls this the
        *language-committed* condition. Results are saved to `outputs/` for Lab 3.
        """,
    ),
    code(
        "s2-all",
        """
        rows, t0, audio_seconds = [], time.perf_counter(), 0.0
        for lang, clips in [("tagalog", tl), ("cebuano", ceb)]:
            for c in clips.itertuples():
                wav = paminaw.load_16k_mono(c.path)
                with torch.inference_mode():
                    text = asr({"raw": wav.squeeze(0).numpy(), "sampling_rate": 16_000},
                               generate_kwargs={"language": "tagalog", "task": "transcribe"})["text"]
                rows.append({"lang": lang, "idx": c.idx, "fleurs_id": c.fleurs_id,
                             "hypothesis": text, "reference": c.transcription})
                audio_seconds += c.duration_s
                if len(rows) % 10 == 0 or len(rows) == 2 * N_CLIPS:
                    print(f"{len(rows)} of {2 * N_CLIPS} clips done ({time.perf_counter() - t0:.0f} s)")
        seconds = time.perf_counter() - t0
        print(f"Transcribed {audio_seconds / 60:.1f} min of audio in {seconds:.0f} s "
              f"(real-time factor {seconds / audio_seconds:.2f}).")

        mine = pd.DataFrame(rows)
        Path("outputs").mkdir(exist_ok=True)
        mine.to_csv("outputs/whisper_small_yours.csv", index=False)
        pd.set_option("display.max_colwidth", 90)
        mine[["lang", "reference", "hypothesis"]].groupby("lang").head(3)
        """,
    ),
    md(
        "s2-ex4-md",
        """
        ### Exercise 4 · Which language does Whisper hear?

        Remove the language hint on a Cebuano clip and let Whisper decide. **Before you run it:**
        what do you expect it to detect? Then read the **language token** Whisper chose.
        """,
    ),
    code(
        "s2-ex4",
        """
        samples = paminaw.load_16k_mono(ceb.path[0]).squeeze(0).numpy()

        with torch.inference_mode():  # note: no "language" key this time
            no_hint = asr({"raw": samples, "sampling_rate": 16_000}, generate_kwargs={"task": "transcribe"})
        print("REFERENCE (Cebuano):", ceb.transcription[0])
        print("NO HINT            :", no_hint["text"])

        # Ask Whisper directly which language token it would start with
        features = asr.feature_extractor(samples, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            lang_ids = asr.model.detect_language(features.input_features.to(device, dtype))
        print("Detected language token:", asr.tokenizer.convert_ids_to_tokens(lang_ids.tolist()))
        print("Does Whisper even have a Cebuano token?", "<|ceb|>" in asr.tokenizer.get_vocab())
        """,
    ),
    code(
        "s2-ex4-sol",
        """
        # Solution: check all Cebuano clips, and compare with Whisper large-v3 from the paper's runs
        detected = []
        for p in ceb.path:
            w = paminaw.load_16k_mono(p).squeeze(0).numpy()
            f = asr.feature_extractor(w, sampling_rate=16_000, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = asr.model.detect_language(f.to(device, dtype))
            detected += asr.tokenizer.convert_ids_to_tokens(ids.tolist())
        print(f"{WHISPER_MODEL}, {len(detected)} Cebuano clips:", pd.Series(detected).value_counts().to_dict())

        large = paminaw.load_precomputed("Whisper large-v3")
        print("Whisper large-v3 (paper run), 100 Cebuano clips:",
              large[large.lang == "cebuano"]["detected_language"].value_counts().to_dict())
        """,
        only="solution",
    ),
    md(
        "s2-ex4-ans",
        """
        **Answer.** Whisper detects **Tagalog** (`<|tl|>`). It has no Cebuano token at all, so the
        nearest language it is allowed to write is the related Central Philippine language it
        knows. In the paper, Whisper large-v3 chose Tagalog for **every one** of the 541 Cebuano
        test clips, and the "with hint" and "no hint" runs came out byte-identical. Its Cebuano
        errors are largely Tagalog words: Cebuano *nga* written as Tagalog *na*/*ng*, *tawo* as
        *tao*, *dili* as *hindi*. Smaller Whisper models sometimes pick another language on a
        clip, but never Cebuano.
        """,
        only="solution",
    ),
]

S2B = [
    md(
        "s2b-md",
        """
        ## 2b · MMS-1B-all: a model that *does* know Cebuano (GPU runtimes)

        **MMS** (Massively Multilingual Speech, Meta) is one shared wav2vec 2.0 model plus a small
        **adapter** for each of 1,162 languages, including Cebuano (`ceb`) and Tagalog (`tgl`).
        It is a **CTC** model: instead of writing word by word, it scores every letter (or a
        "blank") for each 20 ms frame. We take the most likely one per frame (**argmax**), then
        merge repeats and drop blanks.

        ### Three choices every time you run a model

        | Choice | What we use | Why |
        |---|---|---|
        | **Device** | `cuda` if there is a GPU, else `cpu` | A GPU does thousands of multiplications at once. MMS has a billion parameters, so it only runs live on a GPU here. |
        | **Precision** | `float16` for Whisper on a GPU; `float32` for MMS and anything on a CPU | Half precision halves memory and speeds up GPUs, but CPUs do not benefit. The MMS adapters are float32 weights, so the paper runs MMS in float32. |
        | **`torch.inference_mode()`** | always, when transcribing | Training has to remember every step to compute gradients; inference does not. Switching that off saves memory and time. |

        On a CPU runtime this lab loads the paper's precomputed MMS outputs for the same clips
        instead.
        """,
        tags=("lab2b",),
    ),
    code(
        "s2b-run",
        SECTION_GUARD
        + """
        import pandas as pd
        import torch
        import transformers
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        transformers.logging.set_verbosity_error()  # hide library warnings; real errors still show
        SMOKE_TEST = os.environ.get("PAMINAW_SMOKE") == "1"
        paminaw.set_seed()
        RUN_MMS = torch.cuda.is_available() and not SMOKE_TEST
        N_MMS = 100 if RUN_MMS else 20
        clips = {lang: paminaw.load_subset(lang, n=N_MMS) for lang in ("cebuano", "tagalog")}

        if RUN_MMS:
            processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", dtype=torch.float32).to("cuda")
            rows = []
            for lang, code in [("cebuano", "ceb"), ("tagalog", "tgl")]:
                processor.tokenizer.set_target_lang(code)  # the adapter's alphabet
                mms.load_adapter(code)  # swap in the language adapter (~2M parameters)
                mms.to("cuda")
                for c in clips[lang].itertuples():
                    wav = paminaw.load_16k_mono(c.path).squeeze(0).numpy()
                    inputs = processor(wav, sampling_rate=16_000, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        logits = mms(**inputs).logits
                    ids = torch.argmax(logits, dim=-1)[0]  # best letter per 20 ms frame
                    rows.append({"lang": lang, "idx": c.idx, "hypothesis": processor.decode(ids)})
            mms_out = pd.DataFrame(rows)
            print(f"MMS ran live on {torch.cuda.get_device_name(0)} for {len(mms_out)} clips.")
        else:
            print("Note: no GPU (or smoke test): loading the paper's precomputed MMS-1B-all outputs "
                  "for the same clips instead.")
            pre = paminaw.load_precomputed("MMS-1B-all")
            keys = pd.concat(clips.values())[["lang", "idx"]]
            mms_out = pre.merge(keys, on=["lang", "idx"])[["lang", "idx", "hypothesis"]]

        ref = pd.concat(clips.values())[["lang", "idx", "transcription"]]
        show = ref.merge(mms_out, on=["lang", "idx"]).rename(columns={"transcription": "reference"})
        pd.set_option("display.max_colwidth", 90)
        show[show.lang == "cebuano"].head(3)[["reference", "hypothesis"]]
        """.replace("\n        ", "\n"),
        tags=("lab2b",),
    ),
]

# ---------------------------------------------------------------------------
# Section 3 · Score
# ---------------------------------------------------------------------------

S3 = [
    md(
        "s3-md",
        """
        ## 3 · Score (about 20 minutes)

        **WER** (word error rate) counts the **s**ubstitutions, **d**eletions and **i**nsertions
        needed to turn the model's words into the reference words, divided by the number of
        reference words: WER = (S + D + I) / N. It can go above 100% when a model adds many extra
        words. **CER** does the same with characters.

        Before comparing, texts are usually **normalised**: lowercased, punctuation removed, and
        so on. *How* you normalise is a choice. The paper behind this workshop (Golosino, Morales
        & Adlaon 2026) shows that this choice alone moves Whisper's Tagalog WER by 15 points and
        can change which model ranks first.
        """,
    ),
    code(
        "s3-setup",
        SECTION_GUARD
        + """
        # Lab 3 setup: safe to run even if you skipped Labs 1 and 2
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd

        from paminaw.compare import (PAPER_TABLE3_WER, convention_table, ranking_table,
                                     score_systems, top_substitutions)
        from paminaw.plotting import plot_convention_bars, set_style

        SMOKE_TEST = os.environ.get("PAMINAW_SMOKE") == "1"
        paminaw.set_seed()
        set_style()
        pd.set_option("display.max_colwidth", 90)
        refs = paminaw.load_manifest()  # the 200 references (FLEURS "transcription" field)

        MINE = Path("outputs/whisper_small_yours.csv")
        if MINE.exists():
            mine = pd.read_csv(MINE, keep_default_na=False)
            source = "your Lab 2 run"
        else:  # skipped Lab 2: use a ready-made Whisper-small run on the same clips
            n = 2 if SMOKE_TEST else 20
            mine = pd.read_csv(paminaw.data.data_file("precomputed/whisper-small.csv"),
                               keep_default_na=False).rename(columns={"hypothesis_raw": "hypothesis"})
            mine = mine.groupby("lang").head(n)
            source = "a ready-made Whisper-small run (you skipped Lab 2)"
        n_mine = mine.groupby("lang").size().min()
        print(f"Whisper-small outputs: {len(mine)} clips from {source}.")
        """.replace("\n        ", "\n"),
    ),
    md(
        "s3-ex5-md",
        """
        ### Exercise 5 · Write `normalize()` yourself

        The paper's **primary** convention has four steps:
        1. Unicode **NFC** (so `ñ` is always one character, never `n` + a separate tilde)
        2. **case-fold** (lowercase, done properly for every language)
        3. replace every character that is not a letter, digit, `_` or space **with a space**
        4. **collapse** runs of spaces into one, and trim the ends

        Steps 1 and 2 are done. Finish steps 3 and 4 (hint: `re.sub`), then run the check below.
        """,
    ),
    code(
        "s3-ex5",
        """
        import re
        import unicodedata


        def normalize(text):
            text = unicodedata.normalize("NFC", text)  # 1. compose characters
            text = text.casefold()  # 2. lowercase
            # 3. TODO: text = re.sub(r"[^\\w\\s]", " ", text)
            # 4. TODO: text = re.sub(r"\\s+", " ", text).strip()
            return text
        """,
        only="exercise",
    ),
    code(
        "s3-ex5",
        """
        import re
        import unicodedata


        def normalize(text):
            text = unicodedata.normalize("NFC", text)  # 1. compose characters
            text = text.casefold()  # 2. lowercase
            text = re.sub(r"[^\\w\\s]", " ", text)  # 3. punctuation → space
            text = re.sub(r"\\s+", " ", text).strip()  # 4. collapse spaces
            return text
        """,
        only="solution",
    ),
    code(
        "s3-ex5-check",
        """
        examples = ["Maayong buntag, Doc!", "Kumusta  ka na?", "Sen\\u0303or N\\u0303ino",
                    "ang populasyon kay 800. kini", "bag-ohon"]
        for ex in examples:
            ok = normalize(ex) == paminaw.normalize(ex, "primary")
            hint = "" if ok else f"   (expected {paminaw.normalize(ex)!r})"
            print("PASS" if ok else "FAIL", f"{ex!r:34} → {normalize(ex)!r}{hint}")
        print("\\nFrom here on we use paminaw.normalize, which also has the other three conventions.")
        """,
    ),
    md(
        "s3-worked-md",
        """
        ### Two worked examples

        **1 · `kumusta ka na` → `kumusta kana`.** The model merged two words. That is one
        substitution (*ka* → *kana*) plus one deletion (*na*) over 3 reference words:
        WER = **2/3**. What about CER? It depends on a definition most people never look at:

        * `jiwer.cer`, the library default, **counts spaces as characters**: 13 reference
          characters and one missing space, so CER = **1/13 ≈ 7.7%**.
        * The paper computes CER over **non-whitespace characters only**: `kumustakana` against
          `kumustakana` is 0 errors out of 11, so CER = **0/11 = 0%**.

        Same strings, two "correct" CERs. Just like normalisation, the definition has to be
        stated next to the number.

        **2 · `Maayong buntag, Doc!` → `maayong buntag doc`.** Every word differs if you compare
        exactly as written, so WER = **100%** under `none`. After the primary normalisation the
        texts are identical, so WER = **0%**.
        """,
    ),
    code(
        "s3-worked",
        """
        ref, hyp = "kumusta ka na", "kumusta kana"
        print(f"WER                          : {paminaw.wer(ref, hyp):.4f}  (2/3 = {2 / 3:.4f})")
        print(f"CER, jiwer default (spaces)  : {paminaw.cer_jiwer_default(ref, hyp):.4f}  (1/13 = {1 / 13:.4f})")
        print(f"CER, paper (no whitespace)   : {paminaw.cer(ref, hyp):.4f}  (0/11)")

        ref, hyp = "Maayong buntag, Doc!", "maayong buntag doc"
        pd.DataFrame(
            [{"convention": c, "reference": paminaw.normalize(ref, c), "hypothesis": paminaw.normalize(hyp, c),
              "WER %": 100 * paminaw.wer(paminaw.normalize(ref, c), paminaw.normalize(hyp, c))}
             for c in paminaw.CONVENTIONS]
        ).round(1)
        """,
    ),
    md(
        "s3-mine-md",
        """
        ### Your Whisper-small: raw vs normalised

        The FLEURS references are lowercase with almost no punctuation. Whisper writes capitals
        and punctuation. Watch what that does to the score.
        """,
    ),
    code(
        "s3-mine",
        """
        mine_sys = mine.assign(system="Whisper-small (yours)")[["system", "lang", "idx", "hypothesis"]]
        label = f"workshop subset ({n_mine} clips per language)"
        s_mine = score_systems(refs, mine_sys, conventions=["none", "primary"], subset_label=label)
        print(f"Scores on the {label}; not the paper's full-split numbers.")
        s_mine.pivot_table(index="lang", columns="convention", values=["wer", "cer"]).round(1)
        """,
    ),
    md(
        "s3-four-md",
        """
        ### The paper's three systems, four conventions

        These are the paper's own, unnormalised outputs from Whisper large-v3, MMS-1B-all and
        SeamlessM4T v2 Large for the **same 200 clips**. Nothing is re-run: only the scoring
        changes.
        """,
    ),
    code(
        "s3-four",
        """
        pre = paminaw.load_precomputed()
        scores = score_systems(refs, pre, subset_label="workshop subset (100 clips)")
        print("WER (%) on the workshop subset (100 clips per language):")
        convention_table(scores, "wer")
        """,
    ),
    code(
        "s3-plots",
        """
        fig, axes = plt.subplots(2, 1, figsize=(13, 10))
        for ax, lang in zip(axes, ["tagalog", "cebuano"]):
            plot_convention_bars(scores, lang, ax=ax)
        plt.tight_layout()
        plt.show()
        """,
    ),
    code(
        "s3-paper",
        """
        paper = pd.DataFrame(PAPER_TABLE3_WER).T
        paper.index.names = ["system", "lang"]
        print("For comparison, the paper's Table 3: WER (%) on the FULL test splits "
              "(964 Tagalog, 541 Cebuano utterances):")
        paper
        """,
    ),
    md(
        "s3-ex6-md",
        """
        ### Exercise 6 · Does the ranking change?

        Rank 1 = lowest WER. Look along each language. **Question:** Under which convention would you pick a
        different "best" model for Tagalog? Is any result on Cebuano robust to every convention?
        """,
    ),
    code(
        "s3-ex6",
        """
        ranking_table(scores, "wer")
        """,
    ),
    code(
        "s3-ex6-sol",
        """
        # Solution: print the order per convention, and flag changes
        for lang in ["tagalog", "cebuano"]:
            orders = {}
            for conv in paminaw.CONVENTIONS:
                s = scores[(scores.lang == lang) & (scores.convention == conv)].sort_values("wer")
                orders[conv] = " < ".join(f"{r.system.split()[0]} {r.wer:.1f}" for r in s.itertuples())
            changed = len({o.split(" ")[0] for o in orders.values()}) > 1
            print(f"{lang.upper()}: winner {'CHANGES' if changed else 'stays the same'} across conventions")
            for conv, o in orders.items():
                print(f"   {conv:9} {o}")
        """,
        only="solution",
    ),
    md(
        "s3-ex6-ans",
        """
        **Answer.** On the full test split (paper, Sect. 4.2), Whisper large-v3 goes from **last**
        on Tagalog under `none` (27.11) to **first** under `primary` (12.23, just ahead of MMS at
        12.50). The scorer alone decides which of the two prints first. The two are statistically
        indistinguishable on Tagalog. MMS barely moves, because its output has no capitals or
        punctuation to normalise away.

        On Cebuano, MMS beats Whisper under **every** convention by a margin no scoring choice
        comes close to. That difference is real; the Tagalog one is not.

        Your subset numbers will differ a little from the full split; 100 clips is a small
        sample. Check whether the same pattern shows.
        """,
        only="solution",
    ),
]

WRAP = [
    md(
        "wrap-md",
        """
        ## Wrap-up: where the models fall short

        * **Language coverage beats size.** Whisper large-v3 (1.55 B parameters) has no Cebuano
          token and degrades from 12 % to 43 % WER going from Tagalog to Cebuano. MMS (1 B
          parameters) declares Cebuano and barely moves (12.5 % → 13.6 %).
        * **It writes the nearest language it knows.** Below are Whisper large-v3's most frequent
          Cebuano → Tagalog substitutions on our subset.
        * **Read speech is the easy case.** FLEURS sentences are read aloud from prepared text, so
          these numbers are an upper bound. Spontaneous, code-switched Cebuano–Tagalog–English
          speech is much harder, and there is not yet a public benchmark for it.
        * **Scoring is part of the result.** Always report the normalisation convention and
          which CER definition you used.

        **What fine-tuning involves.** Collect transcribed audio in the target language (hours,
        not thousands of hours), then train only a small part of the model. For MMS that is a new
        **adapter** of about 2 M parameters, while the billion shared weights stay frozen. Hugging
        Face has a step-by-step guide: [Fine-tuning MMS adapter models]({MMS_BLOG}).

        **Links**
        * Paper data, hypotheses and scoring code: [Zenodo {ZENODO_SHORT}]({ZENODO})
        * Workshop repository: <https://github.com/{REPO}> (questions → Issues)

        *Salamat kaayo! Maraming salamat!*
        """.replace("{MMS_BLOG}", MMS_BLOG)
        .replace("{ZENODO}", ZENODO)
        .replace("{ZENODO_SHORT}", "10.5281/zenodo.23010849")
        .replace("{REPO}", REPO),
    ),
    code(
        "wrap-subs",
        SECTION_GUARD
        + """
        import pandas as pd
        from paminaw.compare import top_substitutions

        large = paminaw.load_precomputed("Whisper large-v3")
        ceb_refs = paminaw.load_manifest("cebuano")
        both = ceb_refs.merge(large[large.lang == "cebuano"], on=["lang", "idx"])
        print("Whisper large-v3 on 100 Cebuano clips: most frequent word substitutions "
              "(reference → output)")
        top_substitutions(both["transcription"], both["hypothesis"], n=10)
        """.replace("\n        ", "\n"),
    ),
]


def next_lab(target: str, label: str) -> Cell:
    """A closing cell that links to the next standalone lab."""
    return md(
        "next",
        f"""
        **Next:** [{label}]({target}) · or open the
        [full workshop notebook](paminaw_workshop.ipynb)
        """,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def notebooks(solutions: bool) -> dict[str, list[Cell]]:
    sub = "solutions/" if solutions else ""
    return {
        "paminaw_workshop.ipynb": title_main(sub + "paminaw_workshop.ipynb", solutions)
        + SETUP + S1 + S2 + S2B + S3 + WRAP,
        "01_load_and_see.ipynb": title_section(
            sub + "01_load_and_see.ipynb", "Lab 1 · Load & see",
            "Load recordings as tensors, resample, and compute mel spectrograms.", solutions)
        + SETUP + S1 + [next_lab("02_transcribe.ipynb", "Lab 2 · Transcribe")],
        "02_transcribe.ipynb": title_section(
            sub + "02_transcribe.ipynb", "Lab 2 · Transcribe",
            "Run Whisper (and MMS on a GPU) on Tagalog and Cebuano.", solutions)
        + SETUP + S2 + S2B + [next_lab("03_score.ipynb", "Lab 3 · Score")],
        "03_score.ipynb": title_section(
            sub + "03_score.ipynb", "Lab 3 · Score",
            "WER, CER and the four scoring conventions of the paper.", solutions)
        + SETUP + S3 + WRAP,
    }  # fmt: skip


def build(cells: list[Cell], solutions: bool):
    want = "solution" if solutions else "exercise"
    nb = new_notebook()
    nb.cells = [c.build() for c in cells if c.only in (None, want)]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    return nb


def section_of(cell_id: str) -> str:
    return cell_id.split("-")[0]


def execute(path: Path, timeout: int) -> dict[str, float]:
    """Run a notebook in place; return seconds spent per section."""
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", record_timing=True,
                            resources={"metadata": {"path": str(path.parent)}})  # fmt: skip
    client.execute()
    per_section: dict[str, float] = {}
    for cell in nb.cells:
        ex = cell.metadata.pop("execution", None)
        if cell.cell_type != "code" or not ex:
            continue
        start = datetime.fromisoformat(ex["iopub.execute_input"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(ex["shell.execute_reply"].replace("Z", "+00:00"))
        sec = section_of(cell.id)
        per_section[sec] = per_section.get(sec, 0.0) + (end - start).total_seconds()
    nbformat.write(nb, path)
    return per_section


def carry_outputs(nb, old_path: Path) -> int:
    """Copy saved outputs from the notebook at ``old_path`` into ``nb``, matching code cells by id."""
    if not old_path.is_file():
        return 0
    old_nb = nbformat.read(old_path, as_version=4)
    for key in ("widgets", "language_info"):  # saved widget views and kernel info belong to outputs
        if key in old_nb.metadata:
            nb.metadata[key] = old_nb.metadata[key]
    old = {c.id: c for c in old_nb.cells if c.cell_type == "code"}
    n = 0
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.id in old:
            cell.outputs = old[cell.id].outputs
            cell.execution_count = old[cell.id].execution_count
            n += 1
    return n


def graft(path: Path, donor: Path) -> int:
    """Copy outputs of cells tagged ``lab2b`` from ``donor`` into ``path``, matching by cell id."""
    nb, src = nbformat.read(path, as_version=4), nbformat.read(donor, as_version=4)
    by_id = {c.id: c for c in src.cells}
    n = 0
    for cell in nb.cells:
        tagged = "lab2b" in cell.metadata.get("tags", []) and cell.cell_type == "code"
        if tagged and cell.id in by_id and by_id[cell.id].get("outputs"):
            cell.outputs = by_id[cell.id].outputs
            cell.execution_count = by_id[cell.id].execution_count
            n += 1
    nbformat.write(nb, path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run each notebook and save outputs")
    ap.add_argument("--smoke", action="store_true", help="2 clips, whisper-tiny, MMS skipped")
    ap.add_argument("--only", nargs="*", default=None, help="notebook file names to build")
    ap.add_argument("--graft-2b", type=Path, default=None, help="executed notebook from a GPU run")
    ap.add_argument(
        "--keep-outputs", action="store_true", help="rebuild sources but keep saved outputs"
    )
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--timings", type=Path, default=None, help="write per-section seconds as JSON")
    args = ap.parse_args()

    if args.smoke:
        os.environ["PAMINAW_SMOKE"] = "1"
    else:
        os.environ.pop("PAMINAW_SMOKE", None)
    timings = {}
    for solutions in (False, True):
        for name, cells in notebooks(solutions).items():
            if args.only and name not in args.only:
                continue
            path = NB_DIR / ("solutions" if solutions else "") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            nb = build(cells, solutions)
            kept = carry_outputs(nb, path) if args.keep_outputs else 0
            nbformat.write(nb, path)
            rel = path.relative_to(ROOT).as_posix()
            if args.execute:
                t0 = time.perf_counter()
                per = execute(path, args.timeout)
                timings[rel] = {k: round(v, 1) for k, v in per.items()}
                print(f"ran {rel} in {time.perf_counter() - t0:.0f} s · per section {timings[rel]}")
            else:
                print(f"wrote {rel}" + (f" (kept outputs of {kept} cells)" if kept else ""))
            if args.graft_2b and name in ("paminaw_workshop.ipynb", "02_transcribe.ipynb"):
                print(f"   grafted {graft(path, args.graft_2b)} Lab 2b cell(s)")
    if args.timings:
        args.timings.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
