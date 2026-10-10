"""Pretrained speech recognisers: Whisper (encoder-decoder) and MMS (CTC + adapters).

Three choices matter when running a model (the paper's Sect. 3.3 settings in brackets):

* **device**: a GPU (``"cuda"``) is much faster; Colab's free CPU works for small models.
* **precision**: ``float16`` halves memory and speeds up a GPU [Whisper, SeamlessM4T];
  CPUs need ``float32``, and so does MMS, whose adapters load as float32 weights [MMS].
* **inference mode**: ``torch.inference_mode()`` turns off gradient bookkeeping,
  which we never need when only transcribing.

Offline, set ``HF_HOME`` to the ``hf_home`` folder made by ``paminaw prepare-offline``
and ``HF_HUB_OFFLINE=1``; models then load from that cache with no code changes.

Newer ``transformers`` (4.56+) call the precision argument ``dtype=``; older
releases call it ``torch_dtype=``. :func:`dtype_kwargs` picks the right one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio import TARGET_SR, load_16k_mono
from .data import LANGUAGES, resolve_lang

__all__ = [
    "MMSTranscriber",
    "WhisperTranscriber",
    "dtype_kwargs",
    "pick_device_dtype",
    "to_array",
]


def _hf() -> Any:
    """The ``transformers`` module, imported on first use (and easy to replace in tests)."""
    import transformers

    return transformers


def pick_device_dtype(prefer_gpu: bool = True) -> tuple[str, torch.dtype]:
    """``("cuda", torch.float16)`` when a GPU is available, else ``("cpu", torch.float32)``."""
    if prefer_gpu and torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _transformers_version() -> tuple[int, int]:
    major, minor = _hf().__version__.split(".")[:2]
    return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)


def dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """``{"dtype": dtype}`` on transformers ≥ 4.56, ``{"torch_dtype": dtype}`` on older releases."""
    return {"dtype": dtype} if _transformers_version() >= (4, 56) else {"torch_dtype": dtype}


def to_array(audio: str | Path | torch.Tensor | np.ndarray) -> np.ndarray:
    """Turn a file path, a (1, samples) tensor or an array into a 1-D float32 NumPy array at 16 kHz.

    Tensors and arrays are assumed to be 16 kHz already.
    """
    if isinstance(audio, (str, Path)):
        audio = load_16k_mono(audio)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().to("cpu", torch.float32).numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected mono audio, got shape {arr.shape}. "
            "Fix: pass a (1, samples) tensor or a file path."
        )
    return arr


class WhisperTranscriber:
    """Whisper through the ``transformers`` ASR pipeline.

    Whisper has a Tagalog language token but **no Cebuano token**, so Cebuano
    audio is transcribed with the nearest declared language, Tagalog (paper Sect. 4.3).
    """

    def __init__(
        self,
        model: str = "openai/whisper-small",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        language: str | None = "tagalog",
        revision: str | None = None,
    ) -> None:
        auto_device, auto_dtype = pick_device_dtype()
        self.device = device or auto_device
        self.dtype = dtype or (auto_dtype if self.device == auto_device else torch.float32)
        if self.device == "cpu" and self.dtype == torch.float16:
            raise ValueError(
                "float16 is not supported for Whisper on CPU. Fix: use dtype=torch.float32 on CPU."
            )
        self.language = language
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.pipe = _hf().pipeline(
            "automatic-speech-recognition",
            model=model,
            device=self.device,
            chunk_length_s=30,  # Whisper reads 30 s windows; some FLEURS clips run 31–37 s
            **dtype_kwargs(self.dtype),
            **kwargs,
        )

    def transcribe(
        self,
        audio: str | Path | torch.Tensor | np.ndarray,
        language: str | None = "default",
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe one clip. ``language=None`` lets Whisper detect the language itself."""
        lang = self.language if language == "default" else language
        gen: dict[str, Any] = {"task": "transcribe", "max_new_tokens": max_new_tokens}
        if lang is not None:
            gen["language"] = lang
        inputs = {"raw": to_array(audio), "sampling_rate": TARGET_SR}
        with torch.inference_mode():
            out = self.pipe(inputs, generate_kwargs=gen)
        return out["text"]

    def detect_language(self, audio: str | Path | torch.Tensor | np.ndarray) -> str:
        """Return the language token Whisper picks for this clip, e.g. ``"<|tl|>"``."""
        feats = self.pipe.feature_extractor(
            to_array(audio), sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device, self.dtype)
        with torch.inference_mode():
            lang_ids = self.pipe.model.detect_language(feats)
        return self.pipe.tokenizer.convert_ids_to_tokens(int(lang_ids.reshape(-1)[0]))


class MMSTranscriber:
    """MMS-1B-all: one shared CTC model plus a small adapter per language.

    ``set_language("ceb")`` swaps in the Cebuano adapter; ``"tgl"`` is Tagalog.
    Decoding is greedy: the most likely token per frame (argmax), then the
    tokenizer merges repeats and removes CTC blanks.
    """

    def __init__(
        self,
        model: str = "facebook/mms-1b-all",
        lang: str = "ceb",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        hf = _hf()
        self.device = device or pick_device_dtype()[0]
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.processor = hf.AutoProcessor.from_pretrained(model, **kwargs)
        self.model = hf.Wav2Vec2ForCTC.from_pretrained(
            model, **dtype_kwargs(torch.float32), **kwargs
        ).eval()
        self.lang: str | None = None
        self.set_language(lang)

    def set_language(self, lang: str) -> None:
        """Load the adapter for ``lang`` ("ceb"/"cebuano" or "tgl"/"tagalog")."""
        code = lang if lang in ("ceb", "tgl") else LANGUAGES[resolve_lang(lang)]["mms"]
        self.processor.tokenizer.set_target_lang(code)
        self.model.load_adapter(code)
        self.model.to(self.device)  # load_adapter can leave new weights on the CPU
        self.lang = code

    def transcribe(self, audio: str | Path | torch.Tensor | np.ndarray) -> str:
        """Transcribe one clip with the current adapter."""
        inputs = self.processor(to_array(audio), sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        return self.processor.decode(ids)
