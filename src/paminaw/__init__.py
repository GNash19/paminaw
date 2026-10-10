"""Paminaw: teaching machines to listen in Cebuano and Tagalog.

Companion package for the PyCon Davao 2026 workshop *Speech Processing Using
PyTorch*. The short public API::

    import paminaw
    clips = paminaw.load_subset("cebuano", n=20)        # FLEURS clips + references
    wav = paminaw.load_16k_mono(clips.path[0])          # (1, samples) tensor at 16 kHz
    paminaw.normalize("Maayong buntag, Doc!")           # 'maayong buntag doc'
    paminaw.wer(["kumusta ka na"], ["kumusta kana"])    # 0.666...
"""

__version__ = "1.0.0"

from .audio import load_16k_mono, load_audio, log_mel, mel_spectrogram, resample_16k, to_mono
from .data import SEED, load_manifest, load_precomputed, load_subset
from .env import set_seed, setup_workshop
from .metrics import cer, cer_jiwer_default, mer, wer
from .normalize import CONVENTIONS, normalize

__all__ = [
    "CONVENTIONS",
    "SEED",
    "__version__",
    "cer",
    "cer_jiwer_default",
    "load_16k_mono",
    "load_audio",
    "load_manifest",
    "load_precomputed",
    "load_subset",
    "log_mel",
    "mel_spectrogram",
    "mer",
    "normalize",
    "resample_16k",
    "set_seed",
    "setup_workshop",
    "to_mono",
    "wer",
]
