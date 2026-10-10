"""Audio as tensors: load, mix to mono, resample to 16 kHz, mel spectrograms.

These helpers mirror the workshop slide code one step per function::

    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=400, hop_length=160, n_mels=80)(wav)
    log_mel = torch.log(mel + 1e-6)

Since torchaudio 2.9, ``torchaudio.load`` decodes through TorchCodec, which
needs FFmpeg. Colab has both. On laptops without them, :func:`load_audio`
falls back to ``soundfile`` and converts with ``torch.from_numpy``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

__all__ = [
    "TARGET_SR",
    "AudioLoadError",
    "load_16k_mono",
    "load_audio",
    "log_mel",
    "mel_spectrogram",
    "resample_16k",
    "to_mono",
]

#: Every model in this workshop expects 16,000 samples per second.
TARGET_SR = 16_000

_backend_note_shown = False


class AudioLoadError(RuntimeError):
    """Raised when no available decoder can read an audio file."""


def _load_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    return torch.from_numpy(data.T.copy()), int(sr)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Read an audio file into a float32 tensor shaped (channels, samples), plus its sample rate.

    Tries ``torchaudio.load`` first (as on the slides) and falls back to
    ``soundfile`` when TorchCodec or FFmpeg is missing.
    """
    global _backend_note_shown
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(
            f"Audio file not found: {path}. "
            "Fix: run the setup cell (cell 0) again so the clips are downloaded."
        )
    try:
        wav, sr = torchaudio.load(str(path))
        return wav.to(torch.float32), int(sr)
    except Exception as torchaudio_err:  # TorchCodec/FFmpeg missing, or an old backend
        try:
            wav, sr = _load_with_soundfile(path)
        except Exception as sf_err:
            raise AudioLoadError(
                f"Could not read {path.name} with torchaudio ({torchaudio_err}) "
                f"or soundfile ({sf_err}). Fix: run `pip install soundfile` and try again."
            ) from sf_err
        if not _backend_note_shown:
            print(
                "ℹ️  torchaudio.load is not available here (it needs TorchCodec + FFmpeg), "
                "so paminaw is reading audio with soundfile instead. Results are identical."
            )
            _backend_note_shown = True
        return wav, sr


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """Average all channels into one, keeping the (1, samples) shape."""
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    return wav.mean(dim=0, keepdim=True)


def resample_16k(wav: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Resample ``wav`` from ``sr`` to ``target_sr`` (16 kHz by default); no-op if already there."""
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def load_16k_mono(path: str | Path) -> torch.Tensor:
    """Load a file and return a mono 16 kHz tensor shaped (1, samples)."""
    wav, sr = load_audio(path)
    return resample_16k(to_mono(wav), sr)


def mel_spectrogram(
    wav: torch.Tensor,
    sample_rate: int = TARGET_SR,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """Mel spectrogram shaped (channels, n_mels, frames).

    Defaults match Whisper's front end: 25 ms windows (400 samples), a 10 ms hop
    (160 samples) and 80 mel bands.
    """
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return transform(wav)


def log_mel(mel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Natural log of a mel spectrogram; ``eps`` avoids log(0) in silent frames."""
    return torch.log(mel + eps)
