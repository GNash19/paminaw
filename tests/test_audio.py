"""Audio loading, mono mix, resampling and mel features."""

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from paminaw import audio
from paminaw.audio import (
    AudioLoadError,
    load_16k_mono,
    load_audio,
    log_mel,
    mel_spectrogram,
    resample_16k,
    to_mono,
)


@pytest.fixture
def stereo_44k(tmp_path):
    sr, n = 44_100, 44_100  # one second
    t = np.arange(n) / sr
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.25 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1).astype("float32"), sr)
    return path


def test_load_audio_returns_channels_by_samples(stereo_44k):
    wav, sr = load_audio(stereo_44k)
    assert sr == 44_100
    assert wav.shape == (2, 44_100)
    assert wav.dtype == torch.float32


def test_soundfile_fallback_matches(stereo_44k, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("Could not load libtorchcodec")

    monkeypatch.setattr(torchaudio, "load", broken)
    monkeypatch.setattr(audio, "_backend_note_shown", False)
    wav, sr = load_audio(stereo_44k)
    assert (wav.shape, sr) == ((2, 44_100), 44_100)
    assert torch.allclose(
        wav[0, :5], torch.tensor(sf.read(stereo_44k)[0][:5, 0], dtype=torch.float32)
    )


def test_missing_file_explains_fix(tmp_path):
    with pytest.raises(AudioLoadError, match="Fix: run the setup cell"):
        load_audio(tmp_path / "nope.wav")


def test_unreadable_file_explains_fix(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioLoadError, match="Fix"):
        load_audio(bad)


def test_to_mono_averages_channels():
    wav = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    assert torch.equal(to_mono(wav), torch.tensor([[2.0, 4.0]]))
    assert to_mono(torch.zeros(10)).shape == (1, 10)


def test_resample_to_16k_changes_length_and_noop_when_already_16k():
    wav = torch.randn(1, 44_100)
    assert resample_16k(wav, 44_100).shape == (1, 16_000)
    same = torch.randn(1, 16_000)
    assert resample_16k(same, 16_000) is same
    assert resample_16k(same, 16_000, target_sr=8_000).shape == (1, 8_000)


def test_load_16k_mono(stereo_44k):
    assert load_16k_mono(stereo_44k).shape == (1, 16_000)


def test_mel_shape_matches_slide_settings():
    wav = torch.randn(1, 16_000)
    mel = mel_spectrogram(wav)  # n_fft=400, hop=160, n_mels=80
    assert mel.shape == (1, 80, 16_000 // 160 + 1)


def test_log_mel_is_finite_on_silence():
    mel = mel_spectrogram(torch.zeros(1, 1_600))
    out = log_mel(mel)
    assert torch.isfinite(out).all()
    assert out.min() == pytest.approx(np.log(1e-6), rel=1e-4)
