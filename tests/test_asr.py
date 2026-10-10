"""ASR wrappers with the models mocked out (no downloads)."""

import types

import numpy as np
import pytest
import torch
import transformers

from paminaw import asr


def test_pick_device_dtype_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert asr.pick_device_dtype() == ("cpu", torch.float32)


def test_pick_device_dtype_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert asr.pick_device_dtype() == ("cuda", torch.float16)
    assert asr.pick_device_dtype(prefer_gpu=False) == ("cpu", torch.float32)


@pytest.mark.parametrize(
    ("version", "key"),
    [("4.44.2", "torch_dtype"), ("4.55.4", "torch_dtype"), ("4.56.0", "dtype"),
     ("5.18.0", "dtype"), ("4.57.0.dev0", "dtype")],
)  # fmt: skip
def test_dtype_kwargs_follows_transformers_version(monkeypatch, version, key):
    monkeypatch.setattr(transformers, "__version__", version)
    assert asr.dtype_kwargs(torch.float16) == {key: torch.float16}


def test_resolve_model_prefers_offline_copy(tmp_path):
    local = tmp_path / "models" / "openai--whisper-small"
    local.mkdir(parents=True)
    assert asr.resolve_model("openai/whisper-small", tmp_path) == "openai/whisper-small"
    (local / "config.json").write_text("{}")
    assert asr.resolve_model("openai/whisper-small", tmp_path) == str(local)


def test_to_array_shapes():
    assert asr.to_array(torch.zeros(1, 160)).shape == (160,)
    assert asr.to_array(np.zeros((2, 160))).shape == (160,)
    with pytest.raises(ValueError, match="Fix"):
        asr.to_array(np.zeros((1, 2, 3)))


class FakePipe:
    def __init__(self):
        self.calls = []
        self.feature_extractor = lambda arr, sampling_rate, return_tensors: types.SimpleNamespace(
            input_features=torch.zeros(1, 80, 3000)
        )
        self.model = types.SimpleNamespace(detect_language=lambda feats: torch.tensor([50348]))
        self.tokenizer = types.SimpleNamespace(convert_ids_to_tokens=lambda i: "<|tl|>")

    def __call__(self, inputs, generate_kwargs):
        self.calls.append((inputs, generate_kwargs))
        return {"text": " Kumusta ka na?"}


@pytest.fixture
def fake_whisper(monkeypatch):
    pipe = FakePipe()
    seen = {}

    def fake_pipeline(task, model, device, **kwargs):
        seen.update(task=task, model=model, device=device, **kwargs)
        return pipe

    monkeypatch.setattr(
        asr, "_hf", lambda: types.SimpleNamespace(__version__="5.18.0", pipeline=fake_pipeline)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return pipe, seen


def test_whisper_transcriber_passes_language_and_task(fake_whisper):
    pipe, seen = fake_whisper
    w = asr.WhisperTranscriber("openai/whisper-tiny")
    assert seen["task"] == "automatic-speech-recognition"
    assert seen["device"] == "cpu"
    assert w.transcribe(torch.zeros(1, 16_000)) == " Kumusta ka na?"
    inputs, gen = pipe.calls[-1]
    assert gen["language"] == "tagalog" and gen["task"] == "transcribe"
    assert inputs["sampling_rate"] == 16_000 and inputs["raw"].shape == (16_000,)
    w.transcribe(torch.zeros(1, 16_000), language=None)
    assert "language" not in pipe.calls[-1][1]


def test_whisper_detect_language(fake_whisper):
    w = asr.WhisperTranscriber("openai/whisper-tiny")
    assert w.detect_language(torch.zeros(1, 16_000)) == "<|tl|>"


def test_whisper_refuses_fp16_on_cpu(fake_whisper):
    with pytest.raises(ValueError, match=r"Fix: use dtype=torch\.float32"):
        asr.WhisperTranscriber("openai/whisper-tiny", device="cpu", dtype=torch.float16)


def test_mms_transcriber_switches_adapter_and_decodes_argmax(monkeypatch):
    events = []

    class FakeTokenizer:
        def set_target_lang(self, code):
            events.append(("tokenizer", code))

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @classmethod
        def from_pretrained(cls, path, **kw):
            return cls()

        def __call__(self, arr, sampling_rate, return_tensors):
            return {"input_values": torch.zeros(1, len(arr))}

        def decode(self, ids):
            return "-".join(str(i) for i in ids.tolist())

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            events.append(("dtype", kw))
            return cls()

        def eval(self):
            return self

        def load_adapter(self, code):
            events.append(("adapter", code))

        def to(self, device):
            return self

        def __call__(self, input_values):
            logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
            return types.SimpleNamespace(logits=logits)

    fake_hf = types.SimpleNamespace(
        __version__="5.18.0", AutoProcessor=FakeProcessor, Wav2Vec2ForCTC=FakeModel
    )
    monkeypatch.setattr(asr, "_hf", lambda: fake_hf)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = asr.MMSTranscriber(lang="cebuano")
    assert ("tokenizer", "ceb") in events and ("adapter", "ceb") in events
    assert next(e for e in events if e[0] == "dtype")[1] == {"dtype": torch.float32}
    assert m.transcribe(np.zeros(16_000, dtype=np.float32)) == "1-0-1"
    m.set_language("tgl")
    assert m.lang == "tgl" and ("adapter", "tgl") in events
