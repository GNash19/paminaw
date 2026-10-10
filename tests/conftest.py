"""Test setup: block all network access so the suite is offline and fast."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests.")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
