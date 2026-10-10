"""Plots for the workshop: waveforms, mel spectrograms and convention charts.

Styled for a projector: light background, large fonts, high contrast and a
colour-blind-safe palette (Okabe & Ito).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes

from .audio import TARGET_SR
from .normalize import CONVENTIONS

__all__ = [
    "PALETTE",
    "plot_convention_bars",
    "plot_mel",
    "plot_waveform",
    "set_style",
]

#: Okabe–Ito colours, readable for most kinds of colour blindness.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def set_style(font_size: int = 16) -> None:
    """Large, light, high-contrast matplotlib defaults for slides and laptops."""
    plt.rcParams.update({
        "figure.figsize": (12, 4),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "font.size": font_size,
    })  # fmt: skip


def _ax(ax: Axes | None, figsize: tuple[float, float] = (12, 3.5)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_waveform(
    wav: torch.Tensor | np.ndarray,
    sr: int = TARGET_SR,
    title: str = "Waveform",
    ax: Axes | None = None,
    color: str | None = None,
) -> Axes:
    """Amplitude over time (seconds) for a mono clip."""
    y = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    y = y.reshape(-1) if y.ndim == 1 or y.shape[0] == 1 else y.mean(axis=0)
    ax = _ax(ax)
    t = np.arange(len(y)) / sr
    ax.plot(t, y, linewidth=0.6, color=color or PALETTE[0])
    ax.set(title=title, xlabel="time (s)", ylabel="amplitude", xlim=(0, t[-1] if len(t) else 1))
    return ax


def plot_mel(
    log_mel: torch.Tensor | np.ndarray,
    sr: int = TARGET_SR,
    hop_length: int = 160,
    title: str = "Log-mel spectrogram",
    ax: Axes | None = None,
) -> Axes:
    """Log-mel spectrogram as an image: time on x, mel band on y, brighter = louder."""
    m = log_mel.detach().cpu().numpy() if isinstance(log_mel, torch.Tensor) else np.asarray(log_mel)
    if m.ndim == 3:
        m = m[0]
    ax = _ax(ax, figsize=(12, 4))
    seconds = m.shape[1] * hop_length / sr
    img = ax.imshow(
        m, origin="lower", aspect="auto", cmap="magma", extent=(0, seconds, 0, m.shape[0])
    )
    ax.set(title=title, xlabel="time (s)", ylabel="mel band")
    ax.grid(False)
    plt.colorbar(img, ax=ax, label="log energy", pad=0.01)
    return ax


def plot_convention_bars(
    scores: pd.DataFrame,
    lang: str,
    metric: str = "wer",
    conventions: Sequence[str] = CONVENTIONS,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Grouped bars: one group per convention, one bar per system, for one language."""
    df = scores[scores["lang"] == lang]
    systems = list(dict.fromkeys(df["system"]))
    ax = _ax(ax, figsize=(12, 5))
    width = 0.8 / max(len(systems), 1)
    x = np.arange(len(conventions))
    for i, system in enumerate(systems):
        vals = [
            df[(df["system"] == system) & (df["convention"] == c)][metric].iloc[0]
            for c in conventions
        ]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=system,
                      color=PALETTE[i % len(PALETTE)])  # fmt: skip
        ax.bar_label(bars, fmt="%.1f", fontsize=11, padding=2)
    ax.set_xticks(x, conventions)
    label = df["subset"].iloc[0] if "subset" in df.columns and len(df) else ""
    ax.set(
        ylabel=f"{metric.upper()} (%)",
        title=title or f"{lang.capitalize()}: {metric.upper()} by scoring convention — {label}",
    )
    ax.legend(loc="upper right")
    ax.margins(y=0.15)
    return ax
