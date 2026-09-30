"""Quantisation (FR-08): weight-only int8/int4 with an AWQ-style activation-aware variant,
plus the serving-cost model that turns bits per weight into dollars per million tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np


@dataclass
class QTensor:
    q: np.ndarray  # integer codes
    scale: np.ndarray  # per output channel
    bits: int
    col_scale: np.ndarray | None = None  # AWQ per-input-channel scaling, folded back at dequantise


def quantize(w: np.ndarray, bits: int, col_scale: np.ndarray | None = None) -> QTensor:
    """Symmetric per-output-channel post-training quantisation."""
    scaled = w * col_scale if col_scale is not None else w
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(scaled).max(axis=1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(scaled / scale), -qmax, qmax).astype(np.int8)
    return QTensor(q, scale.astype(np.float32), bits, col_scale)


def dequantize(t: QTensor) -> np.ndarray:
    w = t.q.astype(np.float32) * t.scale
    return w / t.col_scale if t.col_scale is not None else w


def output_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    return float(np.mean((x @ w.T - x @ w_hat.T) ** 2))


def awq_quantize(w: np.ndarray, x_calib: np.ndarray, bits: int, grid: int = 11) -> tuple[QTensor, float]:
    """Activation-aware: scale up the input channels that carry large activations before rounding,
    so their weights keep more precision. Searches the exponent alpha on calibration data."""
    act = np.abs(x_calib).mean(axis=0) + 1e-6
    best: tuple[float, QTensor, float] | None = None
    for alpha in np.linspace(0, 1, grid):
        s = act**alpha
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
        t = quantize(w, bits, s)
        err = output_error(w, dequantize(t), x_calib)
        if best is None or err < best[0]:
            best = (err, t, float(alpha))
    assert best is not None
    return best[1], best[2]


def memory_bytes(n_params: int, bits: int) -> int:
    return n_params * bits // 8


# --- Serving economics (assumptions, stated once and reused everywhere) ---------------------

GPU_HOURLY_USD = Decimal("1.20")  # one L40S-class node
FP16_TOKENS_PER_S = 1500  # batched throughput of an 8B model at fp16
UTILISATION = Decimal("0.40")
COMPUTE_CAP = 3.4  # beyond this, decoding is compute-bound rather than memory-bandwidth-bound


def throughput_tokens_per_s(bits: int) -> float:
    """Decoding is memory-bandwidth-bound: fewer bytes per weight, proportionally more tokens/s."""
    return FP16_TOKENS_PER_S * min(16 / bits, COMPUTE_CAP)


def serving_cost_per_mtok(bits: int) -> Decimal:
    tokens_per_hour = Decimal(throughput_tokens_per_s(bits)) * 3600 * UTILISATION
    return (GPU_HOURLY_USD / tokens_per_hour * Decimal(1_000_000)).quantize(Decimal("0.01"))


def monthly_gpu_cost() -> Decimal:
    """Fixed cost of keeping the node up; it is paid whether or not traffic fills it."""
    return GPU_HOURLY_USD * 24 * 30
