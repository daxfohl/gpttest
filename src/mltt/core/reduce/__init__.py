"""Reduction utilities split by rule family (beta, iota, whnf, normalize)."""

from .beta import beta_head_step, beta_step
from .iota import iota_head_step, iota_step
from .normalize import normalize, normalize_step
from .whnf import whnf, whnf_step

__all__ = [
    "beta_head_step",
    "beta_step",
    "iota_head_step",
    "iota_step",
    "normalize",
    "normalize_step",
    "whnf",
    "whnf_step",
]
