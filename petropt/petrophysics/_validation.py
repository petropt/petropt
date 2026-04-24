"""Shared validation helpers for petrophysics calculations."""

from __future__ import annotations

import math


def validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_fraction(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
