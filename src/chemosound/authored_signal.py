from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class AuthoredComplexSTFT:
    channels: tuple[np.ndarray, ...]
    cross_spectra: dict[tuple[int, int], np.ndarray]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError('at least one source channel is required')
        expected = {(left, right) for left in range(len(self.channels)) for right in range(left + 1, len(self.channels))}
        if set(self.cross_spectra) != expected:
            raise ValueError('every source-index pair must retain a cross spectrum')

def keep_complex_stft(channel_stft: Sequence[np.ndarray]) -> AuthoredComplexSTFT:
    if len(channel_stft) == 0:
        raise ValueError('at least one source channel is required')
    channels: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for coefficients in channel_stft:
        values = np.asarray(coefficients, dtype=np.complex128)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
            raise ValueError('each STFT must have frames by positive-frequency bins')
        if np.any(~np.isfinite(values.real)) or np.any(~np.isfinite(values.imag)):
            raise ValueError('STFT coefficients must be finite')
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError('all source channels must share frame-frequency geometry')
        frozen = values.copy()
        frozen.setflags(write=False)
        channels.append(frozen)
    frozen_channels = tuple(channels)
    cross_spectra: dict[tuple[int, int], np.ndarray] = {}
    for left in range(len(frozen_channels)):
        for right in range(left + 1, len(frozen_channels)):
            cross = frozen_channels[left] * np.conjugate(frozen_channels[right])
            cross.setflags(write=False)
            cross_spectra[left, right] = cross
    return AuthoredComplexSTFT(frozen_channels, cross_spectra)

def channel_power_fields(field: AuthoredComplexSTFT) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for coefficients in field.channels:
        power = coefficients.real * coefficients.real + coefficients.imag * coefficients.imag
        power.setflags(write=False)
        result.append(power)
    return tuple(result)
