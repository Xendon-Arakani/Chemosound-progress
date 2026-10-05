from __future__ import annotations
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True, slots=True)
class PairMeasure:
    support: np.ndarray
    probabilities: np.ndarray
    w1_error_bound: float
    coordinate_quantum: float
    input_count: int
    unordered_pair_count: int
    normalization: float
    numeric_mass_error: float

def _fft_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError('FFT length must be positive')
    return 1 << (value - 1).bit_length()

def weighted_pair_grid(log_positions: np.ndarray, weights: np.ndarray, *, coordinate_quantum: float) -> PairMeasure:
    x = np.asarray(log_positions, dtype=np.float64)
    raw_weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or raw_weights.ndim != 1 or x.shape != raw_weights.shape:
        raise ValueError('positions and weights must be equal-length vectors')
    if x.size < 2:
        raise ValueError('at least two atoms are required')
    if not np.all(np.isfinite(x)):
        raise ValueError('positions must be finite')
    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0):
        raise ValueError('weights must be finite and positive')
    if not math.isfinite(coordinate_quantum) or coordinate_quantum <= 0:
        raise ValueError('coordinate_quantum must be finite and positive')
    scale = float(np.max(raw_weights))
    normalized_weights = raw_weights / scale
    normalized_weights /= float(np.sum(normalized_weights, dtype=np.float64))
    order = np.argsort(x, kind='stable')
    x = x[order]
    normalized_weights = normalized_weights[order]
    origin = float(x[0])
    group_index = np.rint((x - origin) / coordinate_quantum).astype(np.int64)
    group_count = int(group_index[-1]) + 1
    grouped_weight = np.bincount(group_index, weights=normalized_weights, minlength=group_count).astype(np.float64, copy=False)
    group_weight_square = np.bincount(group_index, weights=normalized_weights * normalized_weights, minlength=group_count).astype(np.float64, copy=False)
    fft_length = _fft_power_of_two(2 * group_count - 1)
    spectrum = np.fft.rfft(grouped_weight, n=fft_length)
    correlation = np.fft.irfft(np.conjugate(spectrum) * spectrum, n=fft_length)[:group_count]
    correlation = np.maximum(correlation, 0.0)
    pair_normalization = float((1.0 - np.sum(normalized_weights * normalized_weights, dtype=np.float64)) / 2.0)
    if not math.isfinite(pair_normalization) or pair_normalization <= 0:
        raise ValueError('positive weights do not define a stable pair measure')
    mass = correlation
    mass[0] = float(np.sum((grouped_weight * grouped_weight - group_weight_square) / 2.0, dtype=np.float64))
    computed_mass = float(np.sum(mass, dtype=np.float64))
    numeric_mass_error = abs(computed_mass - pair_normalization)
    if computed_mass <= 0:
        raise RuntimeError('pair autocorrelation produced no mass')
    probabilities = mass / computed_mass
    support = np.arange(group_count, dtype=np.float64) * coordinate_quantum
    quantized_x = origin + group_index * coordinate_quantum
    max_coordinate_error = float(np.max(np.abs(x - quantized_x)))
    return PairMeasure(support=support, probabilities=probabilities, w1_error_bound=2.0 * max_coordinate_error, coordinate_quantum=coordinate_quantum, input_count=int(x.size), unordered_pair_count=int(x.size * (x.size - 1) // 2), normalization=pair_normalization, numeric_mass_error=numeric_mass_error)

def molecule_pair_grid(nu_cm_1: np.ndarray, einstein_a_s_1: np.ndarray, *, coordinate_quantum: float) -> PairMeasure:
    nu = np.asarray(nu_cm_1, dtype=np.float64)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError('wavenumbers must be finite and positive')
    return weighted_pair_grid(np.log(nu), einstein_a_s_1, coordinate_quantum=coordinate_quantum)

def wasserstein_atoms(left: PairMeasure, right: PairMeasure) -> tuple[float, float]:
    support = np.union1d(left.support, right.support)
    left_mass = np.zeros(support.size, dtype=np.float64)
    right_mass = np.zeros(support.size, dtype=np.float64)
    np.add.at(left_mass, np.searchsorted(support, left.support), left.probabilities)
    np.add.at(right_mass, np.searchsorted(support, right.support), right.probabilities)
    value = float(np.sum(np.abs(np.cumsum(left_mass - right_mass)[:-1]) * np.diff(support)))
    error = left.w1_error_bound + right.w1_error_bound
    return (value, error)

def write_pair_measure(path: Path, measure: PairMeasure, metadata: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    full_metadata = {**metadata, 'coordinate_quantum': measure.coordinate_quantum, 'wasserstein_error_bound': measure.w1_error_bound, 'input_count': measure.input_count, 'unordered_pair_count': str(measure.unordered_pair_count), 'normalization': measure.normalization, 'numeric_mass_error': measure.numeric_mass_error}
    with partial.open('wb') as stream:
        np.savez_compressed(stream, support=measure.support, probabilities=measure.probabilities, metadata_json=np.asarray(json.dumps(full_metadata, sort_keys=True, separators=(',', ':'))))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(8 * 1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()
