from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
import numpy as np
from .authored_signal import AuthoredComplexSTFT, channel_power_fields

@dataclass(frozen=True, slots=True)
class AtomicMeasure:
    support: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if support.ndim != 1 or probabilities.ndim != 1:
            raise ValueError('support and probabilities must be vectors')
        if support.shape != probabilities.shape or support.size == 0:
            raise ValueError('support and probabilities must be nonempty and aligned')
        if np.any(~np.isfinite(support)) or np.any(support < 0):
            raise ValueError('support must be finite and nonnegative')
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError('probabilities must be finite and nonnegative')
        if not math.isclose(float(np.sum(probabilities, dtype=np.float64)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('probabilities must sum to one')
        if np.any(np.diff(support) <= 0):
            raise ValueError('support must be strictly increasing')

@dataclass(frozen=True, slots=True)
class SurrogateAudit:
    exact_circular_shift: bool
    bin_multisets_kept: bool
    mean_power_max_err: float
    moment_max_error: float
    circular_acf_max_err: float
    cross_bin_zero_delta: float

@dataclass(frozen=True, slots=True)
class ChannelPairMeasures:
    within: tuple[AtomicMeasure, ...]
    cross: dict[tuple[int, int], AtomicMeasure]

    def __post_init__(self) -> None:
        if not self.within:
            raise ValueError('at least one source channel is required')
        expected_pairs = {(left, right) for left in range(len(self.within)) for right in range(left + 1, len(self.within))}
        if set(self.cross) != expected_pairs:
            raise ValueError('cross-channel measures must retain every source-index pair')

def _aggregate_atoms(support: np.ndarray, weights: np.ndarray) -> AtomicMeasure:
    support = np.asarray(support, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if support.shape != weights.shape or support.ndim != 1:
        raise ValueError('atom supports and weights must be aligned vectors')
    keep = weights > 0
    if not np.any(keep):
        raise ValueError('measure has no positive mass')
    support = support[keep]
    weights = weights[keep]
    unique, inverse = np.unique(support, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=unique.size).astype(np.float64, copy=False)
    mass /= np.sum(mass, dtype=np.float64)
    return AtomicMeasure(unique, mass)

def weighted_pair_exact(log_positions: np.ndarray, weights: np.ndarray) -> AtomicMeasure:
    positions = np.asarray(log_positions, dtype=np.float64)
    normalized = np.asarray(weights, dtype=np.float64)
    if positions.ndim != 1 or normalized.ndim != 1 or positions.shape != normalized.shape:
        raise ValueError('positions and weights must be aligned vectors')
    if positions.size < 2:
        raise ValueError('at least two positions are required')
    if np.any(~np.isfinite(positions)):
        raise ValueError('positions must be finite')
    if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0):
        raise ValueError('weights must be finite and positive')
    normalized = normalized / np.sum(normalized, dtype=np.float64)
    left, right = np.triu_indices(positions.size, k=1)
    separation = np.abs(positions[right] - positions[left])
    pair_weight = normalized[left] * normalized[right]
    return _aggregate_atoms(separation, pair_weight)

def molecule_pair_exact(nu_cm_1: np.ndarray, einstein_a_s_1: np.ndarray) -> AtomicMeasure:
    nu = np.asarray(nu_cm_1, dtype=np.float64)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError('wavenumbers must be finite and positive')
    return weighted_pair_exact(np.log(nu), einstein_a_s_1)

def _validate_power(power: np.ndarray, positive_bin_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(power, dtype=np.float64)
    bins = np.asarray(positive_bin_numbers, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('power must have shape (frames, positive-frequency bins)')
    if bins.ndim != 1 or bins.size != values.shape[1]:
        raise ValueError('bin numbers must align with the power columns')
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError('power requires at least one frame and two bins')
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError('power must be finite and nonnegative')
    if np.any(~np.isfinite(bins)) or np.any(bins <= 0) or np.any(np.diff(bins) <= 0):
        raise ValueError('positive bin numbers must be finite and increasing')
    return (values, bins)

def music_pair_exact(power: np.ndarray, positive_bin_numbers: np.ndarray | None=None) -> AtomicMeasure:
    if positive_bin_numbers is None:
        positive_bin_numbers = np.arange(1, np.asarray(power).shape[1] + 1)
    values, bins = _validate_power(power, positive_bin_numbers)
    cross_power = values.T @ values
    left, right = np.triu_indices(bins.size, k=1)
    separation = np.abs(np.log(bins[right]) - np.log(bins[left]))
    return _aggregate_atoms(separation, cross_power[left, right])

def music_pair_grid(power: np.ndarray, *, coordinate_quantum: float, positive_bin_numbers: np.ndarray | None=None) -> tuple[AtomicMeasure, float]:
    if positive_bin_numbers is None:
        positive_bin_numbers = np.arange(1, np.asarray(power).shape[1] + 1)
    values, bins = _validate_power(power, positive_bin_numbers)
    if not math.isfinite(coordinate_quantum) or coordinate_quantum <= 0:
        raise ValueError('coordinate_quantum must be finite and positive')
    log_bins = np.log(bins)
    indexes = np.rint(log_bins / coordinate_quantum).astype(np.int64)
    quantized = indexes.astype(np.float64) * coordinate_quantum
    left, right = np.triu_indices(bins.size, k=1)
    cross_power = values.T @ values
    separation = np.abs(indexes[right] - indexes[left]) * coordinate_quantum
    error_bound = 2.0 * float(np.max(np.abs(log_bins - quantized)))
    return (_aggregate_atoms(separation, cross_power[left, right]), error_bound)

def music_pair_fft_grid(power: np.ndarray, *, coordinate_quantum: float, positive_bin_numbers: np.ndarray | None=None) -> tuple[AtomicMeasure, float]:
    if positive_bin_numbers is None:
        positive_bin_numbers = np.arange(1, np.asarray(power).shape[1] + 1)
    values, bins = _validate_power(power, positive_bin_numbers)
    if not math.isfinite(coordinate_quantum) or coordinate_quantum <= 0:
        raise ValueError('coordinate_quantum must be finite and positive')
    log_bins = np.log(bins)
    indexes = np.rint(log_bins / coordinate_quantum).astype(np.int64)
    group_count = int(indexes[-1]) + 1
    grouped = np.zeros((values.shape[0], group_count), dtype=np.float64)
    for column, group in enumerate(indexes):
        grouped[:, group] += values[:, column]
    fft_length = 1 << (2 * group_count - 2).bit_length()
    transformed = np.fft.rfft(grouped, n=fft_length, axis=1)
    correlation = np.fft.irfft(np.sum(np.conjugate(transformed) * transformed, axis=0), n=fft_length)[:group_count]
    roundoff_floor = np.finfo(np.float64).eps * max(1.0, float(abs(correlation[0]))) * fft_length * 16.0
    correlation[np.abs(correlation) <= roundoff_floor] = 0.0
    correlation = np.maximum(correlation, 0.0)
    correlation[0] = max(0.0, float(correlation[0] - np.sum(values * values, dtype=np.float64)) / 2.0)
    support = np.arange(group_count, dtype=np.float64) * coordinate_quantum
    quantized = indexes.astype(np.float64) * coordinate_quantum
    error_bound = 2.0 * float(np.max(np.abs(log_bins - quantized)))
    return (_aggregate_atoms(support, correlation), error_bound)

def cross_pair_exact(left_power: np.ndarray, right_power: np.ndarray, positive_bin_numbers: np.ndarray | None=None) -> AtomicMeasure:
    left_values = np.asarray(left_power, dtype=np.float64)
    right_values = np.asarray(right_power, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError('paired channels must share frame and frequency geometry')
    if positive_bin_numbers is None:
        positive_bin_numbers = np.arange(1, left_values.shape[1] + 1)
    left_values, bins = _validate_power(left_values, positive_bin_numbers)
    right_values, right_bins = _validate_power(right_values, positive_bin_numbers)
    if not np.array_equal(bins, right_bins):
        raise ValueError('paired channels must share frequency bins')
    cross_power = left_values.T @ right_values
    left_index, right_index = np.indices(cross_power.shape)
    separation = np.abs(np.log(bins[right_index.ravel()]) - np.log(bins[left_index.ravel()]))
    return _aggregate_atoms(separation, cross_power.ravel())

def keep_channel_pairs(channel_power: tuple[np.ndarray, ...] | list[np.ndarray], positive_bin_numbers: np.ndarray | None=None) -> ChannelPairMeasures:
    if not channel_power:
        raise ValueError('at least one source channel is required')
    channels = tuple(channel_power)
    shapes = {np.asarray(power).shape for power in channels}
    if len(shapes) != 1:
        raise ValueError('all channels must share frame and frequency geometry')
    within = tuple((music_pair_exact(power, positive_bin_numbers) for power in channels))
    cross = {(left, right): cross_pair_exact(channels[left], channels[right], positive_bin_numbers) for left in range(len(channels)) for right in range(left + 1, len(channels))}
    return ChannelPairMeasures(within=within, cross=cross)

def stft_channel_pairs(field: AuthoredComplexSTFT, positive_bin_numbers: np.ndarray | None=None) -> ChannelPairMeasures:
    return keep_channel_pairs(channel_power_fields(field), positive_bin_numbers)

def channel_mol_dist(molecule: AtomicMeasure, recording: ChannelPairMeasures) -> dict:
    return {'within': {str(source_index): wasserstein_1(molecule, measure) for source_index, measure in enumerate(recording.within)}, 'cross': {f'{left}|{right}': wasserstein_1(molecule, measure) for (left, right), measure in recording.cross.items()}}

def wasserstein_1(left: AtomicMeasure, right: AtomicMeasure) -> float:
    support = np.union1d(left.support, right.support)
    left_mass = np.zeros(support.size, dtype=np.float64)
    right_mass = np.zeros(support.size, dtype=np.float64)
    np.add.at(left_mass, np.searchsorted(support, left.support), left.probabilities)
    np.add.at(right_mass, np.searchsorted(support, right.support), right.probabilities)
    return float(np.sum(np.abs(np.cumsum(left_mass - right_mass)[:-1]) * np.diff(support), dtype=np.float64))

def circular_shift_power(power: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    offsets = np.asarray(shifts, dtype=np.int64)
    if values.ndim != 2 or offsets.shape != (values.shape[1],):
        raise ValueError('one circular offset is required per frequency bin')
    frames = np.arange(values.shape[0], dtype=np.int64)[:, None]
    columns = np.arange(values.shape[1], dtype=np.int64)[None, :]
    return values[(frames + offsets[None, :]) % values.shape[0], columns]

def circular_acf(trajectory: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('trajectory must be a nonempty vector')
    return np.asarray([np.dot(values, np.roll(values, -lag)) for lag in range(values.size)], dtype=np.float64)

def audit_circular_shift(original: np.ndarray, surrogate: np.ndarray, shifts: np.ndarray) -> SurrogateAudit:
    before = np.asarray(original, dtype=np.float64)
    after = np.asarray(surrogate, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError('original and surrogate must be aligned power matrices')
    expected = circular_shift_power(before, shifts)
    multiset_preserved = all((np.array_equal(np.sort(before[:, column]), np.sort(after[:, column])) for column in range(before.shape[1])))
    moment_errors = []
    acf_errors = []
    for column in range(before.shape[1]):
        for order in range(1, 5):
            moment_errors.append(abs(float(np.mean(before[:, column] ** order)) - float(np.mean(after[:, column] ** order))))
        acf_errors.append(float(np.max(np.abs(circular_acf(before[:, column]) - circular_acf(after[:, column])))))
    before_cross = before.T @ before
    after_cross = after.T @ after
    off_diagonal = ~np.eye(before.shape[1], dtype=bool)
    return SurrogateAudit(exact_circular_shift=bool(np.array_equal(expected, after)), bin_multisets_kept=multiset_preserved, mean_power_max_err=float(np.max(np.abs(np.mean(before, axis=0) - np.mean(after, axis=0)))), moment_max_error=max(moment_errors, default=0.0), circular_acf_max_err=max(acf_errors, default=0.0), cross_bin_zero_delta=float(np.max(np.abs(before_cross[off_diagonal] - after_cross[off_diagonal]))))

def replicate_seed(run_seed: str, decoded_sha256: str, scale_index: int, channel_index: int, replicate_index: int) -> int:
    if not run_seed or not decoded_sha256:
        raise ValueError('run seed and decoded audio identity are required')
    if scale_index < 0 or channel_index < 0 or replicate_index < 1:
        raise ValueError('indexes are outside the frozen seed domain')
    payload = f'{run_seed}|{decoded_sha256}|{scale_index}|{channel_index}|{replicate_index}|philox-v1'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')

def philox_shifts(frame_count: int, positive_bins: int, *, run_seed: str, decoded_sha256: str, scale_index: int, channel_index: int, replicate_index: int) -> np.ndarray:
    if frame_count < 1 or positive_bins < 2:
        raise ValueError('a surrogate requires frames and at least two bins')
    seed = replicate_seed(run_seed, decoded_sha256, scale_index, channel_index, replicate_index)
    generator = np.random.Generator(np.random.Philox(seed))
    return generator.integers(0, frame_count, size=positive_bins, dtype=np.int64)

def dataset_statistics(real_distances: np.ndarray, surrogate_distances: np.ndarray) -> dict:
    real = np.asarray(real_distances, dtype=np.float64)
    surrogate = np.asarray(surrogate_distances, dtype=np.float64)
    if real.ndim != 2 or surrogate.ndim != 3 or surrogate.shape[1:] != real.shape:
        raise ValueError('expected real[M,C] and surrogate[B,M,C]')
    if surrogate.shape[0] < 2:
        raise ValueError('at least two surrogate replicates are required')
    if np.any(~np.isfinite(real)) or np.any(~np.isfinite(surrogate)):
        raise ValueError('distances must be finite')
    t_real = float(np.mean(real, dtype=np.float64))
    t_surrogate = np.mean(surrogate, axis=(1, 2), dtype=np.float64)
    surrogate_mean = float(np.mean(t_surrogate, dtype=np.float64))
    surrogate_sd = float(np.std(t_surrogate, ddof=1, dtype=np.float64))
    delta = surrogate_mean - t_real
    z_score = delta / surrogate_sd if surrogate_sd > 0 else math.nan
    replicates = int(surrogate.shape[0])
    return {'surrogate_replicates': replicates, 't_real': t_real, 't_surrogate': t_surrogate, 'surrogate_mean': surrogate_mean, 'surrogate_sd_sample': surrogate_sd, 'delta_surrogate_minus_real': delta, 'z_score': z_score, 'monte_carlo_p': float((1 + np.count_nonzero(t_surrogate <= t_real)) / (replicates + 1)), 'probability_superiority': float(np.mean(real[None, :, :] < surrogate, dtype=np.float64)), 'species_effect': np.mean(surrogate - real[None, :, :], axis=(0, 1), dtype=np.float64), 'song_effect': np.mean(surrogate - real[None, :, :], axis=(0, 2), dtype=np.float64), 'surrogate_sd_ddof': 1}
