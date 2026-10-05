from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from .model import CancellationToken

@dataclass(frozen=True, slots=True)
class RelationConfig:
    protocol_version: str = '1.0+a01+a02+a03'
    pair_grid_version: str = 'music-relational-loggroup-source-index-v2'
    scales_seconds: tuple[float, ...] = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
    log_frequency_groups: int = 96
    relation_bins: int = 256
    frame_batch: int = 64
    window: str = 'symmetric_hann'
    hop_fraction: float = 0.25
    frame_boundary: str = 'full_frames_only'
    round_rule: str = 'positive_half_up'
    dc_bin: str = 'excluded'
    channel_rule: str = 'source_index_preserving_no_mix'

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    def validate(self) -> None:
        if self.scales_seconds != tuple((2 ** s / 64 for s in range(8))):
            raise ValueError('protocol 1.0 requires exactly eight scales 2^s/64')
        if self.log_frequency_groups < 8:
            raise ValueError('log_frequency_groups must be at least 8')
        if self.relation_bins < 16:
            raise ValueError('relation_bins must be at least 16')
        if self.frame_batch < 1:
            raise ValueError('frame_batch must be positive')

def _positive_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))

def _freq_group_edges(positive_bins: int, requested_groups: int) -> np.ndarray:
    raw = np.geomspace(1.0, float(positive_bins + 1), requested_groups + 1)
    edges = np.unique(np.rint(raw).astype(np.int64))
    edges[0] = 1
    if edges[-1] != positive_bins + 1:
        edges = np.append(edges, positive_bins + 1)
    if edges.size < 2:
        raise ValueError('audio scale has too few positive-frequency bins')
    return edges

def _channel_histogram(channel: np.ndarray, sample_rate: int, window_length: int, hop_length: int, config: RelationConfig, token: CancellationToken) -> tuple[np.ndarray, np.ndarray, dict]:
    if channel.size < window_length:
        raise ValueError(f'recording is shorter than the {window_length / sample_rate:g}s window')
    frames = np.lib.stride_tricks.sliding_window_view(channel, window_length)[::hop_length]
    frame_count = int(frames.shape[0])
    positive_bins = window_length // 2
    group_edges = _freq_group_edges(positive_bins, config.log_frequency_groups)
    starts = group_edges[:-1] - 1
    stops = group_edges[1:] - 1
    group_count = int(starts.size)
    first_bins = group_edges[:-1]
    last_bins = group_edges[1:] - 1
    log_centers = 0.5 * (np.log(first_bins) + np.log(last_bins))
    log_widths = np.log(last_bins) - np.log(first_bins)
    gram = np.zeros((group_count, group_count), dtype=np.float64)
    bin_power_square = np.zeros(group_count, dtype=np.float64)
    hann = np.hanning(window_length).astype(np.float64, copy=False)
    for offset in range(0, frame_count, config.frame_batch):
        token.checkpoint()
        block = np.asarray(frames[offset:offset + config.frame_batch], dtype=np.float64)
        spectrum = np.fft.rfft(block * hann, axis=1)[:, 1:]
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        cumulative = np.pad(np.cumsum(power, axis=1), ((0, 0), (1, 0)))
        grouped = cumulative[:, stops] - cumulative[:, starts]
        gram += grouped.T @ grouped
        square_cumulative = np.pad(np.cumsum(power * power, axis=1), ((0, 0), (1, 0)))
        bin_power_square += np.sum(square_cumulative[:, stops] - square_cumulative[:, starts], axis=0)
    token.checkpoint()
    upper_i, upper_j = np.triu_indices(group_count, k=1)
    cross_weights = gram[upper_i, upper_j]
    cross_relations = log_centers[upper_j] - log_centers[upper_i]
    within_weights = np.maximum(0.0, 0.5 * (np.diag(gram) - bin_power_square))
    within_relations = log_widths / 3.0
    weights = np.concatenate((within_weights, cross_weights))
    relations = np.concatenate((within_relations, cross_relations))
    relation_edges = np.linspace(0.0, math.log(float(positive_bins)), config.relation_bins + 1)
    indexes = np.searchsorted(relation_edges, relations, side='right') - 1
    indexes = np.clip(indexes, 0, config.relation_bins - 1)
    histogram = np.bincount(indexes, weights=weights, minlength=config.relation_bins).astype(np.float64, copy=False)
    total_pair_weight = float(np.sum(histogram))
    if not math.isfinite(total_pair_weight) or total_pair_weight <= 0:
        raise ValueError('scale has no finite positive cross-frequency pair mass')
    histogram /= total_pair_weight
    detail = {'frame_count': frame_count, 'positive_frequency_bins': positive_bins, 'effective_log_frequency_groups': group_count, 'total_pair_weight': total_pair_weight}
    return (relation_edges, histogram, detail)

def compute_music_pairs(path: Path, config: RelationConfig, token: CancellationToken) -> dict:
    config.validate()
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    if sample_rate != info.samplerate:
        raise RuntimeError('decoder sample-rate metadata changed while reading')
    scale_results: list[dict] = []
    for scale_index, duration in enumerate(config.scales_seconds):
        token.checkpoint()
        window_length = _positive_half_up(duration * sample_rate)
        hop_length = _positive_half_up(window_length * config.hop_fraction)
        channel_results: list[dict] = []
        relation_edges: np.ndarray | None = None
        for channel_index in range(info.channels):
            edges, histogram, detail = _channel_histogram(audio[:, channel_index], sample_rate, window_length, hop_length, config, token)
            if relation_edges is None:
                relation_edges = edges
            channel_results.append({'source_channel_index': channel_index, 'probabilities': histogram.tolist(), **detail})
        assert relation_edges is not None
        scale_results.append({'scale_index': scale_index, 'duration_seconds': duration, 'window_length': window_length, 'hop_length': hop_length, 'relation_edges': relation_edges.tolist(), 'source_channels': channel_results})
    token.checkpoint()
    return {'schema_version': 2, 'protocol_version': config.protocol_version, 'representation_version': config.pair_grid_version, 'approximation': {'kind': 'log-frequency group location compression', 'primary_inference_eligible': False, 'pair_mass_preserved': True, 'note': 'All positive FFT-bin powers enter the pair mass. Pair locations are approximated within fixed log-frequency groups.'}, 'config': config.to_dict(), 'audio': {'sample_rate': int(sample_rate), 'channels': int(info.channels), 'frames': int(info.frames), 'duration_seconds': float(info.duration)}, 'scales': scale_results}

def _atom_wasserstein(left_x: np.ndarray, left_p: np.ndarray, right_x: np.ndarray, right_p: np.ndarray) -> float:
    support = np.unique(np.concatenate((left_x, right_x)))
    left_mass = np.zeros(support.size, dtype=np.float64)
    right_mass = np.zeros(support.size, dtype=np.float64)
    np.add.at(left_mass, np.searchsorted(support, left_x), left_p)
    np.add.at(right_mass, np.searchsorted(support, right_x), right_p)
    cdf_difference = np.cumsum(left_mass - right_mass)
    return float(np.sum(np.abs(cdf_difference[:-1]) * np.diff(support)))

def _grid_affinity(left_x: np.ndarray, left_p: np.ndarray, right_x: np.ndarray, right_p: np.ndarray, bins: int) -> float:
    upper = float(max(np.max(left_x), np.max(right_x)))
    edges = np.linspace(0.0, upper * (1.0 + 1e-12), bins + 1)
    left = np.histogram(left_x, bins=edges, weights=left_p)[0]
    right = np.histogram(right_x, bins=edges, weights=right_p)[0]
    left = left / np.sum(left)
    right = right / np.sum(right)
    return float(np.sum(np.sqrt(left * right)))

def compare_relations(left: dict, right: dict) -> dict:
    if len(left['scales']) != 8 or len(right['scales']) != 8:
        raise ValueError('both representations must contain all eight scales')
    scale_details: list[dict] = []
    for left_scale, right_scale in zip(left['scales'], right['scales'], strict=True):
        if left_scale['scale_index'] != right_scale['scale_index']:
            raise ValueError('scale indexes do not align')
        left_edges = np.asarray(left_scale['relation_edges'], dtype=np.float64)
        right_edges = np.asarray(right_scale['relation_edges'], dtype=np.float64)
        left_x = 0.5 * (left_edges[:-1] + left_edges[1:])
        right_x = 0.5 * (right_edges[:-1] + right_edges[1:])
        support = float(max(left_edges[-1], right_edges[-1]))
        channel_comparisons: list[dict] = []
        for left_channel in left_scale['source_channels']:
            left_p = np.asarray(left_channel['probabilities'], dtype=np.float64)
            for right_channel in right_scale['source_channels']:
                right_p = np.asarray(right_channel['probabilities'], dtype=np.float64)
                distance = _atom_wasserstein(left_x, left_p, right_x, right_p)
                affinity = _grid_affinity(left_x, left_p, right_x, right_p, max(left_p.size, right_p.size))
                channel_comparisons.append({'left_source_channel_index': int(left_channel['source_channel_index']), 'right_source_channel_index': int(right_channel['source_channel_index']), 'wasserstein': distance, 'normalized_wasserstein': distance / support, 'hellinger_affinity': affinity})
        scale_details.append({'scale_index': int(left_scale['scale_index']), 'duration_seconds': float(left_scale['duration_seconds']), 'channel_comparisons': channel_comparisons, 'mean_wasserstein': float(np.mean([item['wasserstein'] for item in channel_comparisons])), 'mean_normalized_wasserstein': float(np.mean([item['normalized_wasserstein'] for item in channel_comparisons])), 'mean_hellinger_affinity': float(np.mean([item['hellinger_affinity'] for item in channel_comparisons]))})
    return {'metric_version': 'exploratory-source-channel-cartesian-v2', 'mean_wasserstein': float(np.mean([item['mean_wasserstein'] for item in scale_details])), 'mean_normalized_wasserstein': float(np.mean([item['mean_normalized_wasserstein'] for item in scale_details])), 'mean_hellinger_affinity': float(np.mean([item['mean_hellinger_affinity'] for item in scale_details])), 'scales': scale_details}
