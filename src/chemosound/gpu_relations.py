from __future__ import annotations
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from .model import CancellationToken

@dataclass(frozen=True, slots=True)
class GpuRelationConfig:
    protocol_version: str = '1.0'
    pair_grid_version: str = 'music-relational-certified-grid-gpu-v1'
    scales_seconds: tuple[float, ...] = tuple((2 ** s / 64 for s in range(8)))
    coordinate_quantum: float = 0.005
    stft_frame_batch: int = 128
    surrogate_batch: int = 8
    rng: str = 'numpy-philox-sha256-per-replicate-v1'

    def validate(self) -> None:
        if self.scales_seconds != tuple((2 ** s / 64 for s in range(8))):
            raise ValueError('protocol 1.0 requires exactly eight scales')
        if self.coordinate_quantum <= 0 or not math.isfinite(self.coordinate_quantum):
            raise ValueError('coordinate_quantum must be finite and positive')
        if self.stft_frame_batch < 1 or self.surrogate_batch < 1:
            raise ValueError('GPU batch sizes must be positive')

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

@dataclass(slots=True)
class MolecularGrid:
    quantum: float
    cdf: np.ndarray
    molecule_ids: np.ndarray
    formulas: list[str]
    error_bounds: np.ndarray
    manifest_sha256: str
    pair_grid_version: str

def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('GPU analysis requires the project gpu extra: uv sync --extra gpu') from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('GPU analysis requires a CUDA-capable NVIDIA device')
    return cp

def gpu_inventory() -> dict:
    cp = _cupy()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name']
    if isinstance(name, bytes):
        name = name.decode()
    free, total = cp.cuda.runtime.memGetInfo()
    return {'backend': 'cupy-cuda', 'cupy_version': cp.__version__, 'device': name, 'compute_capability': cp.cuda.Device(0).compute_capability, 'total_vram_bytes': int(total), 'free_vram_bytes': int(free)}

def load_molecular_grid(hitran_root: Path) -> MolecularGrid:
    root = hitran_root.resolve()
    manifest_path = root / 'manifests' / 'snapshot-frozen.json'
    if not manifest_path.is_file():
        raise RuntimeError('a frozen HITRAN snapshot is required')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('status') != 'frozen' or len(manifest.get('items', [])) != 61:
        raise RuntimeError('HITRAN manifest is not a complete frozen snapshot')
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    representation_root = root / 'representations'
    versions = sorted((path for path in representation_root.iterdir() if path.is_dir()))
    if not versions:
        raise RuntimeError('no molecular representations are available')
    version_root = versions[-1]
    records: list[tuple[int, str, np.ndarray, np.ndarray, dict]] = []
    for item in manifest['items']:
        molecule_id = int(item['molecule_id'])
        path = version_root / f'molecule_{molecule_id:03d}_iso_1.npz'
        if not path.is_file():
            raise RuntimeError(f'missing molecular representation: {path}')
        with np.load(path, allow_pickle=False) as artifact:
            support = artifact['support'].astype(np.float64, copy=True)
            probabilities = artifact['probabilities'].astype(np.float64, copy=True)
            metadata = json.loads(str(artifact['metadata_json']))
        if metadata['source_normalized_sha256'] != item['normalized_sha256']:
            raise RuntimeError(f'molecule {molecule_id} representation source mismatch')
        records.append((molecule_id, str(item['formula']), support, probabilities, metadata))
    quantums = {float(record[4]['coordinate_quantum']) for record in records}
    if len(quantums) != 1:
        raise RuntimeError('molecular representations do not share one grid')
    quantum = quantums.pop()
    maximum_index = max((int(round(float(record[2][-1]) / quantum)) for record in records))
    probability_grid = np.zeros((61, maximum_index + 1), dtype=np.float64)
    error_bounds = np.empty(61, dtype=np.float64)
    ids = np.empty(61, dtype=np.int16)
    formulas: list[str] = []
    for row, (molecule_id, formula, support, probabilities, metadata) in enumerate(records):
        indexes = np.rint(support / quantum).astype(np.int64)
        np.add.at(probability_grid[row], indexes, probabilities)
        ids[row] = molecule_id
        formulas.append(formula)
        error_bounds[row] = float(metadata['wasserstein_error_bound'])
    probability_grid /= np.sum(probability_grid, axis=1, keepdims=True)
    return MolecularGrid(quantum=quantum, cdf=np.cumsum(probability_grid, axis=1), molecule_ids=ids, formulas=formulas, error_bounds=error_bounds, manifest_sha256=manifest_sha256, pair_grid_version=version_root.name)

def _positive_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))

def _group_plan(positive_bins: int, quantum: float) -> dict:
    indexes = np.rint(np.log(np.arange(1, positive_bins + 1, dtype=np.float64)) / quantum).astype(np.int64)
    unique, starts = np.unique(indexes, return_index=True)
    stops = np.r_[starts[1:], positive_bins]
    quantized = indexes.astype(np.float64) * quantum
    error = float(2.0 * np.max(np.abs(np.log(np.arange(1, positive_bins + 1, dtype=np.float64)) - quantized)))
    group_count = int(indexes[-1]) + 1
    fft_length = 1 << (2 * group_count - 2).bit_length()
    return {'indexes': indexes, 'unique': unique, 'starts': starts, 'stops': stops, 'group_count': group_count, 'fft_length': fft_length, 'error_bound': error}

def _group_power(cp, power, plan: dict):
    cumulative = cp.pad(cp.cumsum(power, axis=-1), ((0, 0),) * (power.ndim - 1) + ((1, 0),))
    starts = cp.asarray(plan['starts'])
    stops = cp.asarray(plan['stops'])
    active = cumulative[..., stops] - cumulative[..., starts]
    shape = (*power.shape[:-1], plan['group_count'])
    grouped = cp.zeros(shape, dtype=power.dtype)
    grouped[..., cp.asarray(plan['unique'])] = active
    return grouped

def _power_probabilities(cp, power, plan: dict, frame_batch: int):
    leading = power.shape[:-2]
    frame_count = int(power.shape[-2])
    spectrum_sum = cp.zeros((*leading, plan['fft_length'] // 2 + 1), dtype=cp.float64)
    square_sum = cp.zeros(leading, dtype=cp.float64)
    for offset in range(0, frame_count, frame_batch):
        block = power[..., offset:offset + frame_batch, :]
        grouped = _group_power(cp, block, plan)
        transformed = cp.fft.rfft(grouped, n=plan['fft_length'], axis=-1)
        spectrum_sum += cp.sum(transformed.real * transformed.real + transformed.imag * transformed.imag, axis=-2, dtype=cp.float64)
        square_sum += cp.sum(block * block, axis=(-2, -1), dtype=cp.float64)
    correlation = cp.fft.irfft(spectrum_sum, n=plan['fft_length'], axis=-1)[..., :plan['group_count']]
    mass = cp.maximum(correlation, 0.0)
    mass[..., 0] = cp.maximum((correlation[..., 0] - square_sum) / 2.0, 0.0)
    total = cp.sum(mass, axis=-1, dtype=cp.float64)
    if bool(cp.any(~cp.isfinite(total)).item()) or bool(cp.any(total <= 0).item()):
        raise ValueError('music scale has no finite positive pair mass')
    return mass / total[..., None]

def _stft_power(cp, channel: np.ndarray, window_length: int, hop: int, batch: int):
    device_audio = cp.asarray(channel, dtype=cp.float32)
    frame_count = 1 + (channel.size - window_length) // hop
    positive_bins = window_length // 2
    power = cp.empty((frame_count, positive_bins), dtype=cp.float32)
    window = cp.asarray(np.hanning(window_length), dtype=cp.float32)
    for offset in range(0, frame_count, batch):
        count = min(batch, frame_count - offset)
        starts = (offset + cp.arange(count, dtype=cp.int64)) * hop
        sample_indexes = starts[:, None] + cp.arange(window_length, dtype=cp.int64)
        frames = device_audio[sample_indexes]
        transformed = cp.fft.rfft(frames * window, axis=-1)[..., 1:]
        power[offset:offset + count] = transformed.real * transformed.real + transformed.imag * transformed.imag
    return power

def _replicate_seed(run_seed: str, decoded_sha256: str, scale_index: int, channel_index: int, replicate_index: int) -> int:
    payload = f'{run_seed}|{decoded_sha256}|{scale_index}|{channel_index}|{replicate_index}|philox-v1'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')

def _shift_probabilities(cp, power, plan: dict, replicate_indexes: np.ndarray, *, run_seed: str, decoded_sha256: str, scale_index: int, channel_index: int, frame_batch: int):
    frame_count, positive_bins = map(int, power.shape)
    shifts = np.empty((replicate_indexes.size, positive_bins), dtype=np.int64)
    for row, replicate_index in enumerate(replicate_indexes):
        generator = np.random.Generator(np.random.Philox(_replicate_seed(run_seed, decoded_sha256, scale_index, channel_index, int(replicate_index))))
        shifts[row] = generator.integers(0, frame_count, size=positive_bins, dtype=np.int64)
    device_shifts = cp.asarray(shifts)
    spectrum_sum = cp.zeros((replicate_indexes.size, plan['fft_length'] // 2 + 1), dtype=cp.float64)
    square_sum = cp.sum(power * power, dtype=cp.float64)
    columns = cp.arange(positive_bins, dtype=cp.int64)[None, None, :]
    for offset in range(0, frame_count, frame_batch):
        count = min(frame_batch, frame_count - offset)
        rows = (cp.arange(offset, offset + count, dtype=cp.int64)[None, :, None] + device_shifts[:, None, :]) % frame_count
        shifted = power[rows, columns]
        grouped = _group_power(cp, shifted, plan)
        transformed = cp.fft.rfft(grouped, n=plan['fft_length'], axis=-1)
        spectrum_sum += cp.sum(transformed.real * transformed.real + transformed.imag * transformed.imag, axis=1, dtype=cp.float64)
    correlation = cp.fft.irfft(spectrum_sum, n=plan['fft_length'], axis=-1)[:, :plan['group_count']]
    mass = cp.maximum(correlation, 0.0)
    mass[:, 0] = cp.maximum((correlation[:, 0] - square_sum) / 2.0, 0.0)
    return mass / cp.sum(mass, axis=1, dtype=cp.float64)[:, None]

def _molecule_distances(cp, music_probability, music_quantum: float, molecules: MolecularGrid):
    ratio = music_quantum / molecules.quantum
    ratio_integer = int(round(ratio))
    if abs(ratio - ratio_integer) > 1e-10:
        raise ValueError('music grid must be an integer multiple of molecular grid')
    leading = music_probability.shape[:-1]
    music_length = int(music_probability.shape[-1])
    total_length = max(molecules.cdf.shape[1], (music_length - 1) * ratio_integer + 1)
    fine = cp.zeros((*leading, total_length), dtype=cp.float64)
    fine[..., cp.arange(music_length) * ratio_integer] = music_probability
    music_cdf = cp.cumsum(fine, axis=-1)
    molecule_cdf = cp.zeros((61, total_length), dtype=cp.float64)
    molecule_cdf[:, :molecules.cdf.shape[1]] = cp.asarray(molecules.cdf)
    if molecules.cdf.shape[1] < total_length:
        molecule_cdf[:, molecules.cdf.shape[1]:] = 1.0
    result = cp.empty((*leading, 61), dtype=cp.float64)
    flat = music_cdf.reshape((-1, total_length))
    output = result.reshape((-1, 61))
    for molecule_start in range(0, 61, 8):
        molecule_stop = min(61, molecule_start + 8)
        output[:, molecule_start:molecule_stop] = cp.sum(cp.abs(flat[:, None, :] - molecule_cdf[None, molecule_start:molecule_stop, :]), axis=-1, dtype=cp.float64) * molecules.quantum
    return result

def analyze_song_gpu(path: Path, decoded_sha256: str, molecules: MolecularGrid, config: GpuRelationConfig, token: CancellationToken, *, surrogate_replicates: int, run_seed: str) -> dict:
    raise RuntimeError('This legacy channel-averaging path is superseded by tensor_engine.ScaleEngine and chemosound.flow; amendments 01-03 require labeled channel tensors.')
    config.validate()
    cp = _cupy()
    audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] not in (1, 2):
        raise ValueError('protocol 1.0 accepts only mono or stereo')
    real_distance = np.zeros(61, dtype=np.float64)
    surrogate_distance = np.zeros((surrogate_replicates, 61), dtype=np.float64)
    scale_details: list[dict] = []
    started = time.perf_counter()
    for scale_index, duration in enumerate(config.scales_seconds):
        token.checkpoint()
        scale_started = time.perf_counter()
        window_length = _positive_half_up(duration * sample_rate)
        hop = _positive_half_up(window_length / 4)
        if audio.shape[0] < window_length:
            raise ValueError(f'song is shorter than scale {duration:g}s')
        plan = _group_plan(window_length // 2, config.coordinate_quantum)
        channel_powers = [_stft_power(cp, audio[:, channel], window_length, hop, config.stft_frame_batch) for channel in range(audio.shape[1])]
        real_probability = sum((_power_probabilities(cp, power, plan, config.stft_frame_batch) for power in channel_powers)) / audio.shape[1]
        real_distance += cp.asnumpy(_molecule_distances(cp, real_probability, config.coordinate_quantum, molecules)) / 8.0
        for replicate_start in range(0, surrogate_replicates, config.surrogate_batch):
            token.checkpoint()
            replicate_indexes = np.arange(replicate_start + 1, min(surrogate_replicates, replicate_start + config.surrogate_batch) + 1, dtype=np.int64)
            probability = sum((_shift_probabilities(cp, power, plan, replicate_indexes, run_seed=run_seed, decoded_sha256=decoded_sha256, scale_index=scale_index, channel_index=channel, frame_batch=config.stft_frame_batch) for channel, power in enumerate(channel_powers))) / audio.shape[1]
            surrogate_distance[replicate_start:replicate_start + replicate_indexes.size] += cp.asnumpy(_molecule_distances(cp, probability, config.coordinate_quantum, molecules)) / 8.0
        cp.cuda.Stream.null.synchronize()
        scale_details.append({'scale_index': scale_index, 'duration_seconds': duration, 'window_length': window_length, 'hop_length': hop, 'frame_count': int(channel_powers[0].shape[0]), 'positive_frequency_bins': int(channel_powers[0].shape[1]), 'relation_grid_cells': plan['group_count'], 'relation_w1_error_bound': plan['error_bound'], 'wall_seconds': time.perf_counter() - scale_started})
        del channel_powers, real_probability
        cp.get_default_memory_pool().free_all_blocks()
    return {'real_distance': real_distance, 'surrogate_distance': surrogate_distance, 'scale_details': scale_details, 'wall_seconds': time.perf_counter() - started, 'music_w1_error_bound': max((detail['relation_w1_error_bound'] for detail in scale_details))}
