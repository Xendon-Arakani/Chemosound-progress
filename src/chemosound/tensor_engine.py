from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from .cpu_oracle import philox_shifts
from .field_store import FieldWriter, field_blocks, file_hash, read_fields
from .gpu_relations import MolecularGrid, _cupy, _group_plan
from .model import CancellationToken

@dataclass(frozen=True)
class TensorConfig:
    version: str = 'channel-tensor-grid-f64-v2'
    scales_seconds: tuple[float, ...] = tuple((2 ** s / 64 for s in range(8)))
    coordinate_quantum: float = 0.005
    window: str = 'symmetric-hann-full-frames-positive-non-dc'
    shift_law: str = 'independent-bin-channel-scale-circular-philox-v1'
    complex_storage: str = 'full-rfft-c128-chunks-v1'

    def validate(self):
        if self.scales_seconds != tuple((2 ** s / 64 for s in range(8))):
            raise ValueError('exactly the eight protocol scales are required')
        if not math.isfinite(self.coordinate_quantum) or self.coordinate_quantum <= 0:
            raise ValueError('invalid coordinate quantum')
        if self.version != TensorConfig().version or self.window != TensorConfig().window or self.shift_law != TensorConfig().shift_law or self.complex_storage != TensorConfig().complex_storage:
            raise ValueError('unknown tensor operator')

    def document(self):
        return asdict(self)

def measure_labels(channels: int) -> list[dict]:
    return [{'kind': 'within', 'channels': [a]} for a in range(channels)] + [{'kind': 'cross', 'channels': [a, b]} for a in range(channels) for b in range(a + 1, channels)]

def grouped_power(xp, power, plan):
    sums = xp.cumsum(power, axis=-1, dtype=xp.float64)
    ends = sums[:, plan['stops'] - 1]
    starts = plan['starts']
    prior = xp.zeros_like(ends)
    nonzero = np.flatnonzero(starts)
    prior[:, nonzero] = sums[:, starts[nonzero] - 1]
    result = xp.zeros((power.shape[0], plan['group_count']), dtype=xp.float64)
    result[:, plan['unique']] = ends - prior
    return result

def tensor_probabilities(powers, plan, token, *, shifts=None, frame_batch=64, xp=np):
    channels = len(powers)
    labels = measure_labels(channels)
    frames, bins = powers[0].shape
    if any((p.shape != (frames, bins) for p in powers)) or bins < 2:
        raise ValueError('unaligned power geometry')
    spectra = [xp.zeros(plan['fft_length'] // 2 + 1, dtype=xp.complex128) for _ in labels]
    squares = xp.zeros(channels, dtype=xp.float64)
    expected = xp.zeros(len(labels), dtype=xp.float64)
    for start in range(0, frames, frame_batch):
        token.checkpoint()
        stop = min(frames, start + frame_batch)
        transformed, totals, squared = ([], [], [])
        for channel, power in enumerate(powers):
            if shifts is None:
                block = power[start:stop]
            else:
                row = (xp.arange(start, stop)[:, None] + xp.asarray(shifts[channel])[None, :]) % frames
                block = power[row, xp.arange(bins)[None, :]]
            totals.append(xp.sum(block, axis=1, dtype=xp.float64))
            squared.append(xp.sum(block * block, axis=1, dtype=xp.float64))
            transformed.append(xp.fft.rfft(grouped_power(xp, block, plan), n=plan['fft_length'], axis=1))
        for channel in range(channels):
            z = transformed[channel]
            spectra[channel] += xp.sum(z * xp.conj(z), axis=0)
            squares[channel] += xp.sum(squared[channel])
            expected[channel] += xp.sum((totals[channel] ** 2 - squared[channel]) / 2)
        for index, label in enumerate(labels[channels:], start=channels):
            a, b = label['channels']
            spectra[index] += xp.sum(transformed[a] * xp.conj(transformed[b]), axis=0)
            expected[index] += xp.sum(totals[a] * totals[b])
    masses = []
    for index, spectrum in enumerate(spectra):
        token.checkpoint()
        corr = xp.fft.irfft(spectrum, n=plan['fft_length'])
        mass = corr[:plan['group_count']].copy()
        if index < channels:
            mass[0] = (mass[0] - squares[index]) / 2
        else:
            mass[1:] += corr[-(plan['group_count'] - 1):][::-1]
        mass = xp.maximum(mass, 0)
        total = float(xp.sum(mass))
        target = float(expected[index])
        if not math.isfinite(total) or total <= 0 or target <= 0:
            raise ValueError('a labeled measure has no positive pair mass')
        if abs(total - target) > 2e-09 * target:
            raise ArithmeticError('pair-mass identity failed')
        masses.append(mass / total)
    return (xp.stack(masses), labels)

def tensor_distances(xp, probability, quantum, molecules, token):
    ratio = quantum / molecules.quantum
    stride = int(round(ratio))
    if stride < 1 or abs(stride - ratio) > 1e-10:
        raise ValueError('music grid must be an integer multiple of molecular grid')
    length = max(molecules.cdf.shape[1], (probability.shape[1] - 1) * stride + 1)
    fine = xp.zeros((probability.shape[0], length), dtype=xp.float64)
    fine[:, xp.arange(probability.shape[1]) * stride] = probability
    cdf = xp.cumsum(fine, axis=1)
    result = xp.empty((probability.shape[0], len(molecules.molecule_ids)), dtype=xp.float64)
    for start in range(0, len(molecules.molecule_ids), 4):
        token.checkpoint()
        stop = min(start + 4, len(molecules.molecule_ids))
        mcdf = xp.ones((stop - start, length), dtype=xp.float64)
        mcdf[:, :molecules.cdf.shape[1]] = xp.asarray(molecules.cdf[start:stop])
        result[:, start:stop] = xp.sum(xp.abs(cdf[:, None, :-1] - mcdf[None, :, :-1]), axis=-1) * molecules.quantum
    return result

class ScaleEngine:

    def __init__(self, path: Path, source_pcm_hash: str, scale: int, config: TensorConfig, token: CancellationToken, *, frame_batch=64, backend='gpu', field_root=None):
        config.validate()
        self.xp = _cupy() if backend == 'gpu' else np
        xp = self.xp
        self.token, self.config, self.scale = (token, config, scale)
        self.source_pcm_hash, self.frame_batch = (source_pcm_hash, frame_batch)
        info = sf.info(path)
        self.window_length = int(math.floor(config.scales_seconds[scale] * info.samplerate + 0.5))
        self.hop = int(math.floor(self.window_length / 4 + 0.5))
        self.frames = 1 + (info.frames - self.window_length) // self.hop
        bins = self.window_length // 2
        if self.frames < 1 or bins < 2:
            raise ValueError('source shorter than protocol scale')
        self.plan = _group_plan(bins, config.coordinate_quantum)
        self.power_bytes = info.channels * self.frames * bins * 8
        if backend == 'gpu':
            free, _ = xp.cuda.runtime.memGetInfo()
            working = info.channels * frame_batch * bins * 8 * 10 + 768 * 1024 ** 2
            if self.power_bytes + working > free * 0.85:
                raise MemoryError('scale exceeds conservative device-memory budget')
        self.powers = [xp.empty((self.frames, bins), dtype=xp.float64) for _ in range(info.channels)]
        window = xp.asarray(np.hanning(self.window_length), dtype=xp.float64)
        self.complex_hashes = [hashlib.sha256() for _ in range(info.channels)]
        self.cross_sums = {(a, b): xp.zeros(bins, dtype=xp.complex128) for a in range(info.channels) for b in range(a + 1, info.channels)}
        self.field_root = Path(field_root) if field_root is not None else None
        self.field_identity = {'source_decoded_sha256': source_pcm_hash, 'operator': config.document(), 'scale': scale, 'frames': self.frames, 'bins': bins + 1, 'channels': info.channels, 'window_length': self.window_length, 'hop': self.hop}
        self.field_hash = None
        stored = read_fields(self.field_root, self.field_identity, token) if self.field_root is not None else None
        if stored is not None:
            for offset, arrays in field_blocks(self.field_root, stored, token):
                count = arrays['X_0'].shape[0]
                for channel in range(info.channels):
                    host = arrays['X_' + str(channel)][:, 1:]
                    if host.shape != (count, bins) or not np.all(np.isfinite(host)):
                        raise RuntimeError('FIELD_ARRAY_INVALID')
                    self.complex_hashes[channel].update(np.ascontiguousarray(host, dtype='<c16').tobytes())
                    transformed = xp.asarray(host)
                    self.powers[channel][offset:offset + count] = transformed.real ** 2 + transformed.imag ** 2
                for (a, b), total in self.cross_sums.items():
                    values = arrays[f'Q_{a}_{b}'][:, 1:]
                    if values.shape != (count, bins) or not np.all(np.isfinite(values)):
                        raise RuntimeError('FIELD_ARRAY_INVALID')
                    total += xp.sum(xp.asarray(values), axis=0)
            self.field_hash = file_hash(self.field_root / 'fields.json')
            token.checkpoint()
            return
        writer = FieldWriter(self.field_root, self.field_identity, token) if self.field_root is not None else None
        try:
            self._decode(path, info, window, writer)
            if writer is not None:
                writer.commit()
                self.field_hash = file_hash(self.field_root / 'fields.json')
        finally:
            if writer is not None:
                writer.close()
        token.checkpoint()

    def _decode(self, path, info, window, writer):
        xp, token = self.xp, self.token
        bins = self.powers[0].shape[1]
        frame_batch = self.frame_batch
        with sf.SoundFile(path) as source:
            for offset in range(0, self.frames, frame_batch):
                token.checkpoint()
                count = min(frame_batch, self.frames - offset)
                source.seek(offset * self.hop)
                block = source.read((count - 1) * self.hop + self.window_length, dtype='float64', always_2d=True)
                if block.shape[0] != (count - 1) * self.hop + self.window_length:
                    raise IOError('short native decode')
                channel_spectra = []
                full_spectra = []
                for channel in range(info.channels):
                    token.checkpoint()
                    samples = xp.asarray(block[:, channel])
                    indexes = xp.arange(count)[:, None] * self.hop + xp.arange(self.window_length)[None, :]
                    full = xp.fft.rfft(samples[indexes] * window, axis=1)
                    transformed = full[:, 1:]
                    host = xp.asnumpy(transformed) if xp is not np else transformed
                    self.complex_hashes[channel].update(np.ascontiguousarray(host, dtype='<c16').tobytes())
                    self.powers[channel][offset:offset + count] = transformed.real ** 2 + transformed.imag ** 2
                    channel_spectra.append(transformed)
                    if writer is not None:
                        full_spectra.append(xp.asnumpy(full) if xp is not np else full)
                for (a, b), total in self.cross_sums.items():
                    total += xp.sum(channel_spectra[a] * xp.conj(channel_spectra[b]), axis=0)
                if writer is not None:
                    writer.append(offset, full_spectra)

    def compute(self, molecules: MolecularGrid, replicate_indexes: list[int], run_seed: str):
        xp, token = (self.xp, self.token)
        outputs, real_probability = ([], None)
        for replicate in replicate_indexes:
            token.checkpoint()
            shifts = None if replicate == 0 else [philox_shifts(self.frames, p.shape[1], run_seed=run_seed, decoded_sha256=self.source_pcm_hash, scale_index=self.scale, channel_index=a, replicate_index=replicate) for a, p in enumerate(self.powers)]
            probability, labels = tensor_probabilities(self.powers, self.plan, token, shifts=shifts, frame_batch=self.frame_batch, xp=xp)
            distances = tensor_distances(xp, probability, self.config.coordinate_quantum, molecules, token)
            host = xp.asnumpy(distances) if xp is not np else distances
            if not np.all(np.isfinite(host)) or np.any(host < 0):
                raise ArithmeticError('nonfinite or negative distances')
            outputs.append(host)
            if replicate == 0:
                real_probability = xp.asnumpy(probability) if xp is not np else probability
        result = {'distances': np.stack(outputs), 'replicate_indexes': np.asarray(replicate_indexes, dtype=np.int32), 'error_radius': self.plan['error_bound'] + molecules.error_bounds, 'measure_labels_json': np.asarray(json.dumps(labels, sort_keys=True)), 'operator_json': np.asarray(json.dumps(self.config.document(), sort_keys=True)), 'complex_stft_sha256': np.asarray([h.hexdigest() for h in self.complex_hashes]), 'geometry': np.asarray([self.scale, self.window_length, self.hop, self.frames, len(self.powers), self.powers[0].shape[1]])}
        if self.field_hash is not None:
            result['complex_fields_sha256'] = np.asarray(self.field_hash)
        if real_probability is not None:
            result['real_probability'] = real_probability
            for (a, b), values in self.cross_sums.items():
                result[f'cross_spectral_sum_{a}_{b}'] = xp.asnumpy(values) if xp is not np else values
        return result

    def close(self):
        self.powers.clear()
        self.cross_sums.clear()
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()
