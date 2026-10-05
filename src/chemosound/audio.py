from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from .model import CancellationToken
from .rights import load_rights
LOSSLESS_FORMATS = {'FLAC', 'WAV', 'AIFF'}
INTEGER_PCM_SUBTYPES = {'PCM_S8', 'PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32'}

@dataclass(frozen=True, slots=True)
class ValidationResult:
    source_sha256: str
    decoded_sha256: str
    format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    title: str | None
    artist: str | None
    rights_status: str
    rights_sha256: str | None
    rights_json: dict | None
    rights_error: str | None

    def to_json_dict(self) -> dict:
        return asdict(self)

def _hash_file(path: Path, token: CancellationToken, chunk_size: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(chunk_size)):
            token.checkpoint()
            digest.update(chunk)
    token.checkpoint()
    return digest.hexdigest()

def validate_audio(path: Path, token: CancellationToken) -> ValidationResult:
    token.checkpoint()
    initial_stat = path.stat()
    source_digest = _hash_file(path, token)
    info = sf.info(path)
    if info.format not in LOSSLESS_FORMATS:
        raise ValueError(f'unsupported or non-lossless container: {info.format}')
    if info.subtype not in INTEGER_PCM_SUBTYPES:
        raise ValueError(f'unsupported PCM subtype for canonical hashing: {info.subtype}')
    if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
        raise ValueError('audio metadata has a non-positive dimension')
    decoded_digest = hashlib.sha256()
    header = f'chemosound-canonical-pcm-v1\x00{info.samplerate}\x00{info.channels}\x00{info.frames}\x00'.encode('ascii')
    decoded_digest.update(header)
    decoded_frames = 0
    with sf.SoundFile(path, mode='r') as audio:
        while True:
            token.checkpoint()
            block = audio.read(262144, dtype='int32', always_2d=True)
            if block.size == 0:
                break
            canonical = np.ascontiguousarray(block, dtype='<i4')
            decoded_digest.update(canonical.tobytes(order='C'))
            decoded_frames += int(block.shape[0])
    token.checkpoint()
    final_stat = path.stat()
    if decoded_frames != info.frames:
        raise ValueError('decoded frame count differs from audio metadata')
    if initial_stat.st_size != final_stat.st_size or initial_stat.st_mtime_ns != final_stat.st_mtime_ns or _hash_file(path, token) != source_digest:
        raise RuntimeError('audio changed during validation; rescan before validation')
    rights = load_rights(path)
    title = None
    artist = None
    if rights.document is not None:
        raw_title = rights.document.get('title')
        raw_artist = rights.document.get('artist')
        title = raw_title.strip() if isinstance(raw_title, str) else None
        artist = raw_artist.strip() if isinstance(raw_artist, str) else None
    return ValidationResult(source_sha256=source_digest, decoded_sha256=decoded_digest.hexdigest(), format=info.format, subtype=info.subtype, sample_rate=int(info.samplerate), channels=int(info.channels), frames=int(info.frames), duration_seconds=float(info.duration), title=title, artist=artist, rights_status=rights.status, rights_sha256=rights.sha256, rights_json=rights.document, rights_error=rights.error)
