from __future__ import annotations
import hashlib
import json
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parents[2]
POLICY = Path(__file__).with_name('policy-v1.json')
ITEM = 'In_Vitro_-_La_Danza_de_los_Solitarios_FLAC'

def file_hash(path: Path, algorithm: str='sha256') -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()

def streaminfo(path: Path) -> dict:
    with path.open('rb') as stream:
        if stream.read(4) != b'fLaC':
            raise ValueError('Expected a native FLAC stream')
        header = stream.read(4)
        if len(header) != 4 or header[0] & 127 != 0:
            raise ValueError('Missing initial STREAMINFO')
        if int.from_bytes(header[1:], 'big') != 34:
            raise ValueError('Invalid STREAMINFO length')
        payload = stream.read(34)
        if len(payload) != 34:
            raise ValueError('Truncated STREAMINFO')
    geometry = int.from_bytes(payload[10:18], 'big')
    return {'sample_rate': geometry >> 44, 'channels': (geometry >> 41 & 7) + 1, 'bits_per_sample': (geometry >> 36 & 31) + 1, 'frames': geometry & (1 << 36) - 1, 'pcm_md5': payload[18:].hex()}

def pcm_audit(path: Path, *, block_frames: int=262144) -> dict:
    info = sf.info(path)
    if info.format not in {'FLAC', 'WAV', 'AIFF'}:
        raise ValueError('Unsupported lossless container')
    subtypes = {'PCM_S8': 8, 'PCM_U8': 8, 'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32}
    if info.subtype not in subtypes:
        raise ValueError('Unsupported integer PCM subtype')
    if min(info.samplerate, info.channels, info.frames) <= 0:
        raise ValueError('Nonpositive sample geometry')
    canonical = hashlib.sha256()
    canonical.update(f'chemosound-canonical-pcm-v1\x00{info.samplerate}\x00{info.channels}\x00{info.frames}\x00'.encode('ascii'))
    native_md5 = hashlib.md5()
    frames_read = 0
    with sf.SoundFile(path) as audio:
        while True:
            block = audio.read(block_frames, dtype='int32', always_2d=True)
            if not block.size:
                break
            canonical.update(np.ascontiguousarray(block, dtype='<i4').tobytes())
            bits = subtypes[info.subtype]
            integers = (block.astype(np.int64) >> 32 - bits).reshape(-1)
            packed = np.empty((integers.size, bits // 8), dtype=np.uint8)
            for byte in range(bits // 8):
                packed[:, byte] = integers >> 8 * byte & 255
            native_md5.update(packed.tobytes())
            frames_read += len(block)
    if frames_read != info.frames:
        raise ValueError('Actual decoded frame count differs from metadata')
    result = {'decoded_sha256': canonical.hexdigest(), 'sample_rate': info.samplerate, 'channels': info.channels, 'frames': info.frames, 'bits_per_sample': subtypes[info.subtype], 'native_signed_pcm_md5': native_md5.hexdigest()}
    if info.format == 'FLAC':
        embedded = streaminfo(path)
        for field in ('sample_rate', 'channels', 'frames', 'bits_per_sample'):
            if embedded[field] != result[field]:
                raise ValueError('STREAMINFO/sample geometry disagreement: ' + field)
        if embedded['pcm_md5'] == '0' * 32:
            raise ValueError('FLAC has no embedded PCM checksum')
        if embedded['pcm_md5'] != result['native_signed_pcm_md5']:
            raise ValueError('Embedded FLAC PCM checksum mismatch')
        result['embedded_pcm_md5_verified'] = True
    return result

def exact_sample_compare(left: Path, right: Path) -> bool:
    li, ri = (sf.info(left), sf.info(right))
    if (li.samplerate, li.channels, li.frames) != (ri.samplerate, ri.channels, ri.frames):
        return False
    with sf.SoundFile(left) as la, sf.SoundFile(right) as ra:
        while True:
            lb = la.read(262144, dtype='int32', always_2d=True)
            rb = ra.read(262144, dtype='int32', always_2d=True)
            if not np.array_equal(lb, rb):
                return False
            if not lb.size:
                return True

def reference_pcm_equal(path: Path, flac: Path) -> bool:
    info = streaminfo(path)
    bits = info['bits_per_sample']
    if bits not in (16, 24):
        raise ValueError('Reference comparison supports verified FLAC 16/24-bit PCM')
    command = [str(flac), '--decode', '--stdout', '--force-raw-format', '--endian=little', '--sign=signed', '--silent', str(path)]
    import tempfile
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            with sf.SoundFile(path) as audio:
                while True:
                    block = audio.read(262144, dtype='int32', always_2d=True)
                    if not block.size:
                        break
                    integers = (block.astype(np.int64) >> 32 - bits).reshape(-1)
                    packed = np.empty((integers.size, bits // 8), dtype=np.uint8)
                    for byte in range(bits // 8):
                        packed[:, byte] = integers >> 8 * byte & 255
                    expected = packed.tobytes()
                    actual = bytearray()
                    while len(actual) < len(expected):
                        chunk = process.stdout.read(len(expected) - len(actual))
                        if not chunk:
                            return False
                        actual.extend(chunk)
                    if actual != expected:
                        return False
                if process.stdout.read(1):
                    return False
            code = process.wait(timeout=30)
            errors.seek(0)
            return code == 0 and (not errors.read())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            process.stdout.close()

def admission_decision(gates: dict, policy: dict) -> dict:
    missing = [name for name in policy['required_gates'] if gates.get(name) is not True]
    missing.append('production_admission_not_implemented')
    return {'primary_eligible': False, 'blocking_gates': missing}

def main() -> int:
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    output_root = ROOT / '.chemosound/acquisition-research'
    output_root.mkdir(parents=True, exist_ok=True)
    url = 'https://archive.org/metadata/' + ITEM
    request = urllib.request.Request(url, headers={'User-Agent': policy['http']['user_agent']})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(policy['http']['metadata_limit_bytes'] + 1)
    if len(raw) > policy['http']['metadata_limit_bytes']:
        raise ValueError('Metadata exceeded size limit')
    document = json.loads(raw)
    evidence_sha = hashlib.sha256(raw).hexdigest()
    (output_root / f'{evidence_sha}.json').write_bytes(raw)
    files = {item['name']: item for item in document['files']}
    flac = ROOT / '.chemosound/tools/flac-1.5.0' / policy['reference_flac']['executable_relative_path']
    package = flac.parents[2] / 'flac-1.5.0-win.zip'
    if file_hash(package) != policy['reference_flac']['package_sha256']:
        raise ValueError('Reference decoder package checksum mismatch')
    import zipfile
    with zipfile.ZipFile(package) as archive:
        packaged = archive.read(policy['reference_flac']['executable_relative_path'])
    if flac.read_bytes() != packaged:
        raise ValueError('Reference executable differs from verified package')
    decoder_version = subprocess.run([str(flac), '--version'], capture_output=True, text=True, check=True).stdout.strip()
    connection = sqlite3.connect(f"{(ROOT / '.chemosound/state.sqlite3').as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    reports = []
    for row in connection.execute('SELECT * FROM songs ORDER BY id'):
        path = Path(row['absolute_path'])
        item = files.get(path.name)
        if item is None:
            raise ValueError('Catalog file is absent from source metadata')
        decoded = pcm_audit(path)
        source_sha = file_hash(path)
        source_md5 = file_hash(path, 'md5')
        flac_test = subprocess.run([str(flac), '--test', '--warnings-as-errors', '--silent', str(path)], capture_output=True, text=True)
        source_match = source_md5 == item.get('md5') and path.stat().st_size == int(item['size'])
        pcm_match = decoded['decoded_sha256'] == row['decoded_sha256']
        reference_equal = reference_pcm_equal(path, flac)
        gates = {'source_bytes_match': source_match and source_sha == row['source_sha256'], 'decoded_pcm_match': pcm_match and flac_test.returncode == 0 and reference_equal, 'ordered_channels_preserved': pcm_match}
        reports.append({'song_id': row['id'], 'file': path.name, 'archive_source_label': item.get('source'), 'source_sha256': source_sha, 'source_md5': source_md5, 'archive_checksum_match': source_match, 'database_source_sha256_match': source_sha == row['source_sha256'], 'database_decoded_sha256_match': pcm_match, 'decoder_audit': decoded, 'reference_flac_test_exit': flac_test.returncode, 'reference_native_pcm_byte_equal': reference_equal, 'reference_flac_stderr': flac_test.stderr, 'decision': admission_decision(gates, policy)})
    connection.close()
    report = {'audit_kind': 'design-validation-not-primary-certification', 'captured_at_utc': datetime.now(UTC).isoformat(), 'policy_sha256': file_hash(POLICY), 'verifier_sha256': file_hash(Path(__file__)), 'metadata_url': url, 'metadata_sha256': evidence_sha, 'archive_license_claim': document['metadata'].get('licenseurl'), 'reference_decoder': decoder_version, 'runtime': {'python': sys.version, 'numpy': np.__version__, 'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__}, 'songs': reports, 'strict_primary_eligible_count': sum((item['decision']['primary_eligible'] for item in reports))}
    output = output_root / 'existing-song-audit.json'
    history = output_root / 'reports'
    history.mkdir(exist_ok=True)
    if output.exists():
        previous = output.read_bytes()
        (history / f'{hashlib.sha256(previous).hexdigest()}.json').write_bytes(previous)
    body = (json.dumps(report, indent=2) + '\n').encode('utf-8')
    report_sha = hashlib.sha256(body).hexdigest()
    (history / f'{report_sha}.json').write_bytes(body)
    import os
    staging = output.with_suffix('.json.tmp')
    staging.write_bytes(body)
    os.replace(staging, output)
    verified = sum((all((item['archive_checksum_match'], item['database_source_sha256_match'], item['database_decoded_sha256_match'], item['reference_flac_test_exit'] == 0, item['reference_native_pcm_byte_equal'])) for item in reports))
    print(json.dumps({'report': str(output), 'report_sha256': report_sha, 'verified_songs': verified, 'strict_primary_eligible_count': report['strict_primary_eligible_count']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
