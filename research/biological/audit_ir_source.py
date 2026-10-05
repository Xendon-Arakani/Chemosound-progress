from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
RECORD_URL = 'https://zenodo.org/api/records/7706021'
ROOT = Path(__file__).resolve().parents[2] / 'history' / 'ir_source_v1'

class RangeArchive(io.RawIOBase):

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.ranges: list[dict] = []
        self.entity_tag: str | None = None

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int=0) -> int:
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek mode')
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError('Negative archive position')
        return self.position

    def read(self, length: int=-1) -> bytes:
        length = min(self.size - self.position, length if length >= 0 else self.size - self.position)
        if length <= 0:
            return b''
        if length > 16 * 1024 * 1024:
            raise ValueError('Range audit refuses unexpectedly large archive reads')
        start = self.position
        end = start + length - 1
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server did not supply the exact requested byte range')
            entity_tag = response.headers.get('ETag')
            if self.entity_tag is not None and entity_tag != self.entity_tag:
                raise ValueError('Archive ETag changed during inspection')
            self.entity_tag = entity_tag
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('Archive range length mismatch')
        destination = ROOT / 'source-ranges' / f'{start}-{end}.bin'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        self.ranges.append({'start': start, 'end': end, 'bytes': length, 'sha256': hashlib.sha256(data).hexdigest(), 'etag': entity_tag, 'local_file': str(destination.relative_to(ROOT))})
        self.position += length
        return data

def digest(path: Path, algorithm: str='sha256') -> str:
    hasher = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_samples(index: list[dict]) -> dict:
    inspected = []
    fields = ('spectrum-type', 'adduct', 'optimization-method', 'frequency-method', 'single-point-method', 'data-source', 'x-units', 'y-units')
    for path in sorted((ROOT / 'samples').glob('*_protonated_predicted.xml')):
        doc = ET.parse(path).getroot()
        identifier = doc.findtext('database-id')
        metadata = {tag: doc.findtext(tag) for tag in fields}
        if doc.tag != 'ms-ir-comp' or any((value is None for value in metadata.values())):
            raise ValueError(f'Missing candidate metadata: {path.name}')
        stick_path = ROOT / 'samples' / f'Comp{identifier}pro_unscaled_sticks.xy'
        sticks = np.loadtxt(stick_path, ndmin=2)
        embedded = np.loadtxt(io.StringIO(doc.findtext('fundamental-ir') or ''), ndmin=2)
        if sticks.shape[1] != 2 or not np.isfinite(sticks).all():
            raise ValueError(f'Invalid candidate numerical fields: {stick_path.name}')
        if np.any(sticks[:, 0] <= 0) or np.any(sticks[:, 1] < 0):
            raise ValueError(f'Nonphysical candidate numerical fields: {stick_path.name}')
        if not np.array_equal(sticks, embedded):
            raise ValueError(f'Candidate XML and stick export disagree: {path.name}')
        coordinates = (doc.findtext('xyz-coordinates') or '').strip().splitlines()
        for line in coordinates:
            parts = line.split()
            if len(parts) != 4 or not np.isfinite([float(value) for value in parts[1:]]).all():
                raise ValueError(f'Invalid candidate geometry: {path.name}')
        scaling = doc.find('frequency-scaling')
        energies = doc.find('computed-energies')
        if scaling is None or energies is None:
            raise ValueError(f'Missing candidate scaling or energy metadata: {path.name}')
        inspected.append({'database_id': identifier, 'metadata': metadata, 'frequency_scaling': {child.tag: child.text for child in scaling}, 'energies': {child.tag: child.text for child in energies}, 'document_descriptions': [child.text for child in doc.findall('document-descriptions/document-description')], 'atom_count': len(coordinates), 'mode_count': len(sticks), 'matches_3n_minus_6': len(sticks) == 3 * len(coordinates) - 6, 'all_frequencies_positive': bool(np.all(sticks[:, 0] > 0)), 'all_intensities_positive': bool(np.all(sticks[:, 1] > 0)), 'frequency_min_cm_1': float(sticks[:, 0].min()), 'frequency_max_cm_1': float(sticks[:, 0].max()), 'xml_matches_stick_file_exactly': True, 'einstein_a_field_present': any(('einstein' in child.tag.lower() for child in doc.iter()))})
    agreement = all((item['metadata'] == inspected[0]['metadata'] and item['frequency_scaling'] == inspected[0]['frequency_scaling'] for item in inspected)) if inspected else False
    report = {'inspected_entries': inspected, 'inspected_entries_metadata_and_scaling_agree': agreement, 'protonated_xml_count_in_directory': sum((item['name'].endswith('_protonated_predicted.xml') for item in index)), 'protonated_stick_file_count_in_directory': sum((item['name'].endswith('pro_unscaled_sticks.xy') for item in index)), 'xml_counts_by_adduct': dict(Counter((item['name'].split('_')[-2] for item in index if item['name'].endswith('_predicted.xml')))), 'unsampled_files_numerically_validated': False, 'original_hitran_method_admitted': False, 'biological_music_outcomes_calculated': False}
    (ROOT / 'numerical-inspection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-archive', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD_URL, timeout=30) as response:
        metadata_bytes = response.read()
    record = json.loads(metadata_bytes)
    if str(record['id']) != '7706021':
        raise ValueError('Unexpected source record')
    source = next((item for item in record['files'] if item['key'] == 'ComputedIRspectra.zip'))
    expected_md5 = source['checksum'].removeprefix('md5:')
    if source['checksum'] != 'md5:' + expected_md5 or len(expected_md5) != 32:
        raise ValueError('Unsupported source checksum')
    archive = ROOT / source['key']
    if args.full_archive and (not archive.exists()):
        partial = archive.with_suffix('.zip.partial')
        started = time.monotonic()
        last_update = started
        copied = 0
        with urllib.request.urlopen(source['links']['self'], timeout=45) as response:
            with partial.open('wb') as stream:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    stream.write(chunk)
                    copied += len(chunk)
                    if time.monotonic() - last_update >= 15:
                        print(json.dumps({'downloaded_bytes': copied, 'expected_bytes': source['size']}), flush=True)
                        last_update = time.monotonic()
        if copied != source['size'] or digest(partial, 'md5') != expected_md5:
            raise ValueError('Source archive size or publisher checksum mismatch; partial retained')
        partial.replace(archive)
    if args.full_archive and (archive.stat().st_size != source['size'] or digest(archive, 'md5') != expected_md5):
        raise ValueError('Existing archive failed size or publisher checksum check')
    (ROOT / 'source-record.json').write_bytes(metadata_bytes)
    inspected = []
    remote = None if args.full_archive else RangeArchive(source['links']['self'], source['size'])
    with zipfile.ZipFile(archive if args.full_archive else remote) as z:
        entries = z.infolist()
        index = [{'name': item.filename, 'bytes': item.file_size, 'crc32': f'{item.CRC:08x}'} for item in entries]
        names = [item.filename for item in entries]
        target_ids = ('HMDB0000073', 'HMDB0000259')
        sample_ids = ('HMDB0000001', 'HMDB0000002', 'HMDB0000005')
        for name in names:
            if not any((identifier in name for identifier in target_ids + sample_ids)):
                continue
            if not (name.endswith('_protonated_predicted.xml') or name.endswith('pro_unscaled_sticks.xy')):
                continue
            if z.getinfo(name).file_size > 2 * 1024 * 1024:
                raise ValueError('Unexpectedly large sample member')
            data = z.read(name)
            destination = ROOT / 'samples' / Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            inspected.append({'source_member': name, 'local_file': destination.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'zip_crc_verified': True})
        (ROOT / 'archive-index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    counts = Counter((name.split('/')[1] if '/' in name else 'root' for name in names if not name.endswith('/')))
    numerical_inspection = inspect_samples(index)
    evidence = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'record_url': RECORD_URL, 'dataset_doi': record['metadata']['doi'], 'acquisition_mode': 'full_archive' if args.full_archive else 'selected_members_via_http_ranges', 'publisher_md5': expected_md5, 'publisher_md5_verified': args.full_archive, 'declared_archive_bytes': source['size'], 'archive_sha256': digest(archive) if args.full_archive else None, 'downloaded_ranges': remote.ranges if remote is not None else [], 'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(), 'index_sha256': digest(ROOT / 'archive-index.json'), 'numerical_inspection_sha256': digest(ROOT / 'numerical-inspection.json'), 'audit_script_sha256': digest(Path(__file__).resolve()), 'inspected_entries_metadata_and_scaling_agree': numerical_inspection['inspected_entries_metadata_and_scaling_agree'], 'file_counts_by_folder': dict(counts), 'inspected_members': inspected, 'original_hitran_method_admitted': False, 'biological_music_outcomes_calculated': False}
    (ROOT / 'acquisition-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'audit_complete': True, 'sample_members': len(inspected), 'sample_metadata_agree': evidence['inspected_entries_metadata_and_scaling_agree'], 'evidence_file': str(ROOT / 'acquisition-evidence.json')}), flush=True)
if __name__ == '__main__':
    main()
