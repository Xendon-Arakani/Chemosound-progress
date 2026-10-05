from __future__ import annotations
import contextlib
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from .bounded_relations import molecule_pair_grid, write_pair_measure
from .db import immediate, utc_now
PROTOCOL_MOLECULES: tuple[tuple[int, str, str], ...] = ((1, 'H2O', 'Water'), (2, 'CO2', 'Carbon dioxide'), (3, 'O3', 'Ozone'), (4, 'N2O', 'Nitrous oxide'), (5, 'CO', 'Carbon monoxide'), (6, 'CH4', 'Methane'), (7, 'O2', 'Oxygen'), (8, 'NO', 'Nitric oxide'), (9, 'SO2', 'Sulfur dioxide'), (10, 'NO2', 'Nitrogen dioxide'), (11, 'NH3', 'Ammonia'), (12, 'HNO3', 'Nitric acid'), (13, 'OH', 'Hydroxyl'), (14, 'HF', 'Hydrogen fluoride'), (15, 'HCl', 'Hydrogen chloride'), (16, 'HBr', 'Hydrogen bromide'), (17, 'HI', 'Hydrogen iodide'), (18, 'ClO', 'Chlorine monoxide'), (19, 'OCS', 'Carbonyl sulfide'), (20, 'H2CO', 'Formaldehyde'), (21, 'HOCl', 'Hypochlorous acid'), (22, 'N2', 'Nitrogen'), (23, 'HCN', 'Hydrogen cyanide'), (24, 'CH3Cl', 'Methyl chloride'), (25, 'H2O2', 'Hydrogen peroxide'), (26, 'C2H2', 'Acetylene'), (27, 'C2H6', 'Ethane'), (28, 'PH3', 'Phosphine'), (29, 'COF2', 'Carbonyl fluoride'), (30, 'SF6', 'Sulfur hexafluoride'), (31, 'H2S', 'Hydrogen sulfide'), (32, 'HCOOH', 'Formic acid'), (33, 'HO2', 'Hydroperoxyl'), (34, 'O', 'Oxygen atom'), (35, 'ClONO2', 'Chlorine nitrate'), (36, 'NO+', 'Nitric oxide cation'), (37, 'HOBr', 'Hypobromous acid'), (38, 'C2H4', 'Ethylene'), (39, 'CH3OH', 'Methanol'), (40, 'CH3Br', 'Methyl bromide'), (41, 'CH3CN', 'Acetonitrile'), (42, 'CF4', 'PFC-14 / tetrafluoromethane'), (43, 'C4H2', 'Diacetylene'), (44, 'HC3N', 'Cyanoacetylene'), (45, 'H2', 'Hydrogen'), (46, 'CS', 'Carbon monosulfide'), (47, 'SO3', 'Sulfur trioxide'), (48, 'C2N2', 'Cyanogen'), (49, 'COCl2', 'Phosgene'), (50, 'SO', 'Sulfur monoxide'), (51, 'CH3F', 'Methyl fluoride'), (52, 'GeH4', 'Germane'), (53, 'CS2', 'Carbon disulfide'), (54, 'CH3I', 'Methyl iodide'), (55, 'NF3', 'Nitrogen trifluoride'), (56, 'H3+', 'Trihydrogen cation'), (57, 'CH3', 'Methyl radical'), (58, 'S2', 'Sulfur dimer'), (59, 'COFCl', 'Carbonyl chlorofluoride'), (60, 'HONO', 'Nitrous acid'), (61, 'ClNO2', 'Nitryl chloride'))
STATIC_MOLECULE_IDS = frozenset({30, 35, 42, 55})

@dataclass(frozen=True, slots=True)
class HitranPaths:
    root: Path

    @property
    def source_metadata(self) -> Path:
        return self.root / 'source-metadata'

    @property
    def raw(self) -> Path:
        return self.root / 'raw'

    @property
    def static_packages(self) -> Path:
        return self.root / 'static-packages'

    @property
    def normalized(self) -> Path:
        return self.root / 'normalized'

    @property
    def representations(self) -> Path:
        return self.root / 'representations'

    @property
    def staging(self) -> Path:
        return self.root / 'staging'

    @property
    def quarantine(self) -> Path:
        return self.root / 'quarantine'

    @property
    def manifests(self) -> Path:
        return self.root / 'manifests'

    @property
    def working_manifest(self) -> Path:
        return self.manifests / 'snapshot-working.json'

    @property
    def frozen_manifest(self) -> Path:
        return self.manifests / 'snapshot-frozen.json'

    def ensure(self) -> None:
        for path in (self.source_metadata, self.raw, self.static_packages, self.normalized, self.representations, self.staging, self.quarantine, self.manifests):
            path.mkdir(parents=True, exist_ok=True)

def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(8 * 1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path: Path, document: dict) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + '\n').encode('utf-8')
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)

def _clean_html(value: str) -> str:
    value = re.sub('<[^>]+>', '', value)
    return html.unescape(value).strip().replace('\xa0', ' ')

def _number(value: str) -> float:
    value = _clean_html(value).replace('×', 'x')
    match = re.fullmatch('([0-9.]+)\\s*x\\s*10(-?\\d+)', value)
    if match:
        return float(match.group(1)) * 10 ** int(match.group(2))
    return float(value)

def _main_isotope_rows(text: str) -> dict[int, dict]:
    headings = list(re.finditer('<h4>(.*?)<input[^>]+onclick="select_none\\((\\d+)\\)"[^>]*></h4>', text, re.DOTALL))
    rows: dict[int, dict] = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.end():end]
        row_match = re.search('<tr class="selectable-row">(.*?)</tr>', section, re.DOTALL)
        if row_match is None:
            continue
        row_html = re.sub('<!--.*?-->', '', row_match.group(1), flags=re.DOTALL)
        cells = re.findall('<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
        if len(cells) < 8:
            raise ValueError(f'malformed isotope metadata for molecule {heading.group(2)}')
        global_match = re.search('<input[^>]+id="(\\d+)"', cells[0])
        if global_match is None:
            raise ValueError(f'missing global isotope ID for molecule {heading.group(2)}')
        molecule_id = int(heading.group(2))
        rows[molecule_id] = {'global_isotopologue_id': int(global_match.group(1)), 'isotopologue_formula': _clean_html(cells[1]), 'afgl_code': _clean_html(cells[2]), 'abundance': float(_clean_html(cells[3])), 'expected_source_line_count': int(_clean_html(cells[5])), 'source_nu_min_cm_1': _number(cells[6]), 'source_nu_max_cm_1': _number(cells[7])}
    return rows

def _molecule_rows(text: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for match in re.finditer('<tr class="selectable-row(?: not-in-db)?">(.*?)</tr>', text, re.DOTALL):
        row = match.group(1)
        id_match = re.search('(?:id="(\\d+)"[^>]*class="checkbox"|LBLstatic/(\\d+)/)', row)
        if id_match is None:
            continue
        molecule_id = int(id_match.group(1) or id_match.group(2))
        row = re.sub('<!--.*?-->', '', row, flags=re.DOTALL)
        cells = re.findall('<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) < 8:
            raise ValueError(f'malformed molecule metadata for molecule {molecule_id}')
        rows[molecule_id] = {'expected_all_isotopologue_line_count': int(_clean_html(cells[4])), 'all_isotopologue_nu_min_cm_1': _number(cells[5]), 'all_isotopologue_nu_max_cm_1': _number(cells[6])}
    return rows

def prepare_snapshot(root: Path, molecule_html: Path, isotopologue_html: Path, *, acquired_at: str) -> dict:
    paths = HitranPaths(root.resolve())
    paths.ensure()
    if paths.frozen_manifest.is_file():
        raise RuntimeError(f'HITRAN snapshot is already frozen: {paths.frozen_manifest}')
    molecule_rows = _molecule_rows(molecule_html.read_text(encoding='utf-8'))
    isotope_rows = _main_isotope_rows(isotopologue_html.read_text(encoding='utf-8'))
    expected_ids = {row[0] for row in PROTOCOL_MOLECULES}
    if set(molecule_rows) != expected_ids:
        missing = sorted(expected_ids.difference(molecule_rows))
        extra = sorted(set(molecule_rows).difference(expected_ids))
        raise ValueError(f'official molecule page mismatch; missing={missing}, extra={extra}')
    if set(isotope_rows) != expected_ids.difference(STATIC_MOLECULE_IDS):
        missing = sorted(expected_ids.difference(STATIC_MOLECULE_IDS, isotope_rows))
        extra = sorted(set(isotope_rows).difference(expected_ids))
        raise ValueError(f'official principal-isotopologue page mismatch; missing={missing}, extra={extra}')
    previous: dict[int, dict] = {}
    if paths.working_manifest.is_file():
        old = json.loads(paths.working_manifest.read_text(encoding='utf-8'))
        previous = {int(item['molecule_id']): item for item in old.get('items', [])}
    items: list[dict] = []
    for molecule_id, formula, common_name in PROTOCOL_MOLECULES:
        static = molecule_id in STATIC_MOLECULE_IDS
        metadata = molecule_rows[molecule_id] | isotope_rows.get(molecule_id, {})
        prior = previous.get(molecule_id, {})
        item = {'molecule_id': molecule_id, 'formula': formula, 'common_name': common_name, 'local_isotopologue_id': 1, 'source_kind': 'authenticated_static_package' if static else 'hapi_v1', 'source_url': f'https://hitran.org/files/LBLstatic/{molecule_id}/' if static else 'https://hitran.org/lbl/api', **metadata, 'status': prior.get('status', 'awaiting_static_package' if static else 'planned')}
        for key in ('raw_data_path', 'raw_data_sha256', 'raw_header_path', 'raw_header_sha256', 'observed_source_line_count', 'normalized_path', 'normalized_sha256', 'retained_line_count', 'rejected_nonpositive_nu', 'rejected_nonpositive_a', 'rejected_other_isotopologue', 'observed_nu_min_cm_1', 'observed_nu_max_cm_1', 'observed_a_min_s_1', 'observed_a_max_s_1', 'last_error', 'quarantine_data_path', 'quarantine_data_sha256', 'quarantine_header_path', 'quarantine_header_sha256', 'quarantine_observed_line_count'):
            if key in prior:
                item[key] = prior[key]
        items.append(item)
    document = {'schema_version': 1, 'database_name': 'HITRAN', 'edition': 'HITRAN2024', 'protocol_version': '1.0', 'acquired_at': acquired_at, 'prepared_at': _utc_now(), 'selection': {'molecule_ids': [1, 61], 'molecule_count': 61, 'local_isotopologue_id': 1, 'retain_if': 'nu_cm_1 > 0 and einstein_a_s_1 > 0', 'weight': 'einstein_a_s_1 / sum(einstein_a_s_1)'}, 'source_metadata': [{'path': str(molecule_html.resolve()), 'sha256': _sha256(molecule_html)}, {'path': str(isotopologue_html.resolve()), 'sha256': _sha256(isotopologue_html)}], 'status': 'building', 'items': items}
    _atomic_json(paths.working_manifest, document)
    return document

def load_work_manifest(root: Path) -> tuple[HitranPaths, dict]:
    paths = HitranPaths(root.resolve())
    paths.ensure()
    if not paths.working_manifest.is_file():
        raise RuntimeError('HITRAN snapshot is not prepared')
    return (paths, json.loads(paths.working_manifest.read_text(encoding='utf-8')))

def sync_snapshot_db(connection, root: Path) -> str:
    paths, document = load_work_manifest(root)
    manifest_path = paths.frozen_manifest if document['status'] == 'frozen' and paths.frozen_manifest.is_file() else paths.working_manifest
    snapshot_id = f"hitran2024-{document['acquired_at']}"
    manifest_sha256 = _sha256(manifest_path)
    with immediate(connection):
        connection.execute('INSERT INTO chemistry_snapshots(id, database_name, edition, source_url, acquired_at, manifest_path, manifest_sha256, status, frozen_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET manifest_path=excluded.manifest_path, manifest_sha256=excluded.manifest_sha256, status=excluded.status, frozen_at=excluded.frozen_at', (snapshot_id, document['database_name'], document['edition'], 'https://hitran.org/lbl/', document['acquired_at'], str(manifest_path), manifest_sha256, document['status'], document.get('frozen_at')))
        for item in document['items']:
            expected = item.get('expected_source_line_count', item.get('expected_all_isotopologue_line_count'))
            connection.execute('INSERT INTO chemistry_ingest_items(snapshot_id, molecule_id, source_kind, source_url, status, expected_source_line_count, observed_source_line_count, retained_line_count, raw_data_path, raw_data_sha256, normalized_path, normalized_sha256, last_error, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(snapshot_id, molecule_id) DO UPDATE SET source_kind=excluded.source_kind, source_url=excluded.source_url, status=excluded.status, expected_source_line_count=excluded.expected_source_line_count, observed_source_line_count=excluded.observed_source_line_count, retained_line_count=excluded.retained_line_count, raw_data_path=excluded.raw_data_path, raw_data_sha256=excluded.raw_data_sha256, normalized_path=excluded.normalized_path, normalized_sha256=excluded.normalized_sha256, last_error=excluded.last_error, updated_at=excluded.updated_at', (snapshot_id, item['molecule_id'], item['source_kind'], item['source_url'], item['status'], expected, item.get('observed_source_line_count'), item.get('retained_line_count'), item.get('raw_data_path'), item.get('raw_data_sha256'), item.get('normalized_path'), item.get('normalized_sha256'), item.get('last_error'), utc_now()))
            if item['status'] == 'normalized':
                connection.execute('INSERT INTO molecules(snapshot_id, molecule_id, formula, common_name, local_isotopologue_id, retained_line_count, transition_path, transition_sha256) VALUES(?, ?, ?, ?, 1, ?, ?, ?) ON CONFLICT(snapshot_id, molecule_id) DO UPDATE SET retained_line_count=excluded.retained_line_count, transition_path=excluded.transition_path, transition_sha256=excluded.transition_sha256', (snapshot_id, item['molecule_id'], item['formula'], item['common_name'], item['retained_line_count'], item['normalized_path'], item['normalized_sha256']))
    return snapshot_id

def _raw_base(paths: HitranPaths, molecule_id: int) -> Path:
    return paths.raw / f'molecule_{molecule_id:03d}_iso_1'

def _count_lines(path: Path) -> int:
    count = 0
    with path.open('rb') as stream:
        for count, _line in enumerate(stream, start=1):
            pass
    return count

def acquire_hapi_sources(root: Path) -> dict:
    paths, document = load_work_manifest(root)
    if document['status'] == 'frozen':
        raise RuntimeError('cannot acquire into a frozen HITRAN snapshot')
    try:
        import hapi
    except ImportError as exc:
        raise RuntimeError('hitran-api is required for acquisition') from exc
    completed = skipped = failed = 0
    for item in document['items']:
        if item['source_kind'] != 'hapi_v1':
            continue
        molecule_id = int(item['molecule_id'])
        base = _raw_base(paths, molecule_id)
        data_path = base.with_suffix('.data')
        header_path = base.with_suffix('.header')
        expected = int(item['expected_source_line_count'])
        if data_path.is_file() and header_path.is_file():
            observed = _count_lines(data_path)
            if observed == expected:
                item.update({'status': 'downloaded', 'raw_data_path': str(data_path), 'raw_data_sha256': _sha256(data_path), 'raw_header_path': str(header_path), 'raw_header_sha256': _sha256(header_path), 'observed_source_line_count': observed, 'last_error': None})
                skipped += 1
                continue
            raise RuntimeError(f'existing raw molecule {molecule_id} has {observed} lines; expected {expected}')
        temp_directory = Path(tempfile.mkdtemp(prefix=f'hitran-{molecule_id:03d}-', dir=paths.staging))
        table_name = f'molecule_{molecule_id:03d}_iso_1'
        try:
            hapi.VARIABLES['GLOBAL_HOST'] = 'https://hitran.org'
            hapi.VARIABLES['DISPLAY_FETCH_URL'] = False
            with contextlib.redirect_stdout(io.StringIO()):
                hapi.db_begin(str(temp_directory))
                hapi.queryHITRAN(table_name, [int(item['global_isotopologue_id'])], 0.0, float(item['source_nu_max_cm_1']) + 0.001)
            staged_data = temp_directory / f'{table_name}.data'
            staged_header = temp_directory / f'{table_name}.header'
            observed = _count_lines(staged_data)
            if observed != expected:
                quarantine_data = paths.quarantine / f'molecule_{molecule_id:03d}_iso_1.data'
                quarantine_header = paths.quarantine / f'molecule_{molecule_id:03d}_iso_1.header'
                os.replace(staged_data, quarantine_data)
                os.replace(staged_header, quarantine_header)
                item.update({'quarantine_data_path': str(quarantine_data), 'quarantine_data_sha256': _sha256(quarantine_data), 'quarantine_header_path': str(quarantine_header), 'quarantine_header_sha256': _sha256(quarantine_header), 'quarantine_observed_line_count': observed})
                raise ValueError(f'server returned {observed} lines; metadata declares {expected}')
            os.replace(staged_data, data_path)
            os.replace(staged_header, header_path)
            item.update({'status': 'downloaded', 'raw_data_path': str(data_path), 'raw_data_sha256': _sha256(data_path), 'raw_header_path': str(header_path), 'raw_header_sha256': _sha256(header_path), 'observed_source_line_count': observed, 'last_error': None})
            completed += 1
        except Exception as exc:
            item['status'] = 'failed'
            item['last_error'] = f'{type(exc).__name__}: {exc}'
            failed += 1
            _atomic_json(paths.working_manifest, document)
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)
        _atomic_json(paths.working_manifest, document)
    _atomic_json(paths.working_manifest, document)
    return {'downloaded': completed, 'already_present': skipped, 'failed': failed}

def _v2_json(api_key: str, endpoint: str, query: dict[str, str] | None=None) -> dict:
    encoded_key = urllib.parse.quote(api_key, safe='')
    url = f'https://hitran.org/api/v2/{encoded_key}/{endpoint}'
    if query:
        url += '?' + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={'User-Agent': 'Chemosound/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            document = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HITRAN API v2 returned HTTP {exc.code}') from None
    if document.get('status') != 'OK':
        raise RuntimeError(f"HITRAN API v2 rejected {endpoint}: {document.get('message', 'unknown error')}")
    return document

def _download_v2_result(result_directory: str, filename: str, destination: Path) -> None:
    if Path(filename).name != filename:
        raise ValueError('HITRAN API returned an unsafe result filename')
    quoted = urllib.parse.quote(filename, safe='')
    prefix = result_directory.strip('/')
    url = f'https://hitran.org/{prefix}/{quoted}'
    request = urllib.request.Request(url, headers={'User-Agent': 'Chemosound/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with destination.open('wb') as stream:
                while (chunk := response.read(8 * 1024 * 1024)):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HITRAN result download returned HTTP {exc.code}') from None

def acquire_v2_sources(root: Path, api_key: str) -> dict:
    if not api_key.strip():
        raise ValueError('a HITRAN API key is required')
    paths, document = load_work_manifest(root)
    if document['status'] == 'frozen':
        raise RuntimeError('cannot acquire into a frozen HITRAN snapshot')
    info = _v2_json(api_key, 'info')
    result_directory = str(info['content']['data']['results_dir'])
    completed = skipped = failed = 0
    for item in document['items']:
        molecule_id = int(item['molecule_id'])
        if molecule_id in STATIC_MOLECULE_IDS:
            continue
        base = _raw_base(paths, molecule_id)
        data_path = base.with_suffix('.data')
        header_path = base.with_suffix('.header')
        expected = int(item['expected_source_line_count'])
        if item.get('source_api_version') == 'v2' and data_path.is_file() and header_path.is_file() and (_count_lines(data_path) == expected):
            item.update({'status': 'downloaded', 'raw_data_path': str(data_path), 'raw_data_sha256': _sha256(data_path), 'raw_header_path': str(header_path), 'raw_header_sha256': _sha256(header_path), 'observed_source_line_count': expected, 'last_error': None})
            skipped += 1
            continue
        temp_directory = Path(tempfile.mkdtemp(prefix=f'hitran-v2-{molecule_id:03d}-', dir=paths.staging))
        staged_data = temp_directory / f'molecule_{molecule_id:03d}_iso_1.data'
        staged_header = temp_directory / f'molecule_{molecule_id:03d}_iso_1.header'
        try:
            response_header = _v2_json(api_key, 'transitions', {'iso_ids_list': str(item['global_isotopologue_id']), 'numin': '0', 'numax': str(float(item['source_nu_max_cm_1']) + 0.001), 'head': 'False', 'fixwidth': '0', 'request_params': 'par_line'})
            if response_header.get('content', {}).get('format') != 'text/hapi':
                raise ValueError('HITRAN API did not return the requested HAPI line format')
            filename = str(response_header['content']['data'])
            _download_v2_result(result_directory, filename, staged_data)
            with staged_header.open('wb') as stream:
                stream.write((json.dumps(response_header, indent=2, sort_keys=True) + '\n').encode('utf-8'))
                stream.flush()
                os.fsync(stream.fileno())
            observed = _count_lines(staged_data)
            if observed != expected:
                quarantine_data = paths.quarantine / f'molecule_{molecule_id:03d}_iso_1_v2.data'
                quarantine_header = paths.quarantine / f'molecule_{molecule_id:03d}_iso_1_v2.header'
                os.replace(staged_data, quarantine_data)
                os.replace(staged_header, quarantine_header)
                item.update({'quarantine_data_path': str(quarantine_data), 'quarantine_data_sha256': _sha256(quarantine_data), 'quarantine_header_path': str(quarantine_header), 'quarantine_header_sha256': _sha256(quarantine_header), 'quarantine_observed_line_count': observed})
                raise ValueError(f'server returned {observed} lines; metadata declares {expected}')
            os.replace(staged_data, data_path)
            os.replace(staged_header, header_path)
            item.update({'source_kind': 'hitran_api_v2', 'source_api_version': 'v2', 'source_url': 'https://hitran.org/api/v2/<redacted>/transitions', 'status': 'downloaded', 'raw_data_path': str(data_path), 'raw_data_sha256': _sha256(data_path), 'raw_header_path': str(header_path), 'raw_header_sha256': _sha256(header_path), 'observed_source_line_count': observed, 'last_error': None})
            completed += 1
        except Exception as exc:
            item.update({'source_kind': 'hitran_api_v2', 'source_api_version': 'v2', 'source_url': 'https://hitran.org/api/v2/<redacted>/transitions', 'status': 'failed', 'last_error': f'{type(exc).__name__}: {exc}'})
            failed += 1
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)
        _atomic_json(paths.working_manifest, document)
    _atomic_json(paths.working_manifest, document)
    return {'downloaded': completed, 'already_present': skipped, 'failed': failed}

def register_static(root: Path, molecule_id: int, data_path: Path, header_path: Path | None=None) -> dict:
    if molecule_id not in STATIC_MOLECULE_IDS:
        raise ValueError(f'molecule {molecule_id} is not an authenticated static source')
    paths, document = load_work_manifest(root)
    if document['status'] == 'frozen':
        raise RuntimeError('cannot register data into a frozen HITRAN snapshot')
    source = data_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = paths.static_packages / f'molecule_{molecule_id:03d}_source.par'
    if destination.exists() and _sha256(destination) != _sha256(source):
        raise RuntimeError(f'refusing to replace a different static source: {destination}')
    if not destination.exists():
        shutil.copy2(source, destination)
    item = next((row for row in document['items'] if row['molecule_id'] == molecule_id))
    observed = _count_lines(destination)
    if observed != int(item['expected_all_isotopologue_line_count']):
        raise ValueError(f"static source has {observed} lines; official metadata declares {item['expected_all_isotopologue_line_count']}")
    item.update({'status': 'downloaded', 'raw_data_path': str(destination), 'raw_data_sha256': _sha256(destination), 'observed_source_line_count': observed, 'last_error': None})
    if header_path is not None:
        header_source = header_path.resolve()
        header_destination = paths.static_packages / f'molecule_{molecule_id:03d}_source.header'
        if not header_destination.exists():
            shutil.copy2(header_source, header_destination)
        item['raw_header_path'] = str(header_destination)
        item['raw_header_sha256'] = _sha256(header_destination)
    _atomic_json(paths.working_manifest, document)
    return item

def _parse_hitran_line(raw: bytes, line_number: int) -> tuple[int, int, float, float, bytes, bytes]:
    line = raw.rstrip(b'\r\n')
    if len(line) != 160:
        raise ValueError(f'line {line_number} has {len(line)} bytes; expected 160')
    try:
        molecule_id = int(line[0:2])
        local_isotope_id = int(line[2:3])
        nu = float(line[3:15])
        einstein_a = float(line[25:35])
    except ValueError as exc:
        raise ValueError(f'line {line_number} has invalid fixed-width fields') from exc
    return (molecule_id, local_isotope_id, nu, einstein_a, line[127:133], line[133:145])

def normalize_sources(root: Path) -> dict:
    paths, document = load_work_manifest(root)
    if document['status'] == 'frozen':
        raise RuntimeError('cannot normalize a frozen HITRAN snapshot')
    normalized = already = 0
    for item in document['items']:
        if item['status'] == 'normalized':
            path = Path(item['normalized_path'])
            if path.is_file() and _sha256(path) == item['normalized_sha256']:
                already += 1
                continue
        if item['status'] not in {'downloaded', 'normalized'}:
            continue
        molecule_id = int(item['molecule_id'])
        source = Path(item['raw_data_path'])
        if _sha256(source) != item['raw_data_sha256']:
            raise RuntimeError(f'raw source changed for molecule {molecule_id}')
        nu_values: list[float] = []
        a_values: list[float] = []
        source_rows: list[int] = []
        ierr_values: list[bytes] = []
        iref_values: list[bytes] = []
        rejected_nu = rejected_a = other_isotope_count = 0
        observed_count = 0
        with source.open('rb') as stream:
            for line_number, raw in enumerate(stream, start=1):
                observed_count += 1
                parsed_molecule, local_iso, nu, einstein_a, ierr, iref = _parse_hitran_line(raw, line_number)
                if parsed_molecule != molecule_id:
                    raise ValueError(f'molecule {molecule_id} source contains molecule {parsed_molecule} at line {line_number}')
                if local_iso != 1:
                    if item['source_kind'] == 'authenticated_static_package':
                        other_isotope_count += 1
                        continue
                    raise ValueError(f'molecule {molecule_id} API source contains local isotopologue {local_iso} at line {line_number}; request required 1')
                if not math.isfinite(nu) or nu <= 0:
                    rejected_nu += 1
                    continue
                if not math.isfinite(einstein_a) or einstein_a <= 0:
                    rejected_a += 1
                    continue
                nu_values.append(nu)
                a_values.append(einstein_a)
                source_rows.append(line_number)
                ierr_values.append(ierr)
                iref_values.append(iref)
        if observed_count != int(item['observed_source_line_count']):
            raise RuntimeError(f'source line count changed for molecule {molecule_id}')
        if len(nu_values) < 2:
            raise ValueError(f'molecule {molecule_id} has {len(nu_values)} retained lines; a pair distribution requires at least two')
        nu_array = np.asarray(nu_values, dtype='<f8')
        a_array = np.asarray(a_values, dtype='<f8')
        order = np.argsort(nu_array, kind='stable')
        nu_array = nu_array[order]
        a_array = a_array[order]
        row_array = np.asarray(source_rows, dtype='<u8')[order]
        ierr_array = np.asarray(ierr_values, dtype='S6')[order]
        iref_array = np.asarray(iref_values, dtype='S12')[order]
        destination = paths.normalized / f'molecule_{molecule_id:03d}_iso_1.npz'
        partial = paths.staging / f'molecule_{molecule_id:03d}_iso_1.npz.partial'
        with partial.open('wb') as stream:
            np.savez_compressed(stream, nu_cm_1=nu_array, einstein_a_s_1=a_array, source_line_number=row_array, uncertainty_indices=ierr_array, **{'reference_identifiers': iref_array})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
        item.update({'status': 'normalized', 'normalized_path': str(destination), 'normalized_sha256': _sha256(destination), 'retained_line_count': int(nu_array.size), 'rejected_nonpositive_nu': rejected_nu, 'rejected_nonpositive_a': rejected_a, 'rejected_other_isotopologue': other_isotope_count, 'observed_nu_min_cm_1': float(nu_array[0]), 'observed_nu_max_cm_1': float(nu_array[-1]), 'observed_a_min_s_1': float(np.min(a_array)), 'observed_a_max_s_1': float(np.max(a_array)), 'last_error': None})
        normalized += 1
        _atomic_json(paths.working_manifest, document)
    return {'normalized': normalized, 'already_present': already}

def freeze_snapshot(root: Path) -> tuple[str, dict]:
    paths, document = load_work_manifest(root)
    incomplete = [int(item['molecule_id']) for item in document['items'] if item['status'] != 'normalized']
    if incomplete:
        raise RuntimeError(f'cannot freeze HITRAN snapshot; incomplete molecules: {incomplete}')
    for item in document['items']:
        for path_key, hash_key in (('raw_data_path', 'raw_data_sha256'), ('normalized_path', 'normalized_sha256')):
            path = Path(item[path_key])
            if not path.is_file() or _sha256(path) != item[hash_key]:
                raise RuntimeError(f'snapshot artifact changed: {path}')
    document['status'] = 'frozen'
    document['frozen_at'] = _utc_now()
    _atomic_json(paths.frozen_manifest, document)
    _atomic_json(paths.working_manifest, document)
    digest = _sha256(paths.frozen_manifest)
    (paths.manifests / 'snapshot-frozen.sha256').write_text(f'{digest}  {paths.frozen_manifest.name}\n', encoding='ascii')
    return (digest, document)

def build_molecule_grids(connection, root: Path, *, coordinate_quantum: float) -> dict:
    if not math.isfinite(coordinate_quantum) or coordinate_quantum <= 0:
        raise ValueError('coordinate_quantum must be finite and positive')
    paths, document = load_work_manifest(root)
    snapshot_id = sync_snapshot_db(connection, root)
    pair_grid_version = f'molecule-relational-grid-v1-q{coordinate_quantum:.12g}'
    with immediate(connection):
        connection.execute("UPDATE molecular_representations SET status='pending', attempt_id=NULL, last_error='recovered interrupted attempt', updated_at=? WHERE snapshot_id=? AND representation_version=? AND status='processing'", (utc_now(), snapshot_id, pair_grid_version))
    completed = already = failed = 0
    for item in document['items']:
        if item['status'] != 'normalized':
            continue
        molecule_id = int(item['molecule_id'])
        artifact = paths.representations / pair_grid_version / f'molecule_{molecule_id:03d}_iso_1.npz'
        row = connection.execute("SELECT status, artifact_path, artifact_sha256 FROM molecular_representations WHERE snapshot_id=? AND molecule_id=? AND representation_version=? AND weighting='einstein_a'", (snapshot_id, molecule_id, pair_grid_version)).fetchone()
        if row is not None and row['status'] == 'complete' and row['artifact_path'] and Path(row['artifact_path']).is_file() and (_sha256(Path(row['artifact_path'])) == row['artifact_sha256']):
            already += 1
            continue
        attempt_id = os.urandom(16).hex()
        with immediate(connection):
            connection.execute("INSERT INTO molecular_representations(snapshot_id, molecule_id, representation_version, weighting, status, attempt_id, attempts, started_at, updated_at) VALUES(?, ?, ?, 'einstein_a', 'processing', ?, 1, ?, ?) ON CONFLICT(snapshot_id, molecule_id, representation_version, weighting) DO UPDATE SET status='processing', attempt_id=excluded.attempt_id, attempts=molecular_representations.attempts+1, started_at=excluded.started_at, completed_at=NULL, last_error=NULL, updated_at=excluded.updated_at", (snapshot_id, molecule_id, pair_grid_version, attempt_id, utc_now(), utc_now()))
        try:
            normalized_path = Path(item['normalized_path'])
            if _sha256(normalized_path) != item['normalized_sha256']:
                raise RuntimeError(f'normalized molecule {molecule_id} changed')
            with np.load(normalized_path, allow_pickle=False) as arrays:
                measure = molecule_pair_grid(arrays['nu_cm_1'], arrays['einstein_a_s_1'], coordinate_quantum=coordinate_quantum)
            artifact_sha256 = write_pair_measure(artifact, measure, {'schema_version': 1, 'protocol_version': document['protocol_version'], 'representation_version': pair_grid_version, 'snapshot_id': snapshot_id, 'molecule_id': molecule_id, 'formula': item['formula'], 'weighting': 'normalized Einstein A', 'source_normalized_sha256': item['normalized_sha256'], 'support_units': 'absolute natural-log wavenumber ratio'})
            with immediate(connection):
                connection.execute("UPDATE molecular_representations SET status='complete', retained_line_count=?, unordered_pair_count=?, artifact_path=?, artifact_sha256=?, completed_at=?, updated_at=?, attempt_id=NULL WHERE snapshot_id=? AND molecule_id=? AND representation_version=? AND weighting='einstein_a' AND attempt_id=?", (measure.input_count, measure.unordered_pair_count, str(artifact), artifact_sha256, utc_now(), utc_now(), snapshot_id, molecule_id, pair_grid_version, attempt_id))
            completed += 1
        except Exception as exc:
            with immediate(connection):
                connection.execute("UPDATE molecular_representations SET status='failed', last_error=?, updated_at=?, attempt_id=NULL WHERE snapshot_id=? AND molecule_id=? AND representation_version=? AND weighting='einstein_a' AND attempt_id=?", (f'{type(exc).__name__}: {exc}', utc_now(), snapshot_id, molecule_id, pair_grid_version, attempt_id))
            failed += 1
    return {'snapshot_id': snapshot_id, 'representation_version': pair_grid_version, 'completed': completed, 'already_present': already, 'failed': failed}

def snapshot_status(root: Path) -> dict:
    paths, document = load_work_manifest(root)
    counts: dict[str, int] = {}
    source_lines = retained_lines = raw_bytes = normalized_bytes = 0
    for item in document['items']:
        counts[item['status']] = counts.get(item['status'], 0) + 1
        source_lines += int(item.get('observed_source_line_count', 0))
        retained_lines += int(item.get('retained_line_count', 0))
        for key, accumulator in (('raw_data_path', 'raw'), ('normalized_path', 'normalized')):
            if key in item and Path(item[key]).is_file():
                if accumulator == 'raw':
                    raw_bytes += Path(item[key]).stat().st_size
                else:
                    normalized_bytes += Path(item[key]).stat().st_size
    return {'root': str(paths.root), 'edition': document['edition'], 'status': document['status'], 'molecule_status': counts, 'observed_source_lines': source_lines, 'retained_lines': retained_lines, 'raw_bytes': raw_bytes, 'normalized_bytes': normalized_bytes, 'static_packages_required': sorted(STATIC_MOLECULE_IDS), 'frozen_manifest': str(paths.frozen_manifest) if paths.frozen_manifest.is_file() else None}
