from __future__ import annotations
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('surface_pcm_audit', ROOT / 'research/acquisition/audit_existing.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
UA = 'Chemosound-surface-acquisition/1.0'
GIB = 1024 ** 3
LICENSES = [f'{scheme}://creativecommons.org/licenses/{kind}/{version}/' for scheme in ('http', 'https') for kind in ('by', 'by-sa') for version in ('3.0', '4.0')]

def now():
    return datetime.now(UTC).isoformat()

def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def text(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return ' | '.join((text(item) for item in value))
    return str(value)

def origin_allowed(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or '').lower()
    return parsed.scheme == 'https' and (host == 'archive.org' or host.endswith('.archive.org')) and (not parsed.username)

class Client:

    def __init__(self, batch, budget):
        self.batch = batch
        self.budget = budget
        self.last_request = 0.0
        self.transferred = 0
        self.reserved = 0
        self.throttle_lock = threading.Lock()
        self.budget_lock = threading.Lock()
        self.rate_limited = False
        client = self

        class Redirects(urllib.request.HTTPRedirectHandler):
            max_redirections = 3
            max_repeats = 1

            def redirect_request(self, req, fp, code, msg, headers, newurl):
                if not origin_allowed(newurl):
                    raise ValueError('Redirect outside permitted Archive HTTPS hosts')
                client.throttle()
                return super().redirect_request(req, fp, code, msg, headers, newurl)
        self.opener = urllib.request.build_opener(Redirects())

    def throttle(self):
        with self.throttle_lock:
            if self.rate_limited:
                raise RuntimeError('source_rate_limited')
            time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()

    def open(self, url):
        if not origin_allowed(url):
            raise ValueError('Invalid Archive URL')
        self.throttle()
        try:
            return self.opener.open(urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'identity'}), timeout=10)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                self.rate_limited = True
                atomic(self.batch / 'rate-limit-receipt.json', canonical({'source_url': url, 'captured_at_utc': now(), 'status': 429, 'retry_after': exc.headers.get('Retry-After'), 'action': 'stop_new_requests'}))
            raise

    def evidence(self, url, *, limit=16 * 1024 ** 2):
        with self.open(url) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError('Evidence response exceeds limit')
            receipt = {'source_url': url, 'final_url': response.url, 'captured_at_utc': now(), 'status': response.status, 'headers': dict(response.headers), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        atomic(self.batch / 'evidence' / receipt['sha256'], body)
        return (body, receipt)

    def download(self, url, path, *, size, md5, sha1):
        with self.budget_lock:
            if self.transferred + self.reserved + size > self.budget:
                raise RuntimeError('batch_transfer_budget_reached')
            self.reserved += size
        count = 0
        try:
            return self._download_reserved(url, path, size=size, md5=md5, sha1=sha1)
        finally:
            with self.budget_lock:
                self.reserved -= self._remaining.pop(str(path), size)

    def _download_reserved(self, url, path, *, size, md5, sha1):
        with self.budget_lock:
            if not hasattr(self, '_remaining'):
                self._remaining = {}
            self._remaining[str(path)] = size
        if shutil.disk_usage(self.batch).free < size + 10 * GIB:
            raise RuntimeError('insufficient_free_space')
        started = time.monotonic()
        with self.open(url) as response, path.open('wb') as stream:
            length = response.headers.get('Content-Length')
            if response.status != 200 or (length is not None and int(length) != size):
                raise ValueError('Unexpected download status/content length')
            count = 0
            while True:
                chunk = response.read(1024 ** 2)
                if not chunk:
                    break
                count += len(chunk)
                with self.budget_lock:
                    used_reservation = min(len(chunk), self._remaining[str(path)])
                    self._remaining[str(path)] -= used_reservation
                    self.reserved -= used_reservation
                    self.transferred += len(chunk)
                if count > size or self.transferred > self.budget:
                    raise ValueError('Transfer exceeds expected size or batch budget')
                if time.monotonic() - started > 150:
                    raise TimeoutError('Transfer exceeded 150-second deadline')
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
            receipt = {'requested_url': url, 'final_url': response.url, 'captured_at_utc': now(), 'status': response.status, 'headers': dict(response.headers), 'bytes': count}
        if count != size or audit.file_hash(path, 'md5') != md5:
            raise ValueError('Whole-file size/MD5 mismatch')
        if sha1 and audit.file_hash(path, 'sha1') != sha1:
            raise ValueError('Whole-file SHA-1 mismatch')
        receipt['source_sha256'] = audit.file_hash(path)
        return receipt

def prefetched_downloads(files, client, batch, identifier):
    pool = ThreadPoolExecutor(max_workers=4)
    queue = deque()
    iterator = iter(files)

    def transfer(file):
        key = hashlib.sha256(canonical([identifier, file['name']])).hexdigest()
        url = 'https://archive.org/download/' + urllib.parse.quote(identifier, safe='') + '/' + urllib.parse.quote(file['name'], safe='')
        path = batch / 'staging' / (key + Path(file['name']).suffix.lower())
        for attempt in range(3):
            try:
                return client.download(url, path, size=int(file['size']), md5=file['md5'], sha1=file.get('sha1'))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == 2 or client.rate_limited:
                    raise
                print(json.dumps({'event': 'retry', 'filename': file['name'], 'attempt': attempt + 1, 'error': str(exc)}), flush=True)
                time.sleep(2 ** attempt)

    def submit():
        file = next(iterator, None)
        if file is not None:
            queue.append((file, pool.submit(transfer, file)))
    try:
        for _ in range(4):
            submit()
        while queue:
            file, future = queue.popleft()
            yield (file, future)
            submit()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

def flac_comments(path):
    comments = []
    vendor = None
    with path.open('rb') as stream:
        if stream.read(4) != b'fLaC':
            raise ValueError('Not native FLAC')
        while True:
            header = stream.read(4)
            if len(header) != 4:
                raise ValueError('Truncated FLAC metadata')
            size = int.from_bytes(header[1:], 'big')
            body = stream.read(size)
            if len(body) != size:
                raise ValueError('Truncated FLAC metadata block')
            if header[0] & 127 == 4:
                position = 0

                def field():
                    nonlocal position
                    if position + 4 > len(body):
                        raise ValueError('Truncated FLAC comment length')
                    n = int.from_bytes(body[position:position + 4], 'little')
                    position += 4
                    if position + n > len(body):
                        raise ValueError('Truncated FLAC comment')
                    result = body[position:position + n].decode('utf-8', errors='replace')
                    position += n
                    return result
                vendor = field()
                if position + 4 > len(body):
                    raise ValueError('Missing FLAC comment count')
                count = int.from_bytes(body[position:position + 4], 'little')
                position += 4
                if count > len(body) // 4:
                    raise ValueError('Impossible FLAC comment count')
                for _ in range(count):
                    comments.append(field())
            if header[0] & 128:
                break
    return {'vendor': vendor, 'comments': comments}

def verify_native_wav(path):
    with wave.open(str(path), 'rb') as reference, audit.sf.SoundFile(path) as decoded:
        geometry = (reference.getframerate(), reference.getnchannels(), reference.getnframes())
        if geometry != (decoded.samplerate, decoded.channels, decoded.frames):
            raise ValueError('WAV reference/sample geometry disagreement')
        width = reference.getsampwidth()
        if width not in (1, 2, 3, 4) or reference.getcomptype() != 'NONE':
            raise ValueError('Unsupported native WAV sample width/compression')
        frames = 0
        while True:
            block = decoded.read(262144, dtype='int32', always_2d=True)
            if not block.size:
                if reference.readframes(1):
                    raise ValueError('WAV reference has extra samples')
                break
            integers = (block.astype(audit.np.int64) >> 32 - width * 8).reshape(-1)
            if width == 1:
                integers += 128
            packed = audit.np.empty((integers.size, width), dtype=audit.np.uint8)
            for byte in range(width):
                packed[:, byte] = integers >> 8 * byte & 255
            if reference.readframes(len(block)) != packed.tobytes():
                raise ValueError('WAV raw PCM/native decode byte mismatch')
            frames += len(block)
        if frames != geometry[2]:
            raise ValueError('WAV reference frame count mismatch')
    return True

def wav_comments(path):
    comments = []
    unparsed = []
    names = {b'INAM': 'TITLE', b'IART': 'ARTIST', b'IPRD': 'ALBUM', b'ICOP': 'COPYRIGHT', b'IGNR': 'GENRE', b'ICRD': 'DATE'}
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('Unsupported WAV container')
        total = path.stat().st_size
        while stream.tell() + 8 <= total:
            chunk_header = stream.read(8)
            kind, size = (chunk_header[:4], int.from_bytes(chunk_header[4:], 'little'))
            start = stream.tell()
            if start + size > total:
                raise ValueError('Truncated WAV chunk')
            if kind in {b'LIST', b'id3 ', b'ID3 '} and size <= 8 * 1024 ** 2:
                data = stream.read(size)
                if kind == b'LIST' and data[:4] == b'INFO':
                    pos = 4
                    while pos + 8 <= len(data):
                        key, n = (data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], 'little'))
                        pos += 8
                        if pos + n > len(data):
                            raise ValueError('Truncated WAV INFO tag')
                        value = data[pos:pos + n].rstrip(b'\x00').decode('utf-8', errors='replace')
                        comments.append('RIFF_' + key.decode('ascii', errors='replace') + '=' + value)
                        if key in names:
                            comments.append(names[key] + '=' + value)
                        pos += n + n % 2
                else:
                    unparsed.append({'chunk': kind.decode('ascii', errors='replace'), 'bytes': size, 'sha256': hashlib.sha256(data).hexdigest(), 'status': 'retained-in-original-not-semantically-parsed'})
            stream.seek(start + size + size % 2)
    return {'vendor': None, 'comments': comments, 'unparsed_metadata_chunks': unparsed}

def export_reports(batch, records, *, target, transferred):
    columns = ['number', 'title_tag', 'artist_tag', 'release_title_claim', 'creator_claim', 'filename', 'license_claim', 'duration_seconds', 'sample_rate', 'channels', 'bits_per_sample', 'bytes', 'source_sha256', 'pcm_sha256', 'source_page', 'audio_relative_path', 'flags']
    rows = []
    for index, record in enumerate(records, 1):
        tags = {}
        for comment in record['embedded_tags']['comments']:
            key, sep, value = comment.partition('=')
            if sep:
                tags.setdefault(key.upper(), []).append(value)
        pcm = record['pcm_measurements']
        claims = record['item_metadata_claims']
        rows.append(dict(number=index, title_tag=text(tags.get('TITLE')), artist_tag=text(tags.get('ARTIST')), release_title_claim=text(claims.get('title')), creator_claim=text(claims.get('creator')), filename=record['exact_source_filename'], license_claim=text(record['license_claim']), duration_seconds=round(pcm['frames'] / pcm['sample_rate'], 6), sample_rate=pcm['sample_rate'], channels=pcm['channels'], bits_per_sample=pcm['bits_per_sample'], bytes=record['download_receipt']['bytes'], source_sha256=record['source_sha256'], pcm_sha256=pcm['decoded_sha256'], source_page=record['source_page'], audio_relative_path=record['audio_relative_path'], flags=' | '.join(record['surface_flags'])))
    import io
    csv_stream = io.StringIO(newline='')
    writer = csv.DictWriter(csv_stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic(batch / 'catalog.csv', ('\ufeff' + csv_stream.getvalue()).encode('utf-8'))
    atomic(batch / 'catalog.json', canonical(records))
    total = sum((record['download_receipt']['bytes'] for record in records))
    summary = {'status': 'target_met' if len(records) >= target else 'in_progress_or_partial', 'target': target, 'distinct_pcm_recordings': len(records), 'stored_audio_bytes': total, 'this_process_transfer_bytes': transferred, 'source_releases': len({r['archive_identifier'] for r in records}), 'primary_inference_eligible': False, 'updated_at_utc': now()}
    atomic(batch / 'summary.json', canonical(summary))
    head = '<!doctype html><meta charset="utf-8"><title>Song acquisition catalog</title><style>body{font:15px system-ui;margin:24px;background:#f7f8fa;color:#20252b}table{border-collapse:collapse;background:white}th,td{padding:9px;border:1px solid #dde1e6;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#e9eef5}small{color:#536071}</style>'
    page = [head, '<h1>Downloaded recordings and surface evidence</h1>', f"<p>{len(records)} distinct decoded recordings from {summary['source_releases']} releases; {total / GIB:.2f} GiB.</p>", '<table><thead><tr><th>#</th><th>Recording</th><th>Source and displayed license</th><th>Measured audio</th><th>Evidence / file</th></tr></thead><tbody>']
    for row, record in zip(rows, records):
        label = row['title_tag'] or row['filename']
        evidence_link = urllib.parse.quote(record['evidence_relative_path'], safe='/')
        audio_link = urllib.parse.quote(row['audio_relative_path'], safe='/')
        format_label = Path(row['audio_relative_path']).suffix.upper().lstrip('.')
        page.append(f'''<tr><td>{row['number']}</td><td>{html.escape(label)}<br><small>{html.escape(row['artist_tag'] or row['creator_claim'])}</small></td><td><a href="{html.escape(row['source_page'], quote=True)}">{html.escape(row['release_title_claim'])}</a><br>{html.escape(row['license_claim'])}</td><td>{row['duration_seconds']:.2f} s<br>{row['sample_rate']} Hz / {row['bits_per_sample']} bit / {row['channels']} channels</td><td><a href="{evidence_link}">Evidence JSON</a> · <a href="{audio_link}">{format_label}</a><br><small>{html.escape(row['flags'])}</small></td></tr>''')
    page.append('</tbody></table>')
    atomic(batch / 'catalog.html', '\n'.join(page).encode('utf-8'))
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=300)
    parser.add_argument('--batch', default='2026-10-03')
    parser.add_argument('--budget-gib', type=int, default=30)
    parser.add_argument('--max-items', type=int, default=1000)
    parser.add_argument('--max-file-mib', type=int, default=75)
    args = parser.parse_args()
    if not 1 <= args.target <= 9999 or not re.fullmatch('[0-9]{4}-[0-9]{2}-[0-9]{2}', args.batch):
        parser.error('Invalid batch identifier/target')
    batch = ROOT / 'local-data/song-acquisition' / args.batch
    for name in ('audio', 'records', 'evidence', 'staging', 'rejected'):
        (batch / name).mkdir(parents=True, exist_ok=True)
    client = Client(batch, args.budget_gib * GIB)
    ledger = sqlite3.connect(batch / 'ledger.sqlite3')
    ledger.execute('PRAGMA journal_mode=WAL')
    ledger.execute('PRAGMA synchronous=FULL')
    ledger.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, pcm_sha256 TEXT UNIQUE NOT NULL, record_json TEXT NOT NULL)')
    ledger.execute('CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY, state TEXT NOT NULL, detail_json TEXT NOT NULL)')
    records = [json.loads(row[0]) for row in ledger.execute('SELECT record_json FROM records ORDER BY rowid')]
    known_pcm = {r['pcm_measurements']['decoded_sha256']: batch / r['audio_relative_path'] for r in records}
    known_files = {r['source_sha256'] for r in records}
    for record in records:
        path = batch / record['audio_relative_path']
        if audit.file_hash(path) != record['source_sha256']:
            raise ValueError('Previously committed batch audio changed')
    old_db = ROOT / '.chemosound/state.sqlite3'
    if old_db.exists():
        with sqlite3.connect(f'{old_db.as_uri()}?mode=ro', uri=True) as conn:
            for path, source_sha, pcm_sha in conn.execute('SELECT absolute_path,source_sha256,decoded_sha256 FROM songs'):
                path = Path(path)
                if path.is_file():
                    actual_source_sha = audit.file_hash(path)
                    actual_pcm_sha = audit.pcm_audit(path)['decoded_sha256']
                    known_files.add(actual_source_sha)
                    known_pcm.setdefault(actual_pcm_sha, path)
    policy = json.loads(audit.POLICY.read_text(encoding='utf-8'))
    flac = ROOT / '.chemosound/tools/flac-1.5.0' / policy['reference_flac']['executable_relative_path']
    package = flac.parents[2] / 'flac-1.5.0-win.zip'
    if audit.file_hash(package) != policy['reference_flac']['package_sha256']:
        raise ValueError('Unverified reference decoder package')
    import zipfile
    with zipfile.ZipFile(package) as archive:
        if flac.read_bytes() != archive.read(policy['reference_flac']['executable_relative_path']):
            raise ValueError('Reference executable differs from pinned package')
    source_query = 'collection:netlabels AND mediatype:audio AND format:Flac AND (' + ' OR '.join((f'licenseurl:"{url}"' for url in LICENSES)) + ')'
    client.budget -= sum((record['download_receipt']['bytes'] for record in records))
    run_policy = {'target': args.target, 'audio_budget_bytes': args.budget_gib * GIB, 'per_item_limit': 8, 'maximum_parallel_audio_downloads': 4, 'per_file_max_bytes': args.max_file_mib * 1024 ** 2, 'selection': 'downloads-desc search; exact filename order', 'query': source_query, 'source_license_allowlist': LICENSES, 'primary_inference_eligible': False, 'code_sha256': audit.file_hash(Path(__file__)), 'captured_at_utc': now()}
    policy_bytes = canonical(run_policy)
    run_policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    atomic(batch / 'evidence' / run_policy_sha, policy_bytes)
    atomic(batch / 'evidence' / run_policy['code_sha256'], Path(__file__).read_bytes())
    atomic(batch / 'batch-policy.json', policy_bytes)
    export_reports(batch, records, target=args.target, transferred=client.transferred)

    def decision(key, state, details):
        detail = canonical({'state': state, 'details': details, 'captured_at_utc': now()}).decode('utf-8')
        ledger.execute('INSERT OR REPLACE INTO decisions VALUES(?,?,?)', (key, state, detail))
        ledger.commit()
    scanned = 0
    seen_items = set()
    release_counts = {}
    for r in records:
        release_counts[r['archive_identifier']] = release_counts.get(r['archive_identifier'], 0) + 1
    for page in range(1, 11):
        if len(records) >= args.target or scanned >= args.max_items:
            break
        search = 'https://archive.org/advancedsearch.php?' + urllib.parse.urlencode({'q': source_query, 'fl[]': ['identifier', 'title', 'creator', 'licenseurl'], 'sort[]': ['downloads desc', 'identifier asc'], 'rows': 100, 'page': page, 'output': 'json'}, doseq=True)
        body, search_receipt = client.evidence(search)
        document = json.loads(body)
        for entry in document['response']['docs']:
            if len(records) >= args.target or scanned >= args.max_items:
                break
            identifier = entry['identifier']
            if identifier in seen_items:
                continue
            seen_items.add(identifier)
            scanned += 1
            if release_counts.get(identifier, 0) >= 8:
                continue
            print(json.dumps({'event': 'inspect_release', 'identifier': identifier, 'downloaded': len(records), 'scanned': scanned}), flush=True)
            try:
                source_page = 'https://archive.org/details/' + urllib.parse.quote(identifier, safe='')
                metadata_url = 'https://archive.org/metadata/' + urllib.parse.quote(identifier, safe='')
                raw, metadata_receipt = client.evidence(metadata_url)
                item = json.loads(raw)
                metadata = item.get('metadata', {})
                licenses = metadata.get('licenseurl')
                license_list = licenses if isinstance(licenses, list) else [licenses]
                if not license_list or any((license not in LICENSES for license in license_list)):
                    decision('item:' + identifier, 'excluded_license_surface', metadata)
                    continue
                if re.search('\\b(audio sources|sample pack|sample library|multitrack stems|audio sessions)\\b', text(metadata.get('title')), re.I):
                    decision('item:' + identifier, 'excluded_nonfinal_package_surface', metadata)
                    continue
                try:
                    _, page_receipt = client.evidence(source_page, limit=8 * 1024 ** 2)
                except Exception as exc:
                    page_receipt = {'source_url': source_page, 'error': str(exc), 'captured_at_utc': now()}
                planned_files = []
                for file in sorted(item.get('files', []), key=lambda f: f['name']):
                    if not file['name'].lower().endswith(('.flac', '.wav')) or file.get('source') != 'original':
                        continue
                    key = hashlib.sha256(canonical([identifier, file['name']])).hexdigest()
                    if ledger.execute('SELECT 1 FROM records WHERE id=?', (key,)).fetchone():
                        continue
                    previous = ledger.execute('SELECT state FROM decisions WHERE id=?', (key,)).fetchone()
                    if previous and previous[0] in {'exact_duplicate', 'excluded_size', 'excluded_nonfinal_file', 'excluded_decode', 'excluded_transfer'}:
                        continue
                    size = int(file.get('size', 0))
                    if not 0 < size <= args.max_file_mib * 1024 ** 2 or not re.fullmatch('[0-9a-f]{32}', file.get('md5', '')):
                        decision(key, 'excluded_size', file)
                        continue
                    if re.search('(^|[/\\\\])(stems?|samples?|multitracks?)([/\\\\]|$)', file['name'], re.I):
                        decision(key, 'excluded_nonfinal_file', file)
                        continue
                    planned_files.append(file)
                planned_files = planned_files[:min(8 - release_counts.get(identifier, 0), args.target - len(records))]
                for file, pending_download in prefetched_downloads(planned_files, client, batch, identifier):
                    if len(records) >= args.target or release_counts.get(identifier, 0) >= 8:
                        break
                    if not file['name'].lower().endswith(('.flac', '.wav')) or file.get('source') != 'original':
                        continue
                    key = hashlib.sha256(canonical([identifier, file['name']])).hexdigest()
                    if ledger.execute('SELECT 1 FROM records WHERE id=?', (key,)).fetchone():
                        continue
                    previous = ledger.execute('SELECT state FROM decisions WHERE id=?', (key,)).fetchone()
                    if previous and previous[0] in {'exact_duplicate', 'excluded_size', 'excluded_nonfinal_file', 'excluded_decode', 'excluded_transfer'}:
                        continue
                    size = int(file.get('size', 0))
                    if not 0 < size <= args.max_file_mib * 1024 ** 2 or not re.fullmatch('[0-9a-f]{32}', file.get('md5', '')):
                        decision(key, 'excluded_size', file)
                        continue
                    if re.search('(^|[/\\\\])(stems?|samples?|multitracks?)([/\\\\]|$)', file['name'], re.I):
                        decision(key, 'excluded_nonfinal_file', file)
                        continue
                    url = 'https://archive.org/download/' + urllib.parse.quote(identifier, safe='') + '/' + urllib.parse.quote(file['name'], safe='')
                    staging = batch / 'staging' / (key + Path(file['name']).suffix.lower())
                    try:
                        print(json.dumps({'event': 'download', 'identifier': identifier, 'filename': file['name'], 'bytes': size, 'completed': len(records)}), flush=True)
                        receipt = pending_download.result()
                        source_sha = receipt['source_sha256']
                        pcm = audit.pcm_audit(staging)
                        if source_sha in known_files or pcm['decoded_sha256'] in known_pcm:
                            match = known_pcm.get(pcm['decoded_sha256'])
                            if match is not None and (not audit.exact_sample_compare(staging, match)):
                                raise ValueError('PCM hash match without sample equality')
                            decision(key, 'exact_duplicate', {'file': file, 'source_sha256': source_sha, 'pcm_sha256': pcm['decoded_sha256']})
                            staging.unlink()
                            continue
                        pcm['container_format'] = audit.sf.info(staging).format
                        flac_exit = None
                        if pcm['container_format'] == 'FLAC':
                            test = subprocess.run([str(flac), '--test', '--warnings-as-errors', '--silent', str(staging)], capture_output=True, text=True, timeout=120)
                            if test.returncode or test.stderr or (not audit.reference_pcm_equal(staging, flac)):
                                raise ValueError('Reference FLAC integrity/native decode comparison failed: ' + test.stderr)
                            flac_exit = test.returncode
                            comments = flac_comments(staging)
                            reference_name = 'Xiph FLAC 1.5.0 raw decoder'
                        else:
                            verify_native_wav(staging)
                            comments = wav_comments(staging)
                            reference_name = 'Python stdlib wave raw PCM reader'
                        surface = text(metadata.get('description')) + '\n' + text(metadata.get('title')) + '\n' + file['name'] + '\n' + '\n'.join(comments['comments'])
                        flags = sorted(set((word.lower() for word in re.findall('\\b(remix|cover|mashup|samples?|live|mix|edit|stem|preview|excerpt|remaster)\\b', surface, re.I))))
                        flags.extend(['master_history_unverified', 'work_relationships_unverified', 'rights_authority_unverified'])
                        safe_name = re.sub('[<>:"/\\\\|?*\\x00-\\x1f]', '_', Path(file['name']).name)[:120].rstrip('. ')
                        relative = 'audio/' + key[:16] + '-' + safe_name
                        target = batch / relative
                        os.replace(staging, target)
                        record = {'schema': 'chemosound-surface-observation-v1', 'record_id': key, 'acquisition_run_policy_sha256': run_policy_sha, 'archive_identifier': identifier, 'source_page': source_page, 'exact_source_filename': file['name'], 'source_file_record_claims': file, 'item_metadata_claims': metadata, 'license_claim': licenses, 'metadata_response_receipt': metadata_receipt, 'source_page_receipt': page_receipt, 'discovery_receipt': search_receipt, 'download_receipt': receipt, 'source_sha256': source_sha, 'pcm_measurements': pcm, 'reference_flac_integrity_exit': flac_exit, 'native_reference_reader': reference_name, 'reference_native_pcm_bytes_equal': True, 'embedded_tags': comments, 'surface_flags': flags, 'audio_relative_path': relative, 'evidence_relative_path': 'records/' + key + '.json', 'primary_inference_eligible': False, 'unverified': ['authentic_master_history', 'complete_authored_boundaries', 'rightsholder_authority', 'compositions_and_covers', 'transformed_duplicates_and_shared_samples']}
                        atomic(batch / record['evidence_relative_path'], canonical(record))
                        ledger.execute('INSERT INTO records VALUES(?,?,?)', (key, pcm['decoded_sha256'], canonical(record).decode('utf-8')))
                        ledger.commit()
                        records.append(record)
                        known_files.add(source_sha)
                        known_pcm[pcm['decoded_sha256']] = target
                        release_counts[identifier] = release_counts.get(identifier, 0) + 1
                        status = export_reports(batch, records, target=args.target, transferred=client.transferred)
                        print(json.dumps({'event': 'committed', **status, 'last_file': file['name']}), flush=True)
                    except RuntimeError as exc:
                        if str(exc) in {'batch_transfer_budget_reached', 'insufficient_free_space', 'source_rate_limited'}:
                            export_reports(batch, records, target=args.target, transferred=client.transferred)
                            print(json.dumps({'event': 'stopped_resource_limit', 'reason': str(exc)}), flush=True)
                            return 2
                        decision(key, 'failed_transfer', {'file': file, 'error': str(exc)})
                        print(json.dumps({'event': 'file_failed', 'filename': file['name'], 'error': str(exc)}), flush=True)
                    except Exception as exc:
                        decision(key, 'excluded_decode' if isinstance(exc, ValueError) else 'failed_transfer', {'file': file, 'error': str(exc)})
                        if staging.exists():
                            os.replace(staging, batch / 'rejected' / (key + staging.suffix))
                        print(json.dumps({'event': 'file_failed', 'filename': file['name'], 'error': str(exc)}), flush=True)
            except Exception as exc:
                decision('item:' + identifier, 'failed_item', {'error': str(exc)})
                print(json.dumps({'event': 'item_failed', 'identifier': identifier, 'error': str(exc)}), flush=True)
    summary = export_reports(batch, records, target=args.target, transferred=client.transferred)
    decisions = dict(ledger.execute('SELECT state,count(*) FROM decisions GROUP BY state').fetchall())
    print(json.dumps({'event': 'batch_complete', **summary, 'decisions': decisions, 'batch_path': str(batch)}, indent=2), flush=True)
    return 0 if len(records) >= args.target else 2
if __name__ == '__main__':
    raise SystemExit(main())
