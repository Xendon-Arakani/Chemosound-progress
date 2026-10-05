from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from .audio import ValidationResult, validate_audio
from .db import event, get_meta, immediate, set_meta, utc_now
from .model import AUDIO_EXTENSIONS, CancellationToken, Cancelled, Paths
from .rights import load_rights
VALIDATION_STAGE = 'validate_audio_v1'

@dataclass(frozen=True, slots=True)
class RunSummary:
    completed: int
    failed_this_run: int
    failed_total: int
    cancelled: bool
    exhausted: bool

def _is_frozen(connection: sqlite3.Connection) -> bool:
    return get_meta(connection, 'corpus_frozen', '0') == '1'

def add_root(connection: sqlite3.Connection, root: Path) -> None:
    if _is_frozen(connection):
        raise RuntimeError('corpus is frozen; audio roots cannot be changed')
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    with immediate(connection):
        connection.execute('INSERT OR IGNORE INTO roots(path, created_at) VALUES(?, ?)', (str(resolved), utc_now()))
        event(connection, 'root_added', detail={'path': str(resolved)})

def refresh_duplicates(connection: sqlite3.Connection) -> None:
    with immediate(connection):
        connection.execute('UPDATE songs SET duplicate_of=NULL')
        rows = connection.execute("SELECT s.id, s.source_sha256, s.decoded_sha256 FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage=? AND w.status='complete' AND s.missing=0 ORDER BY s.ordinal", (VALIDATION_STAGE,)).fetchall()
        sources = {}
        decoded = {}
        for row in rows:
            source = row['source_sha256']
            pcm = row['decoded_sha256']
            if not source or not pcm:
                raise RuntimeError('complete validation item lacks an identity hash')
            parent = sources.get(source, decoded.get(pcm))
            if parent is not None:
                connection.execute('UPDATE songs SET duplicate_of=? WHERE id=?', (parent, row['id']))
            else:
                parent = int(row['id'])
            sources[source] = parent
            decoded[pcm] = parent

def scan(connection: sqlite3.Connection, extensions=None) -> dict[str, int]:
    if _is_frozen(connection):
        raise RuntimeError('corpus is frozen; scanning new or changed audio is refused')
    extensions = AUDIO_EXTENSIONS if extensions is None else frozenset(extensions)
    roots = connection.execute('SELECT id, path FROM roots ORDER BY id').fetchall()
    if not roots:
        raise RuntimeError('no audio roots configured; run init --audio-root PATH')
    discovered: list[tuple[int, Path, str]] = []
    for root in roots:
        root_path = Path(root['path'])
        if not root_path.is_dir():
            continue
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.casefold() in extensions:
                resolved = path.resolve()
                discovered.append((int(root['id']), resolved, str(resolved.relative_to(root_path))))
    discovered.sort(key=lambda item: str(item[1]).casefold())
    now = utc_now()
    added = changed = unchanged = 0
    seen_paths: set[str] = set()
    for root_id, path, relative in discovered:
        path_text = str(path)
        seen_paths.add(path_text)
        stat = path.stat()
        current_rights = load_rights(path)
        with immediate(connection):
            row = connection.execute('SELECT id, size_bytes, mtime_ns, source_sha256, rights_status, rights_sha256 FROM songs WHERE absolute_path = ?', (path_text,)).fetchone()
            if row is None:
                ordinal = int(connection.execute('SELECT COALESCE(MAX(ordinal), 0) + 1 AS value FROM songs').fetchone()['value'])
                cursor = connection.execute('INSERT INTO songs(ordinal, root_id, absolute_path, relative_path, size_bytes, mtime_ns, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)', (ordinal, root_id, path_text, relative, stat.st_size, stat.st_mtime_ns, now, now))
                song_id = int(cursor.lastrowid)
                connection.execute("INSERT INTO work_items(song_id, stage, status, updated_at) VALUES(?, ?, 'pending', ?)", (song_id, VALIDATION_STAGE, now))
                event(connection, 'song_discovered', song_id=song_id, detail={'path': path_text})
                added += 1
            elif row['size_bytes'] != stat.st_size or row['mtime_ns'] != stat.st_mtime_ns or row['rights_status'] != current_rights.status or (row['rights_sha256'] != current_rights.sha256) or (row['source_sha256'] and _result_digest(path) != row['source_sha256']):
                song_id = int(row['id'])
                connection.execute("UPDATE songs SET size_bytes=?, mtime_ns=?, missing=0, source_sha256=NULL, decoded_sha256=NULL, format=NULL, subtype=NULL, sample_rate=NULL, channels=NULL, frames=NULL, duration_seconds=NULL, rights_status='unchecked', rights_sha256=NULL, rights_json=NULL, duplicate_of=NULL, updated_at=? WHERE id=?", (stat.st_size, stat.st_mtime_ns, now, song_id))
                connection.execute("UPDATE work_items SET status='pending', attempt_id=NULL, started_at=NULL, completed_at=NULL, result_path=NULL, result_sha256=NULL, last_error=NULL, updated_at=? WHERE song_id=? AND stage=?", (now, song_id, VALIDATION_STAGE))
                event(connection, 'song_changed', song_id=song_id, detail={'path': path_text})
                changed += 1
            else:
                connection.execute('UPDATE songs SET missing=0, updated_at=? WHERE id=?', (now, row['id']))
                unchanged += 1
    with immediate(connection):
        known = connection.execute('SELECT id, absolute_path FROM songs WHERE missing=0').fetchall()
        missing_ids = [int(row['id']) for row in known if row['absolute_path'] not in seen_paths]
        for song_id in missing_ids:
            connection.execute('UPDATE songs SET missing=1, updated_at=? WHERE id=?', (now, song_id))
            event(connection, 'song_missing', song_id=song_id)
    refresh_duplicates(connection)
    return {'added': added, 'changed': changed, 'unchanged': unchanged, 'missing': len(missing_ids)}

def _result_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def _delete_if_attempt(path: Path, attempt_id: str) -> None:
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return
    if document.get('attempt_id') == attempt_id:
        path.unlink(missing_ok=True)

def recover_interrupted(connection: sqlite3.Connection, paths: Paths) -> int:
    rows = connection.execute("SELECT song_id, attempt_id, result_path FROM work_items WHERE stage=? AND status='processing' ORDER BY song_id", (VALIDATION_STAGE,)).fetchall()
    for row in rows:
        attempt_id = str(row['attempt_id'] or '')
        published = Path(row['result_path']) if row['result_path'] else paths.validation_results / f"{int(row['song_id']):08d}.json"
        _delete_if_attempt(published, attempt_id)
        for partial in paths.staging.glob(f'{attempt_id}.*'):
            partial.unlink(missing_ok=True)
        with immediate(connection):
            connection.execute("UPDATE work_items SET status='pending', attempt_id=NULL, started_at=NULL, result_path=NULL, result_sha256=NULL, last_error='recovered after interrupted process', updated_at=? WHERE song_id=? AND stage=? AND status='processing'", (utc_now(), row['song_id'], VALIDATION_STAGE))
            event(connection, 'work_recovered', song_id=int(row['song_id']), attempt_id=attempt_id)
    return len(rows)

def audit_completed(connection: sqlite3.Connection) -> int:
    rows = connection.execute("SELECT song_id, result_path, result_sha256 FROM work_items WHERE stage=? AND status='complete' ORDER BY song_id", (VALIDATION_STAGE,)).fetchall()
    damaged: list[int] = []
    for row in rows:
        result_path = Path(row['result_path']) if row['result_path'] else None
        if result_path is None or not result_path.is_file() or (not row['result_sha256']) or (_result_digest(result_path) != row['result_sha256']):
            damaged.append(int(row['song_id']))
    for song_id in damaged:
        with immediate(connection):
            connection.execute("UPDATE work_items SET status='pending', attempt_id=NULL, started_at=NULL, completed_at=NULL, result_path=NULL, result_sha256=NULL, last_error='committed result artifact failed startup audit', updated_at=? WHERE song_id=? AND stage=? AND status='complete'", (utc_now(), song_id, VALIDATION_STAGE))
            event(connection, 'completed_result_requeued', song_id=song_id)
    return len(damaged)

def _claim_next(connection: sqlite3.Connection) -> sqlite3.Row | None:
    with immediate(connection):
        row = connection.execute("SELECT s.id, s.ordinal, s.absolute_path, s.size_bytes, s.mtime_ns FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage=? AND w.status='pending' AND s.missing=0 ORDER BY s.ordinal LIMIT 1", (VALIDATION_STAGE,)).fetchone()
        if row is None:
            return None
        attempt_id = uuid.uuid4().hex
        now = utc_now()
        connection.execute("UPDATE work_items SET status='processing', attempt_id=?, attempts=attempts+1, started_at=?, completed_at=NULL, result_path=NULL, result_sha256=NULL, last_error=NULL, updated_at=? WHERE song_id=? AND stage=? AND status='pending'", (attempt_id, now, now, row['id'], VALIDATION_STAGE))
        event(connection, 'work_claimed', song_id=int(row['id']), attempt_id=attempt_id)
        values = dict(row)
        values['attempt_id'] = attempt_id
        return _dict_row(values)

class _dict_row(dict):
    pass

def _write_result(paths: Paths, song: sqlite3.Row | dict, result: ValidationResult) -> tuple[Path, str]:
    attempt_id = str(song['attempt_id'])
    final_path = paths.validation_results / f"{int(song['id']):08d}.json"
    partial_path = paths.staging / f'{attempt_id}.json.partial'
    document = {'schema_version': 1, 'stage': VALIDATION_STAGE, 'song_id': int(song['id']), 'ordinal': int(song['ordinal']), 'attempt_id': attempt_id, 'absolute_path': str(song['absolute_path']), 'result': result.to_json_dict()}
    encoded = (json.dumps(document, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with partial_path.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial_path, final_path)
    return (final_path, _result_digest(final_path))

def _mark_complete(connection: sqlite3.Connection, song: sqlite3.Row | dict, result: ValidationResult, result_path: Path, result_sha256: str) -> None:
    with immediate(connection):
        current = connection.execute('SELECT status, attempt_id FROM work_items WHERE song_id=? AND stage=?', (song['id'], VALIDATION_STAGE)).fetchone()
        if current is None or current['status'] != 'processing' or current['attempt_id'] != song['attempt_id']:
            raise RuntimeError('work claim changed before commit')
        duplicate = connection.execute("SELECT s.id FROM songs s JOIN work_items w ON w.song_id=s.id WHERE s.id<>? AND w.stage=? AND w.status='complete' AND s.missing=0 AND (s.source_sha256=? OR s.decoded_sha256=?) AND s.duplicate_of IS NULL ORDER BY s.ordinal LIMIT 1", (song['id'], VALIDATION_STAGE, result.source_sha256, result.decoded_sha256)).fetchone()
        duplicate_of = None if duplicate is None else int(duplicate['id'])
        now = utc_now()
        connection.execute('UPDATE songs SET source_sha256=?, decoded_sha256=?, format=?, subtype=?, sample_rate=?, channels=?, frames=?, duration_seconds=?, title=?, artist=?, rights_status=?, rights_sha256=?, rights_json=?, duplicate_of=?, updated_at=? WHERE id=?', (result.source_sha256, result.decoded_sha256, result.format, result.subtype, result.sample_rate, result.channels, result.frames, result.duration_seconds, result.title, result.artist, result.rights_status, result.rights_sha256, json.dumps(result.rights_json, sort_keys=True) if result.rights_json else None, duplicate_of, now, song['id']))
        connection.execute("UPDATE work_items SET status='complete', completed_at=?, result_path=?, result_sha256=?, last_error=NULL, updated_at=? WHERE song_id=? AND stage=?", (now, str(result_path), result_sha256, now, song['id'], VALIDATION_STAGE))
        event(connection, 'work_committed', song_id=int(song['id']), attempt_id=str(song['attempt_id']), detail={'duplicate_of': duplicate_of, 'rights_status': result.rights_status})

def _return_to_pending(connection: sqlite3.Connection, song: sqlite3.Row | dict, reason: str) -> None:
    with immediate(connection):
        connection.execute("UPDATE work_items SET status='pending', attempt_id=NULL, started_at=NULL, result_path=NULL, result_sha256=NULL, last_error=?, updated_at=? WHERE song_id=? AND stage=? AND attempt_id=?", (reason, utc_now(), song['id'], VALIDATION_STAGE, song['attempt_id']))
        event(connection, 'work_cancelled', song_id=int(song['id']), attempt_id=str(song['attempt_id']), detail={'reason': reason})

def _mark_failed(connection: sqlite3.Connection, song: sqlite3.Row | dict, error: BaseException) -> None:
    message = f'{type(error).__name__}: {error}'
    with immediate(connection):
        connection.execute("UPDATE work_items SET status='failed', completed_at=?, last_error=?, updated_at=? WHERE song_id=? AND stage=? AND attempt_id=?", (utc_now(), message, utc_now(), song['id'], VALIDATION_STAGE, song['attempt_id']))
        event(connection, 'work_failed', song_id=int(song['id']), attempt_id=str(song['attempt_id']), detail={'error': message})

def run_validation(connection: sqlite3.Connection, paths: Paths, token: CancellationToken, *, after_publish: Callable[[Path, dict], None] | None=None) -> RunSummary:
    if _is_frozen(connection):
        raise RuntimeError('corpus is frozen; validation inputs cannot be changed')
    paths.ensure()
    audit_completed(connection)
    recover_interrupted(connection, paths)
    refresh_duplicates(connection)
    completed = failed = 0
    while not token.cancelled:
        song = _claim_next(connection)
        if song is None:
            break
        (paths.validation_results / f"{int(song['id']):08d}.json").unlink(missing_ok=True)
        final_path: Path | None = None
        try:
            source_path = Path(song['absolute_path'])
            stat = source_path.stat()
            if stat.st_size != song['size_bytes'] or stat.st_mtime_ns != song['mtime_ns']:
                raise RuntimeError('file changed after scan; rescan before validation')
            result = validate_audio(source_path, token)
            token.checkpoint()
            final_path, result_hash = _write_result(paths, song, result)
            if after_publish is not None:
                after_publish(final_path, song)
            token.checkpoint()
            _mark_complete(connection, song, result, final_path, result_hash)
            completed += 1
        except Cancelled as exc:
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _return_to_pending(connection, song, str(exc))
            break
        except (KeyboardInterrupt, SystemExit):
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _return_to_pending(connection, song, 'process interrupted')
            raise
        except Exception as exc:
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _mark_failed(connection, song, exc)
            failed += 1
    refresh_duplicates(connection)
    pending = int(connection.execute("SELECT COUNT(*) AS n FROM work_items WHERE stage=? AND status='pending'", (VALIDATION_STAGE,)).fetchone()['n'])
    failed_total = int(connection.execute("SELECT COUNT(*) AS n FROM work_items WHERE stage=? AND status='failed'", (VALIDATION_STAGE,)).fetchone()['n'])
    return RunSummary(completed, failed, failed_total, token.cancelled, pending == 0)

def retry_failed(connection: sqlite3.Connection) -> int:
    if _is_frozen(connection):
        raise RuntimeError('corpus is frozen')
    with immediate(connection):
        cursor = connection.execute("UPDATE work_items SET status='pending', attempt_id=NULL, started_at=NULL, completed_at=NULL, last_error=NULL, updated_at=? WHERE stage=? AND status='failed'", (utc_now(), VALIDATION_STAGE))
        count = int(cursor.rowcount)
        event(connection, 'failed_retried', detail={'count': count})
    return count

def status(connection: sqlite3.Connection) -> dict:
    counts = {row['status']: int(row['n']) for row in connection.execute('SELECT status, COUNT(*) AS n FROM work_items WHERE stage=? GROUP BY status', (VALIDATION_STAGE,))}
    accepted = int(connection.execute("SELECT COUNT(*) AS n FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage=? AND w.status='complete' AND s.missing=0 AND s.duplicate_of IS NULL", (VALIDATION_STAGE,)).fetchone()['n'])
    duplicates = int(connection.execute('SELECT COUNT(*) AS n FROM songs WHERE duplicate_of IS NOT NULL').fetchone()['n'])
    rights_missing = int(connection.execute("SELECT COUNT(*) AS n FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage=? AND w.status='complete' AND s.duplicate_of IS NULL AND s.rights_status<>'valid'", (VALIDATION_STAGE,)).fetchone()['n'])
    return {'frozen': _is_frozen(connection), 'manifest_sha256': get_meta(connection, 'manifest_sha256'), 'work': counts, 'accepted_unique_audio': accepted, 'duplicates': duplicates, 'rights_not_valid': rights_missing}

def freeze(connection: sqlite3.Connection, *, max_songs: int=9999) -> tuple[int, str]:
    if _is_frozen(connection):
        existing = get_meta(connection, 'manifest_sha256')
        count = int(get_meta(connection, 'manifest_song_count', '0') or 0)
        if not existing:
            raise RuntimeError('corpus is marked frozen but has no manifest hash')
        return (count, existing)
    audit_completed(connection)
    refresh_duplicates(connection)
    incomplete = int(connection.execute("SELECT COUNT(*) AS n FROM work_items WHERE stage=? AND status<>'complete'", (VALIDATION_STAGE,)).fetchone()['n'])
    if incomplete:
        raise RuntimeError(f'cannot freeze: {incomplete} validation items are incomplete')
    missing = int(connection.execute('SELECT COUNT(*) AS n FROM songs WHERE missing=1').fetchone()['n'])
    if missing:
        raise RuntimeError(f'cannot freeze: {missing} cataloged files are missing')
    rows = connection.execute("SELECT s.id, s.ordinal, s.absolute_path, s.relative_path, s.size_bytes, s.mtime_ns, s.source_sha256, s.decoded_sha256, s.sample_rate, s.channels, s.frames, s.rights_sha256, s.rights_status FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage=? AND w.status='complete' AND s.duplicate_of IS NULL ORDER BY s.ordinal", (VALIDATION_STAGE,)).fetchall()
    if not rows:
        raise RuntimeError('cannot freeze an empty corpus')
    if len(rows) > max_songs:
        raise RuntimeError(f'corpus has {len(rows)} songs; configured maximum is {max_songs}')
    invalid = [row for row in rows if row['rights_status'] != 'valid']
    if invalid:
        raise RuntimeError(f'cannot freeze: {len(invalid)} unique files lack a valid rights record')
    changed_inputs: set[str] = set()
    all_rows = connection.execute('SELECT absolute_path, source_sha256 FROM songs ORDER BY ordinal').fetchall()
    for row in all_rows:
        path = Path(row['absolute_path'])
        if not path.is_file() or _result_digest(path) != row['source_sha256']:
            changed_inputs.add(str(path))
    for row in rows:
        path = Path(row['absolute_path'])
        if not path.is_file():
            changed_inputs.add(str(path))
            continue
        stat = path.stat()
        current_rights = load_rights(path)
        if stat.st_size != row['size_bytes'] or stat.st_mtime_ns != row['mtime_ns'] or current_rights.status != 'valid' or (current_rights.sha256 != row['rights_sha256']):
            changed_inputs.add(str(path))
    if changed_inputs:
        raise RuntimeError(f'cannot freeze: {len(changed_inputs)} audio or rights inputs changed; rescan and rerun')
    manifest_lines = []
    rights_lines = []
    for row in rows:
        manifest_lines.append(json.dumps({'ordinal': row['ordinal'], 'relative_path': row['relative_path'], 'source_sha256': row['source_sha256'], 'decoded_sha256': row['decoded_sha256'], 'sample_rate': row['sample_rate'], 'channels': row['channels'], 'frames': row['frames']}, sort_keys=True, separators=(',', ':')))
        rights_lines.append(json.dumps({'ordinal': row['ordinal'], 'decoded_sha256': row['decoded_sha256'], 'rights_sha256': row['rights_sha256']}, sort_keys=True, separators=(',', ':')))
    manifest = ('\n'.join(manifest_lines) + '\n').encode('utf-8')
    digest = hashlib.sha256(manifest).hexdigest()
    rights_manifest = ('\n'.join(rights_lines) + '\n').encode('utf-8')
    rights_digest = hashlib.sha256(rights_manifest).hexdigest()
    with immediate(connection):
        set_meta(connection, 'corpus_frozen', '1')
        set_meta(connection, 'manifest_sha256', digest)
        set_meta(connection, 'rights_manifest_sha256', rights_digest)
        set_meta(connection, 'manifest_song_count', str(len(rows)))
        set_meta(connection, 'manifest_frozen_at', utc_now())
        event(connection, 'corpus_frozen', detail={'count': len(rows), 'analysis_sha256': digest, 'rights_sha256': rights_digest})
    return (len(rows), digest)
