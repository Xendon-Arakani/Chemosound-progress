from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from .db import event, immediate, utc_now
from .model import CancellationToken, Cancelled, Paths
from .relations import RelationConfig, compare_relations, compute_music_pairs

@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    run_id: str
    completed_this_run: int
    failed_this_run: int
    failed_total: int
    pending: int
    cancelled: bool
    exhausted: bool
    pair_count: int
    result_path: str | None

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def _eligible_songs(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = connection.execute("SELECT s.id, s.ordinal, s.absolute_path, s.relative_path, s.size_bytes, s.mtime_ns, s.decoded_sha256, s.title, s.artist FROM songs s JOIN work_items w ON w.song_id=s.id WHERE w.stage='validate_audio_v1' AND w.status='complete' AND s.missing=0 AND s.duplicate_of IS NULL AND s.rights_status='valid' ORDER BY s.ordinal").fetchall()
    return list(rows)

def _input_identity(rows: list[sqlite3.Row]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        document = {'ordinal': int(row['ordinal']), 'decoded_sha256': row['decoded_sha256']}
        digest.update(json.dumps(document, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def prepare_music_pilot(connection: sqlite3.Connection, config: RelationConfig) -> str:
    config.validate()
    songs = _eligible_songs(connection)
    if not songs:
        raise RuntimeError('no unique, rights-valid, validated songs are available')
    input_sha256 = _input_identity(songs)
    config_sha256 = config.sha256()
    run_id = f'exp-{input_sha256[:16]}-{config_sha256[:12]}'
    now = utc_now()
    with immediate(connection):
        connection.execute("INSERT OR IGNORE INTO analysis_runs(id, kind, protocol_version, representation_version, config_json, config_sha256, input_set_sha256, input_song_count, status, created_at) VALUES(?, 'exploratory', ?, ?, ?, ?, ?, ?, 'pending', ?)", (run_id, config.protocol_version, config.pair_grid_version, config.canonical_json(), config_sha256, input_sha256, len(songs), now))
        for song in songs:
            connection.execute("INSERT OR IGNORE INTO song_representations(run_id, song_id, status, source_decoded_sha256, updated_at) VALUES(?, ?, 'pending', ?, ?)", (run_id, song['id'], song['decoded_sha256'], now))
        event(connection, 'analysis_run_prepared', detail={'run_id': run_id, 'kind': 'exploratory', 'songs': len(songs), 'input_set_sha256': input_sha256, 'config_sha256': config_sha256})
    return run_id

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

def recover_analysis(connection: sqlite3.Connection, paths: Paths, run_id: str) -> int:
    rows = connection.execute("SELECT song_id, attempt_id, artifact_path FROM song_representations WHERE run_id=? AND status='processing' ORDER BY song_id", (run_id,)).fetchall()
    for row in rows:
        attempt_id = str(row['attempt_id'] or '')
        if row['artifact_path']:
            _delete_if_attempt(Path(row['artifact_path']), attempt_id)
        for partial in paths.staging.glob(f'{attempt_id}.relation.json.partial'):
            partial.unlink(missing_ok=True)
        with immediate(connection):
            connection.execute("UPDATE song_representations SET status='pending', attempt_id=NULL, started_at=NULL, artifact_path=NULL, artifact_sha256=NULL, last_error='recovered after interrupted process', updated_at=? WHERE run_id=? AND song_id=? AND status='processing'", (utc_now(), run_id, row['song_id']))
            event(connection, 'analysis_work_recovered', song_id=int(row['song_id']), attempt_id=attempt_id, detail={'run_id': run_id})
    return len(rows)

def _audit_completed(connection: sqlite3.Connection, run_id: str) -> int:
    rows = connection.execute("SELECT song_id, artifact_path, artifact_sha256 FROM song_representations WHERE run_id=? AND status='complete' ORDER BY song_id", (run_id,)).fetchall()
    damaged: list[int] = []
    for row in rows:
        path = Path(row['artifact_path']) if row['artifact_path'] else None
        if path is None or not path.is_file() or (not row['artifact_sha256']) or (_file_sha256(path) != row['artifact_sha256']):
            damaged.append(int(row['song_id']))
    for song_id in damaged:
        with immediate(connection):
            connection.execute("UPDATE song_representations SET status='pending', attempt_id=NULL, started_at=NULL, completed_at=NULL, artifact_path=NULL, artifact_sha256=NULL, last_error='artifact failed startup audit', updated_at=? WHERE run_id=? AND song_id=?", (utc_now(), run_id, song_id))
            event(connection, 'analysis_artifact_requeued', song_id=song_id, detail={'run_id': run_id})
    return len(damaged)

def _claim_next(connection: sqlite3.Connection, run_id: str) -> dict | None:
    with immediate(connection):
        row = connection.execute("SELECT r.song_id, r.source_decoded_sha256, s.ordinal, s.absolute_path, s.relative_path, s.size_bytes, s.mtime_ns, s.title, s.artist FROM song_representations r JOIN songs s ON s.id=r.song_id WHERE r.run_id=? AND r.status='pending' ORDER BY s.ordinal LIMIT 1", (run_id,)).fetchone()
        if row is None:
            return None
        attempt_id = uuid.uuid4().hex
        now = utc_now()
        connection.execute("UPDATE song_representations SET status='processing', attempt_id=?, attempts=attempts+1, started_at=?, completed_at=NULL, artifact_path=NULL, artifact_sha256=NULL, last_error=NULL, updated_at=? WHERE run_id=? AND song_id=? AND status='pending'", (attempt_id, now, now, run_id, row['song_id']))
        connection.execute("UPDATE analysis_runs SET status='processing', last_error=NULL WHERE id=?", (run_id,))
        event(connection, 'analysis_work_claimed', song_id=int(row['song_id']), attempt_id=attempt_id, detail={'run_id': run_id})
        result = dict(row)
        result['attempt_id'] = attempt_id
        return result

def _write_pair_grid(paths: Paths, run_id: str, song: dict, representation: dict) -> tuple[Path, str]:
    run_directory = paths.relation_results / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    final_path = run_directory / f"{int(song['song_id']):08d}.json"
    partial_path = paths.staging / f"{song['attempt_id']}.relation.json.partial"
    document = {'schema_version': 1, 'run_id': run_id, 'attempt_id': song['attempt_id'], 'song': {'id': int(song['song_id']), 'ordinal': int(song['ordinal']), 'relative_path': song['relative_path'], 'decoded_sha256': song['source_decoded_sha256'], 'title': song['title'], 'artist': song['artist']}, 'representation': representation}
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with partial_path.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial_path, final_path)
    return (final_path, _file_sha256(final_path))

def _mark_complete(connection: sqlite3.Connection, run_id: str, song: dict, artifact_path: Path, artifact_sha256: str) -> None:
    with immediate(connection):
        row = connection.execute('SELECT status, attempt_id FROM song_representations WHERE run_id=? AND song_id=?', (run_id, song['song_id'])).fetchone()
        if row is None or row['status'] != 'processing' or row['attempt_id'] != song['attempt_id']:
            raise RuntimeError('analysis work claim changed before commit')
        now = utc_now()
        connection.execute("UPDATE song_representations SET status='complete', completed_at=?, artifact_path=?, artifact_sha256=?, last_error=NULL, updated_at=? WHERE run_id=? AND song_id=?", (now, str(artifact_path), artifact_sha256, now, run_id, song['song_id']))
        event(connection, 'analysis_work_committed', song_id=int(song['song_id']), attempt_id=str(song['attempt_id']), detail={'run_id': run_id})

def _return_pending(connection: sqlite3.Connection, run_id: str, song: dict, reason: str) -> None:
    with immediate(connection):
        connection.execute("UPDATE song_representations SET status='pending', attempt_id=NULL, started_at=NULL, artifact_path=NULL, artifact_sha256=NULL, last_error=?, updated_at=? WHERE run_id=? AND song_id=? AND attempt_id=?", (reason, utc_now(), run_id, song['song_id'], song['attempt_id']))
        event(connection, 'analysis_work_cancelled', song_id=int(song['song_id']), attempt_id=str(song['attempt_id']), detail={'run_id': run_id, 'reason': reason})

def _mark_failed(connection: sqlite3.Connection, run_id: str, song: dict, error: BaseException) -> None:
    message = f'{type(error).__name__}: {error}'
    with immediate(connection):
        now = utc_now()
        connection.execute("UPDATE song_representations SET status='failed', completed_at=?, last_error=?, updated_at=? WHERE run_id=? AND song_id=? AND attempt_id=?", (now, message, now, run_id, song['song_id'], song['attempt_id']))
        connection.execute("UPDATE analysis_runs SET status='failed', last_error=? WHERE id=?", (message, run_id))
        event(connection, 'analysis_work_failed', song_id=int(song['song_id']), attempt_id=str(song['attempt_id']), detail={'run_id': run_id, 'error': message})

def _load_artifacts(connection: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = connection.execute("SELECT r.song_id, r.artifact_path, r.artifact_sha256, s.ordinal FROM song_representations r JOIN songs s ON s.id=r.song_id WHERE r.run_id=? AND r.status='complete' ORDER BY s.ordinal", (run_id,)).fetchall()
    documents: list[dict] = []
    for row in rows:
        path = Path(row['artifact_path'])
        if _file_sha256(path) != row['artifact_sha256']:
            raise RuntimeError(f'analysis artifact changed: {path}')
        documents.append(json.loads(path.read_text(encoding='utf-8')))
    return documents

def _write_similarity(paths: Paths, run_id: str, document: dict) -> tuple[Path, str]:
    final_path = paths.similarity_results / f'{run_id}.json'
    partial_path = paths.staging / f'{run_id}.similarity.json.partial'
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with partial_path.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial_path, final_path)
    return (final_path, _file_sha256(final_path))

def _finish_similarities(connection: sqlite3.Connection, paths: Paths, run_id: str) -> tuple[int, Path]:
    artifacts = _load_artifacts(connection, run_id)
    pairs: list[dict] = []
    for left_index, left in enumerate(artifacts):
        for right in artifacts[left_index + 1:]:
            metrics = compare_relations(left['representation'], right['representation'])
            pairs.append({'left_song_id': left['song']['id'], 'left_title': left['song']['title'], 'right_song_id': right['song']['id'], 'right_title': right['song']['title'], **metrics})
    pairs.sort(key=lambda item: item['mean_wasserstein'])
    result = {'schema_version': 1, 'run_id': run_id, 'exploratory_only': True, 'inference_permitted': False, 'reason': 'The music corpus is not frozen, n is small, no HITRAN representation is present, and the music measure uses recorded log-group compression.', 'song_count': len(artifacts), 'pair_count': len(pairs), 'pairs': pairs}
    result_path, result_sha256 = _write_similarity(paths, run_id, result)
    with immediate(connection):
        connection.execute('DELETE FROM similarity_pairs WHERE run_id=?', (run_id,))
        for pair in pairs:
            connection.execute('INSERT INTO similarity_pairs(run_id, left_song_id, right_song_id, metric_version, mean_wasserstein, mean_hellinger_affinity, scale_distances_json, detail_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)', (run_id, pair['left_song_id'], pair['right_song_id'], pair['metric_version'], pair['mean_wasserstein'], pair['mean_hellinger_affinity'], json.dumps(pair['scales'], sort_keys=True), json.dumps({'mean_normalized_wasserstein': pair['mean_normalized_wasserstein'], 'left_title': pair['left_title'], 'right_title': pair['right_title']}, sort_keys=True)))
        now = utc_now()
        connection.execute("UPDATE analysis_runs SET status='complete', result_path=?, result_sha256=?, last_error=NULL, completed_at=? WHERE id=?", (str(result_path), result_sha256, now, run_id))
        event(connection, 'analysis_run_completed', detail={'run_id': run_id, 'pairs': len(pairs)})
    return (len(pairs), result_path)

def run_music_pilot(connection: sqlite3.Connection, paths: Paths, token: CancellationToken, config: RelationConfig | None=None) -> AnalysisSummary:
    config = config or RelationConfig()
    paths.ensure()
    run_id = prepare_music_pilot(connection, config)
    recover_analysis(connection, paths, run_id)
    _audit_completed(connection, run_id)
    completed = failed = 0
    while not token.cancelled:
        song = _claim_next(connection, run_id)
        if song is None:
            break
        final_path: Path | None = None
        try:
            source = Path(song['absolute_path'])
            stat = source.stat()
            if stat.st_size != song['size_bytes'] or stat.st_mtime_ns != song['mtime_ns']:
                raise RuntimeError('audio changed after validation; rescan before analysis')
            representation = compute_music_pairs(source, config, token)
            token.checkpoint()
            final_path, artifact_sha256 = _write_pair_grid(paths, run_id, song, representation)
            token.checkpoint()
            _mark_complete(connection, run_id, song, final_path, artifact_sha256)
            completed += 1
        except Cancelled as exc:
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _return_pending(connection, run_id, song, str(exc))
            break
        except (KeyboardInterrupt, SystemExit):
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _return_pending(connection, run_id, song, 'process interrupted')
            raise
        except Exception as exc:
            if final_path is not None:
                _delete_if_attempt(final_path, str(song['attempt_id']))
            _mark_failed(connection, run_id, song, exc)
            failed += 1
    counts = {row['status']: int(row['n']) for row in connection.execute('SELECT status, COUNT(*) AS n FROM song_representations WHERE run_id=? GROUP BY status', (run_id,))}
    pending = counts.get('pending', 0)
    failed_total = counts.get('failed', 0)
    pair_count = 0
    result_path: Path | None = None
    if pending == 0 and failed_total == 0 and (not token.cancelled):
        pair_count, result_path = _finish_similarities(connection, paths, run_id)
    return AnalysisSummary(run_id=run_id, completed_this_run=completed, failed_this_run=failed, failed_total=failed_total, pending=pending, cancelled=token.cancelled, exhausted=pending == 0, pair_count=pair_count, result_path=str(result_path) if result_path else None)

def analysis_status(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute("SELECT r.*, (SELECT COUNT(*) FROM song_representations s WHERE s.run_id=r.id AND s.status='complete') AS songs_complete, (SELECT COUNT(*) FROM similarity_pairs p WHERE p.run_id=r.id) AS pairs FROM analysis_runs r ORDER BY r.created_at").fetchall()
    return [{'run_id': row['id'], 'kind': row['kind'], 'representation_version': row['representation_version'], 'status': row['status'], 'input_song_count': int(row['input_song_count']), 'songs_complete': int(row['songs_complete']), 'pairs': int(row['pairs']), 'input_set_sha256': row['input_set_sha256'], 'config_sha256': row['config_sha256'], 'result_path': row['result_path'], 'last_error': row['last_error']} for row in rows]
