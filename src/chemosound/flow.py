from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio import validate_audio
from .db import connect, immediate, utc_now
from .gpu_relations import load_molecular_grid
from .model import CancellationToken, Cancelled
from .tensor_engine import ScaleEngine, TensorConfig, measure_labels

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)

def engine_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in ('tensor_engine.py', 'gpu_relations.py', 'cpu_oracle.py', 'flow.py', 'audio.py', 'model.py')}

@contextmanager
def worker_lock(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    path = Path(root) / 'worker.lock'
    with path.open('a+b') as stream:
        stream.seek(0)
        stream.write(b'0')
        stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('another worker holds this run; it must stop before recovery') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

class FileToken(CancellationToken):

    def __init__(self, pause_path):
        super().__init__(CancellationToken.create().event)
        self.pause_path = Path(pause_path)

    @property
    def cancelled(self):
        return self.event.is_set() or self.pause_path.exists()

def open_state(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = connect(root / 'flow.sqlite3')
    db.executescript("\n      CREATE TABLE IF NOT EXISTS identity(key TEXT PRIMARY KEY,value TEXT NOT NULL);\n      CREATE TABLE IF NOT EXISTS inputs(id INTEGER PRIMARY KEY,document TEXT NOT NULL);\n      CREATE TABLE IF NOT EXISTS units(\n        song_id INTEGER NOT NULL REFERENCES inputs(id),scale INTEGER NOT NULL,\n        replicate_start INTEGER NOT NULL,replicate_count INTEGER NOT NULL,\n        status TEXT NOT NULL CHECK(status IN ('pending','processing','complete','failed')),\n        attempt TEXT,attempts INTEGER NOT NULL DEFAULT 0,path TEXT,sha256 TEXT,error TEXT,\n        PRIMARY KEY(song_id,scale,replicate_start));\n      CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at TEXT NOT NULL,\n        kind TEXT NOT NULL,detail TEXT NOT NULL);\n    ")
    return db

def event(db, kind, detail):
    db.execute('INSERT INTO events(at,kind,detail) VALUES(?,?,?)', (utc_now(), kind, canonical(detail).decode()))

def prepare(root, batch, chemistry, *, replicates=2, shard_size=8, quantum=0.005, token=None):
    root, batch, chemistry = (Path(root).resolve(), Path(batch).resolve(), Path(chemistry).resolve())
    if replicates < 1 or shard_size < 1 or replicates > 9999:
        raise ValueError('replicates 1..9999 and positive shard size required')
    token = token or CancellationToken.create()
    config = TensorConfig(coordinate_quantum=quantum)
    config.validate()
    records = json.loads((batch / 'catalog.json').read_text(encoding='utf-8'))
    if len(records) != 300:
        raise ValueError('this rehearsal requires the complete 300-recording batch')
    inputs = []
    for index, record in enumerate(records, 1):
        token.checkpoint()
        source = (batch / record['audio_relative_path']).resolve()
        if not source.is_relative_to(batch / 'audio'):
            raise ValueError('input escaped batch audio directory')
        validated = validate_audio(source, token)
        pcm = record['pcm_measurements']
        if validated.source_sha256 != record['source_sha256'] or validated.decoded_sha256 != pcm['decoded_sha256'] or validated.frames != pcm['frames'] or (validated.channels != pcm['channels']) or (validated.sample_rate != pcm['sample_rate']):
            raise RuntimeError('source receipt mismatch')
        if validated.duration_seconds < max(config.scales_seconds):
            raise ValueError('source shorter than longest scale')
        inputs.append({'id': index, 'path': str(source), 'source_sha256': validated.source_sha256, 'decoded_sha256': validated.decoded_sha256, 'frames': validated.frames, 'channels': validated.channels, 'sample_rate': validated.sample_rate, 'size_bytes': source.stat().st_size, 'acquisition_receipt_sha256': digest(batch / record['evidence_relative_path'])})
        if index % 25 == 0:
            print(json.dumps({'event': 'native_input_audit', 'complete': index, 'target': len(records)}), flush=True)
    if len({x['decoded_sha256'] for x in inputs}) != len(inputs):
        raise ValueError('duplicate input PCM identities')
    molecules = load_molecular_grid(chemistry)
    chemical_files = {'manifests/snapshot-frozen.json': molecules.manifest_sha256}
    version = chemistry / 'representations' / molecules.pair_grid_version
    for path in sorted(version.glob('*.npz')):
        token.checkpoint()
        chemical_files[str(path.relative_to(chemistry))] = digest(path)
    if len(chemical_files) != 62 or list(molecules.molecule_ids) != list(range(1, 62)):
        raise RuntimeError('incomplete molecule representation identity')
    snapshot = json.loads((chemistry / 'manifests/snapshot-frozen.json').read_text(encoding='utf-8'))
    for item in snapshot['items']:
        token.checkpoint()
        normalized = chemistry / 'normalized' / Path(item['normalized_path']).name
        if digest(normalized) != item['normalized_sha256']:
            raise RuntimeError('normalized chemistry source changed')
    manifest = {'schema_version': 1, 'mode': 'operational_rehearsal', 'primary_inference_eligible': False, 'config': config.document(), 'engine_identity': engine_identity(), 'replicates': replicates, 'shard_size': shard_size, 'inputs': inputs, 'chemistry_root': str(chemistry), 'chemistry_files': chemical_files, 'molecular_grid_quantum': molecules.quantum, 'molecular_representation_version': molecules.pair_grid_version, 'statistics_policy': 'retain_labeled_tensor_no_inferential_pooling'}
    manifest['run_seed'] = hashlib.sha256(canonical(manifest)).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    with worker_lock(root):
        db = open_state(root)
        try:
            raw = canonical(manifest)
            old = db.execute("SELECT value FROM identity WHERE key='manifest'").fetchone()
            if old and old['value'].encode() != raw:
                raise RuntimeError('run identity changed; use a new isolated state directory')
            with immediate(db):
                db.execute("INSERT OR IGNORE INTO identity VALUES('manifest',?)", (raw.decode(),))
                for entry in inputs:
                    db.execute('INSERT OR IGNORE INTO inputs VALUES(?,?)', (entry['id'], canonical(entry).decode()))
                    for scale in range(8):
                        db.execute("INSERT OR IGNORE INTO units(song_id,scale,replicate_start,replicate_count,status) VALUES(?,?,0,1,'pending')", (entry['id'], scale))
                        for start in range(1, replicates + 1, shard_size):
                            db.execute("INSERT OR IGNORE INTO units(song_id,scale,replicate_start,replicate_count,status) VALUES(?,?,?,?,'pending')", (entry['id'], scale, start, min(shard_size, replicates - start + 1)))
                event(db, 'prepared', {'inputs': len(inputs), 'seed': manifest['run_seed']})
            atomic(root / 'run-manifest.json', raw)
        finally:
            db.close()
    return manifest

def status(root):
    db = open_state(root)
    try:
        counts = {r['status']: r['n'] for r in db.execute('SELECT status,count(*) n FROM units GROUP BY status')}
        songs = db.execute("SELECT count(*) FROM inputs i WHERE NOT EXISTS(SELECT 1 FROM units u WHERE u.song_id=i.id AND status!='complete')").fetchone()[0]
        n = db.execute('SELECT count(*) FROM inputs').fetchone()[0]
        return {'songs_complete': songs, 'songs_total': n, 'units': counts, 'pause_requested': (Path(root) / 'PAUSE').exists(), 'primary_inference_eligible': False}
    finally:
        db.close()

def recover(db, root):
    root = Path(root).resolve()
    rows = db.execute("SELECT * FROM units WHERE status IN ('processing','complete')").fetchall()
    for row in rows:
        key = (row['song_id'], row['scale'], row['replicate_start'])
        damaged = row['status'] == 'complete' and (not row['path'] or not Path(row['path']).is_file() or digest(row['path']) != row['sha256'])
        if row['status'] == 'processing' or damaged:
            if row['path']:
                path = Path(row['path']).resolve()
                if not path.is_relative_to(root / 'artifacts'):
                    raise RuntimeError('recovery path escaped run artifacts')
                path.unlink(missing_ok=True)
            if row['attempt']:
                (root / 'staging' / (row['attempt'] + '.npz.partial')).unlink(missing_ok=True)
            with immediate(db):
                db.execute("UPDATE units SET status='pending',attempt=NULL,path=NULL,sha256=NULL,error=? WHERE song_id=? AND scale=? AND replicate_start=?", ('recovered interruption' if not damaged else 'artifact hash audit failed', *key))
                event(db, 'recovered' if not damaged else 'artifact_requeued', {'key': key})

def verify_payload(payload, entry, scale, replicate_indexes, molecule_count=61):
    labels = json.loads(str(payload['measure_labels_json']))
    if labels != measure_labels(entry['channels']):
        raise RuntimeError('channel tensor labels changed')
    expected = (len(replicate_indexes), len(labels), molecule_count)
    if payload['distances'].shape != expected or not np.all(np.isfinite(payload['distances'])):
        raise RuntimeError('distance tensor incomplete or nonfinite')
    if np.any(payload['distances'] < 0):
        raise RuntimeError('negative distances')
    if list(payload['replicate_indexes']) != replicate_indexes:
        raise RuntimeError('replicate shard identity mismatch')
    if int(payload['geometry'][0]) != scale or int(payload['geometry'][4]) != entry['channels']:
        raise RuntimeError('scale geometry mismatch')
    if payload['error_radius'].shape != (molecule_count,) or not np.all(np.isfinite(payload['error_radius'])):
        raise RuntimeError('missing numerical error certificate')

def run(root, *, backend='gpu', frame_batch=64, max_units=None, hook=None):
    if frame_batch < 1 or (max_units is not None and max_units < 1):
        raise ValueError('positive frame batch and unit limit required')
    root = Path(root).resolve()
    token = FileToken(root / 'PAUSE')
    for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, 'SIGBREAK', None)):
        if signum is not None:
            try:
                signal.signal(signum, lambda _s, _f: token.cancel())
            except ValueError:
                pass
    engine = None
    key = None
    completed = 0
    with worker_lock(root):
        db = open_state(root)
        try:
            identity = db.execute("SELECT value FROM identity WHERE key='manifest'").fetchone()
            if identity is None:
                raise RuntimeError('prepare a run first')
            manifest = json.loads(identity['value'])
            if manifest['engine_identity'] != engine_identity():
                raise RuntimeError('engine code changed after run preparation')
            if canonical(manifest) != (root / 'run-manifest.json').read_bytes():
                raise RuntimeError('run manifest changed')
            config = TensorConfig(**{**manifest['config'], 'scales_seconds': tuple(manifest['config']['scales_seconds'])})
            chemistry = Path(manifest['chemistry_root'])
            for path, expected in manifest['chemistry_files'].items():
                token.checkpoint()
                if digest(chemistry / path) != expected:
                    raise RuntimeError('chemistry artifact changed after preparation')
            molecules = load_molecular_grid(chemistry)
            if molecules.manifest_sha256 != manifest['chemistry_files']['manifests/snapshot-frozen.json']:
                raise RuntimeError('chemistry snapshot changed')
            if getattr(molecules, 'pair_grid_version', None) != manifest.get('molecular_representation_version'):
                raise RuntimeError('chemistry representation version changed')
            for entry in manifest['inputs']:
                token.checkpoint()
                source = Path(entry['path'])
                if source.stat().st_size != entry['size_bytes'] or digest(source) != entry['source_sha256']:
                    raise RuntimeError('source audio changed after preparation')
            recover(db, root)
            (root / 'artifacts').mkdir(exist_ok=True)
            (root / 'staging').mkdir(exist_ok=True)
            while not token.cancelled and (max_units is None or completed < max_units):
                row = db.execute("SELECT * FROM units WHERE status='pending' ORDER BY song_id,scale,replicate_start LIMIT 1").fetchone()
                if row is None:
                    break
                unit = dict(row)
                unit_key = (unit['song_id'], unit['scale'], unit['replicate_start'])
                attempt = uuid.uuid4().hex
                final = root / 'artifacts' / f"{unit['song_id']:04d}-{unit['scale']}-{unit['replicate_start']:05d}-{attempt}.npz"
                partial = root / 'staging' / f'{attempt}.npz.partial'
                with immediate(db):
                    db.execute("UPDATE units SET status='processing',attempt=?,attempts=attempts+1,path=?,error=NULL WHERE song_id=? AND scale=? AND replicate_start=?", (attempt, str(final), *unit_key))
                    event(db, 'claimed', {'key': unit_key, 'attempt': attempt})
                started = time.perf_counter()
                try:
                    if hook:
                        hook('claimed', unit_key, root)
                    entry = manifest['inputs'][unit['song_id'] - 1]
                    engine_key = (unit['song_id'], unit['scale'])
                    if key != engine_key:
                        if engine is not None:
                            engine.close()
                        engine = None
                        source = Path(entry['path'])
                        if digest(source) != entry['source_sha256']:
                            raise RuntimeError('source changed before scale processing')
                        engine = ScaleEngine(source, entry['decoded_sha256'], unit['scale'], config, token, frame_batch=frame_batch, backend=backend)
                        key = engine_key
                    indexes = list(range(unit['replicate_start'], unit['replicate_start'] + unit['replicate_count']))
                    payload = engine.compute(molecules, indexes, manifest['run_seed'])
                    verify_payload(payload, entry, unit['scale'], indexes)
                    payload['run_seed'] = np.asarray(manifest['run_seed'])
                    payload['source_decoded_sha256'] = np.asarray(entry['decoded_sha256'])
                    with partial.open('wb') as stream:
                        np.savez_compressed(stream, **payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if hook:
                        hook('partial_fsynced', unit_key, root)
                    token.checkpoint()
                    os.replace(partial, final)
                    if hook:
                        hook('published', unit_key, root)
                    token.checkpoint()
                    artifact_hash = digest(final)
                    with immediate(db):
                        db.execute("UPDATE units SET status='complete',sha256=?,error=NULL WHERE song_id=? AND scale=? AND replicate_start=? AND attempt=? AND status='processing'", (artifact_hash, *unit_key, attempt))
                        event(db, 'committed', {'key': unit_key, 'wall_seconds': time.perf_counter() - started, 'artifact_bytes': final.stat().st_size, 'frame_batch': frame_batch, 'backend': backend})
                    completed += 1
                    if hook:
                        hook('committed', unit_key, root)
                    if unit['scale'] == 7 and unit['replicate_start'] + unit['replicate_count'] > manifest['replicates']:
                        print(json.dumps({'event': 'song_complete', 'song_ordinal': unit['song_id'], 'elapsed_last_unit_seconds': time.perf_counter() - started}), flush=True)
                except Cancelled:
                    partial.unlink(missing_ok=True)
                    final.unlink(missing_ok=True)
                    with immediate(db):
                        db.execute("UPDATE units SET status='pending',attempt=NULL,path=NULL,sha256=NULL,error='cooperative pause' WHERE song_id=? AND scale=? AND replicate_start=?", unit_key)
                        event(db, 'paused', {'key': unit_key})
                    break
                except Exception as exc:
                    partial.unlink(missing_ok=True)
                    final.unlink(missing_ok=True)
                    with immediate(db):
                        db.execute("UPDATE units SET status='failed',sha256=NULL,error=? WHERE song_id=? AND scale=? AND replicate_start=?", (f'{type(exc).__name__}: {exc}', *unit_key))
                        event(db, 'failed', {'key': unit_key, 'error': f'{type(exc).__name__}: {exc}'})
                    if engine is not None:
                        engine.close()
                    engine, key = (None, None)
                    print(json.dumps({'event': 'unit_failed', 'key': unit_key, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
                    break
        except Cancelled:
            pass
        finally:
            if engine is not None:
                engine.close()
            db.close()
    snapshot = status(root)
    atomic(root / 'operational-status.json', canonical(snapshot))
    return snapshot

def finalize(root):
    root = Path(root).resolve()
    with worker_lock(root):
        db = open_state(root)
        try:
            manifest = json.loads(db.execute("SELECT value FROM identity WHERE key='manifest'").fetchone()[0])
            if manifest['engine_identity'] != engine_identity():
                raise RuntimeError('engine changed before final audit')
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise RuntimeError('ledger integrity failed')
            rows = db.execute('SELECT * FROM units ORDER BY song_id,scale,replicate_start').fetchall()
            expected = len(manifest['inputs']) * 8 * (1 + math_ceil_div(manifest['replicates'], manifest['shard_size']))
            if len(rows) != expected or any((r['status'] != 'complete' for r in rows)):
                raise RuntimeError('full tensor work set is incomplete')
            inventory = []
            distance_cells = 0
            for row in rows:
                path = Path(row['path'])
                if digest(path) != row['sha256']:
                    raise RuntimeError('final artifact audit failed')
                entry = manifest['inputs'][row['song_id'] - 1]
                indexes = list(range(row['replicate_start'], row['replicate_start'] + row['replicate_count']))
                with np.load(path, allow_pickle=False) as payload:
                    verify_payload(payload, entry, row['scale'], indexes)
                    if str(payload['run_seed']) != manifest['run_seed'] or str(payload['source_decoded_sha256']) != entry['decoded_sha256']:
                        raise RuntimeError('artifact input/run binding mismatch')
                    if json.loads(str(payload['operator_json'])) != json.loads(canonical(manifest['config'])):
                        raise RuntimeError('artifact operator mismatch')
                    geometry = payload['geometry']
                    length = int(np.floor(manifest['config']['scales_seconds'][row['scale']] * entry['sample_rate'] + 0.5))
                    hop = int(np.floor(length / 4 + 0.5))
                    if list(geometry[1:]) != [length, hop, 1 + (entry['frames'] - length) // hop, entry['channels'], length // 2]:
                        raise RuntimeError('full source-frame coverage mismatch')
                    distance_cells += payload['distances'].size
                inventory.append({'key': [row['song_id'], row['scale'], row['replicate_start']], 'path': str(path.relative_to(root)), 'sha256': row['sha256'], 'bytes': path.stat().st_size})
            for entry in manifest['inputs']:
                if digest(entry['path']) != entry['source_sha256']:
                    raise RuntimeError('source changed by the end of rehearsal')
            report = {'schema_version': 1, 'mode': 'operational_rehearsal', 'run_seed': manifest['run_seed'], 'songs_complete': len(manifest['inputs']), 'scales_per_song': 8, 'molecules': 61, 'surrogate_replicates': manifest['replicates'], 'units_complete': len(rows), 'distance_cells_checked': distance_cells, 'finite_labeled_tensors_verified': True, 'source_rehash_passed': True, 'artifact_rehash_passed': True, 'ledger_integrity': 'ok', 'source_channels_preserved': True, 'effects_inspected': False, 'inferential_statistic_computed': False, 'primary_inference_eligible': False, 'completed_at': utc_now()}
            atomic(root / 'artifact-manifest.json', canonical(inventory))
            atomic(root / 'acceptance.json', canonical(report))
            return report
        finally:
            db.close()

def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--batch', required=True)
    p.add_argument('--chemistry', required=True)
    p.add_argument('--replicates', type=int, default=2)
    p.add_argument('--shard-size', type=int, default=8)
    p.add_argument('--quantum', type=float, default=0.005)
    for name in ('run', 'resume'):
        r = sub.add_parser(name)
        r.add_argument('--backend', choices=['gpu', 'cpu'], default='cpu' if getattr(sys, 'frozen', False) else 'gpu')
        r.add_argument('--frame-batch', type=int, default=64)
        r.add_argument('--max-units', type=int)
        r.add_argument('--fault', choices=['claimed', 'partial_fsynced', 'published', 'committed'])
    for name in ('status', 'pause', 'retry-failed', 'finalize'):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    root = Path(args.state)
    if args.command == 'prepare':
        manifest = prepare(root, args.batch, args.chemistry, replicates=args.replicates, shard_size=args.shard_size, quantum=args.quantum)
        result = {'prepared': True, 'run_seed': manifest['run_seed'], 'songs': len(manifest['inputs'])}
    elif args.command in ('run', 'resume'):
        if args.command == 'resume':
            (root / 'PAUSE').unlink(missing_ok=True)
        fault = getattr(args, 'fault', None)

        def hook(stage, _key, _root):
            if stage == fault:
                os._exit(91)
        result = run(root, backend=getattr(args, 'backend', 'gpu'), frame_batch=getattr(args, 'frame_batch', 64), max_units=getattr(args, 'max_units', None), hook=hook if fault else None)
    elif args.command == 'pause':
        root.mkdir(parents=True, exist_ok=True)
        atomic(root / 'PAUSE', b'cooperative pause requested\n')
        result = {'pause_requested': True, 'completed_work_retained': True}
    elif args.command == 'retry-failed':
        with worker_lock(root):
            db = open_state(root)
            with immediate(db):
                count = db.execute("UPDATE units SET status='pending',error=NULL WHERE status='failed'").rowcount
            db.close()
        result = {'requeued': count}
    elif args.command == 'finalize':
        result = finalize(root)
    else:
        result = status(root)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2 if result.get('units', {}).get('failed') else 0
if __name__ == '__main__':
    raise SystemExit(main())
