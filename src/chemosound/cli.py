from __future__ import annotations
import argparse
import getpass
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from .analysis_pipeline import analysis_status, run_music_pilot
from .db import connect, initialize
from .hitran import acquire_v2_sources, build_molecule_grids, freeze_snapshot, normalize_sources, prepare_snapshot, register_static, snapshot_status, sync_snapshot_db
from .model import CancellationToken, Paths
from .pipeline import add_root, freeze, retry_failed, run_validation, scan, status

def _paths(value: str) -> Paths:
    return Paths(Path(value).expanduser().resolve())

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='chemosound')
    parser.add_argument('--state-dir', default='.chemosound')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--audio-root', action='append', required=True)
    sub.add_parser('scan')
    sub.add_parser('run')
    sub.add_parser('status')
    sub.add_parser('analyze-exploratory')
    sub.add_parser('analysis-status')
    sub.add_parser('retry-failed')
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--max-songs', type=int, default=9999)
    hitran_prepare = sub.add_parser('hitran-prepare')
    hitran_prepare.add_argument('--root', required=True)
    hitran_prepare.add_argument('--molecule-html', required=True)
    hitran_prepare.add_argument('--isotopologue-html', required=True)
    hitran_prepare.add_argument('--acquired-at', required=True)
    hitran_acquire = sub.add_parser('hitran-acquire')
    hitran_acquire.add_argument('--root', required=True)
    hitran_static = sub.add_parser('hitran-register')
    hitran_static.add_argument('--root', required=True)
    hitran_static.add_argument('--molecule-id', type=int, required=True)
    hitran_static.add_argument('--data', required=True)
    hitran_static.add_argument('--header')
    hitran_normalize = sub.add_parser('hitran-normalize')
    hitran_normalize.add_argument('--root', required=True)
    hitran_freeze = sub.add_parser('hitran-freeze')
    hitran_freeze.add_argument('--root', required=True)
    hitran_status_parser = sub.add_parser('hitran-status')
    hitran_status_parser.add_argument('--root', required=True)
    hitran_relations = sub.add_parser('hitran-pairs')
    hitran_relations.add_argument('--root', required=True)
    hitran_relations.add_argument('--coordinate-quantum', type=float, default=0.0001)
    return parser

def _bind_stop_signals(token: CancellationToken) -> None:

    def request_stop(signum: int, _frame: object) -> None:
        if token.cancelled:
            raise KeyboardInterrupt
        print(f'\nSignal {signum} received; finishing the current checkpoint safely...', file=sys.stderr)
        token.cancel()
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, request_stop)

def main(argv: list[str] | None=None) -> int:
    args = _parser().parse_args(argv)
    paths = _paths(args.state_dir)
    paths.ensure()
    connection = connect(paths.database)
    try:
        initialize(connection)
        if args.command == 'init':
            for root in args.audio_root:
                add_root(connection, Path(root))
            print(json.dumps({'initialized': True, 'audio_roots': args.audio_root}, indent=2))
            return 0
        if args.command == 'scan':
            print(json.dumps(scan(connection), indent=2, sort_keys=True))
            return 0
        if args.command == 'run':
            token = CancellationToken.create()
            _bind_stop_signals(token)
            summary = run_validation(connection, paths, token)
            print(json.dumps(summary.__dict__ if hasattr(summary, '__dict__') else {'completed': summary.completed, 'failed_this_run': summary.failed_this_run, 'failed_total': summary.failed_total, 'cancelled': summary.cancelled, 'exhausted': summary.exhausted}, indent=2, sort_keys=True))
            if summary.cancelled:
                return 130
            return 2 if summary.failed_total else 0
        if args.command == 'status':
            print(json.dumps(status(connection), indent=2, sort_keys=True))
            return 0
        if args.command == 'analyze-exploratory':
            token = CancellationToken.create()
            _bind_stop_signals(token)
            summary = run_music_pilot(connection, paths, token)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            if summary.cancelled:
                return 130
            return 2 if summary.failed_total else 0
        if args.command == 'analysis-status':
            print(json.dumps(analysis_status(connection), indent=2, sort_keys=True))
            return 0
        if args.command == 'retry-failed':
            print(json.dumps({'retried': retry_failed(connection)}, indent=2))
            return 0
        if args.command == 'freeze':
            count, digest = freeze(connection, max_songs=args.max_songs)
            print(json.dumps({'frozen': True, 'song_count': count, 'manifest_sha256': digest}, indent=2))
            return 0
        if args.command == 'hitran-prepare':
            root = Path(args.root)
            document = prepare_snapshot(root, Path(args.molecule_html), Path(args.isotopologue_html), acquired_at=args.acquired_at)
            snapshot_id = sync_snapshot_db(connection, root)
            print(json.dumps({'snapshot_id': snapshot_id, 'items': len(document['items'])}, indent=2))
            return 0
        if args.command == 'hitran-acquire':
            root = Path(args.root)
            api_key = os.environ.get('HITRAN_API_KEY') or getpass.getpass('HITRAN API key: ')
            try:
                result = acquire_v2_sources(root, api_key)
            finally:
                snapshot_id = sync_snapshot_db(connection, root)
                api_key = ''
            result['snapshot_id'] = snapshot_id
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == 'hitran-register':
            root = Path(args.root)
            item = register_static(root, args.molecule_id, Path(args.data), Path(args.header) if args.header else None)
            sync_snapshot_db(connection, root)
            print(json.dumps(item, indent=2, sort_keys=True))
            return 0
        if args.command == 'hitran-normalize':
            root = Path(args.root)
            result = normalize_sources(root)
            result['snapshot_id'] = sync_snapshot_db(connection, root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == 'hitran-freeze':
            root = Path(args.root)
            digest, document = freeze_snapshot(root)
            sync_snapshot_db(connection, root)
            print(json.dumps({'frozen': True, 'manifest_sha256': digest, 'items': len(document['items'])}, indent=2))
            return 0
        if args.command == 'hitran-status':
            root = Path(args.root)
            result = snapshot_status(root)
            result['snapshot_id'] = sync_snapshot_db(connection, root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == 'hitran-pairs':
            result = build_molecule_grids(connection, Path(args.root), coordinate_quantum=args.coordinate_quantum)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2 if result['failed'] else 0
        raise AssertionError(f'unhandled command: {args.command}')
    except (RuntimeError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    finally:
        connection.close()
