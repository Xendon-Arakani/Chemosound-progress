import argparse
import json
import platform
import signal
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import soundfile as sf
from .db import connect, initialize
from .model import CancellationToken, Paths
from .pipeline import add_root, freeze, retry_failed, run_validation, scan, status

EXTENSIONS = frozenset({'.wav', '.wave', '.aif', '.aiff'})


def parser():
    result = argparse.ArgumentParser(prog='Chemosound')
    result.add_argument('--state-dir', default='.chemosound')
    sub = result.add_subparsers(dest='command', required=True)
    sub.add_parser('runtime')
    init = sub.add_parser('init')
    init.add_argument('--audio-root', action='append', required=True)
    for command in ('scan', 'run', 'status', 'retry-failed'):
        sub.add_parser(command)
    frozen = sub.add_parser('freeze')
    frozen.add_argument('--max-songs', type=int, default=9999)
    return result


def stop_token():
    token = CancellationToken.create()

    def stop(signum, frame):
        if token.cancelled:
            raise KeyboardInterrupt
        token.cancel()

    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    return token


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser().parse_args(argv or ['--help'])
        if args.command == 'runtime':
            print(json.dumps({'version': '0.1.0', 'release_stage': 'progress', 'frozen': bool(getattr(sys, 'frozen', False)), 'python': platform.python_version(), 'numpy': np.__version__, 'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__, 'backends': ['cpu'], 'formats': ['WAV', 'AIFF'], 'primary_inference_eligible': False}, sort_keys=True))
            return 0
        paths = Paths(Path(args.state_dir).expanduser().resolve())
        paths.ensure()
        db = connect(paths.database)
        try:
            initialize(db)
            if args.command == 'init':
                for root in args.audio_root:
                    add_root(db, Path(root))
                result = {'initialized': True, 'audio_roots': args.audio_root}
            elif args.command == 'scan':
                result = scan(db, extensions=EXTENSIONS)
            elif args.command == 'run':
                summary = run_validation(db, paths, stop_token())
                print(json.dumps(asdict(summary), sort_keys=True))
                return 130 if summary.cancelled else 2 if summary.failed_total else 0
            elif args.command == 'status':
                result = status(db)
            elif args.command == 'retry-failed':
                result = {'retried': retry_failed(db)}
            elif args.command == 'freeze':
                count, value = freeze(db, max_songs=args.max_songs)
                result = {'frozen': True, 'song_count': count, 'manifest_sha256': value, 'primary_inference_eligible': False}
            print(json.dumps(result, sort_keys=True))
            return 0
        finally:
            db.close()
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(json.dumps({'error': type(error).__name__, 'detail': str(error)}, sort_keys=True), file=sys.stderr)
        return 2
