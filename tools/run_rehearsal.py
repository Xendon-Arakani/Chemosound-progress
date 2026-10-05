from __future__ import annotations
import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
from chemosound.flow import atomic, canonical, open_state, status, utc_now, worker_lock

def choose_plan(free_gpu_mib, *, ceiling=256):
    if free_gpu_mib is None or free_gpu_mib < 4096:
        return ('cpu', 64)
    tile = 256 if free_gpu_mib >= 8192 else 64
    return ('gpu', min(tile, ceiling))

def free_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi.exe', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '--id=0'], text=True, capture_output=True, timeout=10, check=True)
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

def free_host_memory():
    if sys.platform != 'win32':
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong), ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong), ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong), ('extended_virtual', ctypes.c_ulonglong)]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('host-memory query failed')
    return value.available_physical

def resource_failure(errors):
    return bool(errors) and all((e and (e.startswith('MemoryError:') or e.startswith('OutOfMemoryError:')) for e in errors))

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    args = parser.parse_args(argv)
    root = Path(args.state).resolve()
    ceiling, forced_cpu, failures = (256, False, 0)
    while True:
        snapshot = status(root)
        if snapshot['pause_requested']:
            print(json.dumps({'event': 'supervisor_paused', **snapshot}), flush=True)
            return 0
        if snapshot['songs_complete'] == snapshot['songs_total'] and snapshot['songs_total'] > 0:
            print(json.dumps({'event': 'supervisor_complete', **snapshot}), flush=True)
            return 0
        if snapshot['units'].get('failed'):
            db = open_state(root)
            errors = [r[0] for r in db.execute("SELECT error FROM units WHERE status='failed'")]
            db.close()
            if not resource_failure(errors) or failures >= 3:
                print(json.dumps({'event': 'repair_required', 'errors': errors}), flush=True)
                return 2
            failures += 1
            if ceiling > 64:
                ceiling = 64
            else:
                forced_cpu = True
            with worker_lock(root):
                db = open_state(root)
                db.execute("UPDATE units SET status='pending',error=NULL WHERE status='failed'")
                db.close()
        gpu_free = free_gpu_memory()
        backend, tile = choose_plan(gpu_free, ceiling=ceiling)
        if forced_cpu:
            backend, tile = ('cpu', 64)
        host_free = free_host_memory()
        if backend == 'cpu' and host_free is not None and (host_free < 6 * 1024 ** 3):
            print(json.dumps({'event': 'insufficient_host_memory', 'available_bytes': host_free}), flush=True)
            return 2
        plan = {'at': utc_now(), 'backend': backend, 'frame_batch': tile, 'free_gpu_mib': gpu_free, 'free_host_bytes': host_free, 'resource_retry': failures, 'effects_inspected': False}
        with (root / 'execution-plans.jsonl').open('ab') as stream:
            stream.write(canonical(plan) + b'\n')
        print(json.dumps({'event': 'execution_plan', **plan}), flush=True)
        result = subprocess.run([sys.executable, '-m', 'chemosound.flow', '--state', str(root), 'run', '--backend', backend, '--frame-batch', str(tile)])
        if result.returncode != 0 and (not status(root)['units'].get('failed')):
            print(json.dumps({'event': 'worker_start_or_environment_failure', 'exit_code': result.returncode}), flush=True)
            return result.returncode
if __name__ == '__main__':
    raise SystemExit(main())
