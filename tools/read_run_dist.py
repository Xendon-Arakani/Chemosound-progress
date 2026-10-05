from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from chemosound.flow import atomic, canonical, digest, utc_now, verify_payload

def describe(delta, margin, real, surrogate):
    delta, margin = (np.asarray(delta), np.asarray(margin))
    count = delta.size
    real_mean, control_mean = (float(np.mean(real)), float(np.mean(surrogate)))
    return {'comparisons': count, 'wins': int(np.count_nonzero(delta > 0)), 'losses': int(np.count_nonzero(delta < 0)), 'ties': int(np.count_nonzero(delta == 0)), 'win_percent': 100 * float(np.mean(delta > 0)), 'mean_real_distance': real_mean, 'mean_control_distance': control_mean, 'mean_control_minus_real': float(np.mean(delta)), 'median_control_minus_real': float(np.median(delta)), 'relative_mean_distance_improvement_percent': 100 * (control_mean - real_mean) / control_mean if control_mean > 0 else None, 'wins_exceeding_quantization_bounds': int(np.count_nonzero(delta > margin)), 'losses_exceeding_quantization_bounds': int(np.count_nonzero(delta < -margin)), 'quantization_unresolved': int(np.count_nonzero(np.abs(delta) <= margin)), 'win_percent_exceeding_quantization_bounds': 100 * float(np.mean(delta > margin)), 'mean_difference_quantization_radius': float(np.mean(margin)), 'mean_difference_lower_quantization_bound': float(np.mean(delta - margin)), 'mean_difference_upper_quantization_bound': float(np.mean(delta + margin))}

def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, stream.getvalue().encode('utf-8'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--output-name', default='descriptive-inspection-v1')
    args = parser.parse_args()
    root = Path(args.state).resolve()
    out = (root / args.output_name).resolve()
    if out.parent != root:
        raise ValueError('inspection output must be a direct child of the run directory')
    manifest = json.loads((root / 'run-manifest.json').read_text())
    acceptance = json.loads((root / 'acceptance.json').read_text())
    if acceptance['songs_complete'] != 300 or manifest['replicates'] != 2:
        raise RuntimeError('expected complete two-surrogate rehearsal')
    out.mkdir(exist_ok=True)
    plan = {'analysis': 'descriptive-inspection-v1', 'created_at_utc': utc_now(), 'authorization': 'User requested inspection: tell me how it looks so far. is it any good', 'analysis_source_sha256': digest(__file__), 'run_manifest_sha256': digest(root / 'run-manifest.json'), 'artifact_manifest_sha256': digest(root / 'artifact-manifest.json'), 'cohorts': 'channel_count', 'groups': 'channel_pair_indexes', 'metrics': 'w1_win_delta', 'precision_check': 'quantization_radius', 'consistency': 'recording_molecule', 'selection': 'all_cells', 'inference': 'none', 'exposure': 'observed_corpus'}
    plan_path = out / 'inspection-plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k: v for k, v in old.items() if k != 'created_at_utc'} != {k: v for k, v in plan.items() if k != 'created_at_utc'}:
            raise RuntimeError('inspection plan changed; use a new named inspection')
        plan = old
    else:
        atomic(plan_path, canonical(plan))
    print(json.dumps({'event': 'inspection_plan_persisted', 'at': plan['created_at_utc']}), flush=True)
    inventory = json.loads((root / 'artifact-manifest.json').read_text())
    paths = {}
    for item in inventory:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root / 'artifacts') or digest(path) != item['sha256']:
            raise RuntimeError('artifact identity audit failed')
        key = tuple(item['key'])
        if key in paths:
            raise RuntimeError('duplicate artifact key')
        paths[key] = path
    if len(paths) != 4800:
        raise RuntimeError('incomplete artifact inventory')
    by_group = defaultdict(list)
    scale_rows, molecule_rows, song_rows = ([], [], [])
    cells = 0
    for entry in manifest['inputs']:
        for scale in range(8):
            with np.load(paths[entry['id'], scale, 0], allow_pickle=False) as p0, np.load(paths[entry['id'], scale, 1], allow_pickle=False) as pb:
                verify_payload(p0, entry, scale, [0])
                verify_payload(pb, entry, scale, [1, 2])
                for payload in (p0, pb):
                    if str(payload['run_seed']) != manifest['run_seed'] or str(payload['source_decoded_sha256']) != entry['decoded_sha256']:
                        raise RuntimeError('payload input/run mismatch')
                real, controls = (p0['distances'][0], pb['distances'])
                margin = p0['error_radius'] + pb['error_radius']
                labels = json.loads(str(p0['measure_labels_json']))
                cells += real.size + controls.size
                for index, label in enumerate(labels):
                    group = (entry['channels'], label['kind'], tuple(label['channels']))
                    by_group[group].append((entry['id'], scale, real[index], controls[:, index], margin))
        if entry['id'] % 50 == 0:
            print(json.dumps({'event': 'distances_loaded', 'recordings': entry['id']}), flush=True)
    if cells != acceptance['distance_cells_checked']:
        raise RuntimeError('distance-cell coverage mismatch')
    group_rows = []
    for group, records in sorted(by_group.items()):
        channels, kind, indexes = group
        identity = {'source_channels': channels, 'measure_kind': kind, 'channel_indexes': '-'.join(map(str, indexes))}

        def summarize(subset):
            r = np.stack([v[2] for v in subset])
            s = np.stack([v[3] for v in subset])
            b = np.broadcast_to(np.stack([v[4] for v in subset])[:, None, :], s.shape)
            delta = s - r[:, None, :]
            stats = describe(delta, b, r, s)
            stats.update({f'control_{j + 1}_win_percent': 100 * float(np.mean(delta[:, j] > 0)) for j in range(2)})
            return (stats, delta, b, r, s)
        stats, delta, bounds, r, s = summarize(records)
        song_ids = sorted({v[0] for v in records})
        song_effects, molecule_effects = ([], [])
        for song_id in song_ids:
            subset = [v for v in records if v[0] == song_id]
            song_stats, *_ = summarize(subset)
            song_rows.append({**identity, 'song_id': song_id, **song_stats})
            song_effects.append(song_stats['mean_control_minus_real'])
        for molecule in range(61):
            molecule_stats = describe(delta[:, :, molecule], bounds[:, :, molecule], r[:, molecule], s[:, :, molecule])
            molecule_rows.append({**identity, 'molecule_id': molecule + 1, **molecule_stats})
            molecule_effects.append(molecule_stats['mean_control_minus_real'])
        stats.update({'recordings': len(song_ids), 'recordings_with_positive_mean_effect': int(np.count_nonzero(np.array(song_effects) > 0)), 'molecules_with_positive_mean_effect': int(np.count_nonzero(np.array(molecule_effects) > 0))})
        group_rows.append({**identity, **stats})
        for scale in range(8):
            subset = [v for v in records if v[1] == scale]
            scale_stats, *_ = summarize(subset)
            scale_rows.append({**identity, 'scale_index': scale, 'window_seconds': manifest['config']['scales_seconds'][scale], **scale_stats})
    report = {'inspected_at_utc': utc_now(), 'effects_inspected': True, 'official_inference_performed': False, 'corpus_outcome_exposed': True, 'artifact_rehash_passed': True, 'distance_cells_read': cells, 'matched_control_comparisons': sum((v['comparisons'] for v in group_rows)), 'inspection_plan_sha256': digest(plan_path), 'groups': 'channel_pair_indexes', 'scales': scale_rows, 'interpretation_scope': 'descriptive_w1_only'}
    for name, rows in [('group-summary.csv', group_rows), ('scale-summary.csv', scale_rows), ('molecule-summary.csv', molecule_rows), ('recording-summary.csv', song_rows)]:
        write_csv(out / name, rows)
    atomic(out / 'descriptive-results.json', canonical(report))
    atomic(out / 'analysis_src.raw', Path(__file__).read_bytes())
    atomic(out / 'inspection-evidence.json', canonical({p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'inspection-evidence.json'}))
    print(json.dumps({'distance_cells_read': cells, 'matched_control_comparisons': report['matched_control_comparisons'], 'groups': 'channel_pair_indexes'}, indent=2))
if __name__ == '__main__':
    main()
