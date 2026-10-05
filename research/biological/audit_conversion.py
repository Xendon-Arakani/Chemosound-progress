from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from chemosound.hitran import PROTOCOL_MOLECULES

TEMP_K = 296.0
C2_CM_K = 6.62607015e-34 * 299792458.0 * 100 / 1.380649e-23


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def log_weights(nu, strength, energy, upper):
    arrays = [np.asarray(x, dtype=np.float64) for x in (nu, strength, energy, upper)]
    nu, strength, energy, upper = arrays
    if arrays[0].ndim != 1 or not arrays[0].size or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('TRANSITION_FIELDS_INVALID')
    if np.any(nu <= 0) or np.any(strength <= 0) or np.any(upper <= 0) or np.any(energy < 0):
        raise ValueError('TRANSITION_FIELDS_NONPHYSICAL')
    return np.log(strength) + 2 * np.log(nu) + C2_CM_K * energy / TEMP_K - np.log(upper) - np.log(-np.expm1(-C2_CM_K * nu / TEMP_K))


def normalize(logs):
    shifted = np.exp(logs - np.max(logs))
    return shifted / math.fsum(shifted)


def inspect_item(root, entry):
    old = Path(entry['raw_data_path'])
    path = root / old.parent.name / old.name
    if digest(path) != entry['raw_data_sha256']:
        raise RuntimeError('FROZEN_SOURCE_HASH_MISMATCH')
    values, errors, retained, observed = [], [], 0, 0
    bad_count = 0
    reasons = {}
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            observed += 1
            if int(line[:2]) != entry['molecule_id']:
                raise RuntimeError('SOURCE_MOLECULE_MISMATCH')
            if int(line[2:3]) != 1:
                continue
            nu, a = float(line[3:15]), float(line[25:35])
            if not math.isfinite(nu) or not math.isfinite(a):
                raise RuntimeError('SOURCE_VALUE_NONFINITE')
            if nu <= 0 or a <= 0:
                continue
            retained += 1
            try:
                strength = float(line[15:25])
                energy = float(line[45:55])
                upper = float(line[146:153])
                if not all(math.isfinite(x) for x in (strength, energy, upper)):
                    raise ValueError('TRANSITION_FIELDS_NONFINITE')
                if strength <= 0:
                    raise ValueError('INTENSITY_NONPOSITIVE')
                if upper <= 0:
                    raise ValueError('UPPER_WEIGHT_NONPOSITIVE')
                if energy < 0:
                    raise ValueError('LOWER_ENERGY_UNKNOWN')
            except ValueError as error:
                bad_count += 1
                code = str(error) if str(error).isupper() else 'FIELD_PARSE_FAILED'
                reasons[code] = reasons.get(code, 0) + 1
                if len(errors) < 10:
                    errors.append({'line': number, 'code': code, 'S296': line[15:25].decode('ascii'), 'E_lower': line[45:55].decode('ascii'), 'g_upper': line[146:153].decode('ascii')})
                continue
            values.append((nu, strength, energy, upper, a, number))
    if observed != entry['observed_source_line_count'] or retained != entry['retained_line_count']:
        raise RuntimeError('FROZEN_SOURCE_COUNT_MISMATCH')
    row = {'id': entry['molecule_id'], 'formula': entry['formula'], 'source_sha256': entry['raw_data_sha256'], 'observed': observed, 'retained': retained, 'valid_inputs': len(values), 'invalid_inputs': bad_count, 'invalid_reasons': reasons, 'invalid_line_samples': errors, 'conversion_complete': bad_count == 0, 'comparison_subset': 'valid_retained_lines', 'production_conversion_qualified': False}
    if values:
        data = np.asarray(values, dtype=np.float64)
        logs = log_weights(*data[:, :4].T)
        a_logs = np.log(data[:, 4])
        derived, reference = normalize(logs), normalize(a_logs)
        ratios = logs - a_logs
        centered = ratios - np.median(ratios)
        row.update({'total_variation': float(np.sum(np.abs(derived - reference)) / 2), 'log_ratio_span': float(np.ptp(ratios)), 'log_ratio_quantiles': np.quantile(centered, [0, 0.01, 0.5, 0.99, 1]).tolist(), 'underflowed_weights': int(np.count_nonzero(derived == 0)), 'direct_a_used_in_derivation': False, 'comparison_discrepancy_cause_qualified': False})
        samples = list(dict.fromkeys([int(np.argmin(ratios)), int(np.argmax(ratios))]))
        row['ratio_extreme_samples'] = [{'line': int(data[i, 5]), 'nu_cm_1': float(data[i, 0]), 'S296_cm_per_molecule': float(data[i, 1]), 'E_lower_cm_1': float(data[i, 2]), 'g_upper': float(data[i, 3]), 'reference_A_s_1': float(data[i, 4]), 'ratio_vs_median': float(np.exp(centered[i]))} for i in samples]
    return row


def inspect_candidates():
    base = ROOT / 'history' / 'ir_source_v1'
    inspection = json.loads((base / 'numerical-inspection.json').read_text())
    evidence = json.loads((base / 'acquisition-evidence.json').read_text())
    if digest(base / 'numerical-inspection.json') != evidence['numerical_inspection_sha256']:
        raise RuntimeError('CANDIDATE_INSPECTION_HASH_MISMATCH')
    for member in evidence['inspected_members']:
        path = base / 'samples' / member['local_file']
        if digest(path) != member['sha256']:
            raise RuntimeError('CANDIDATE_SOURCE_HASH_MISMATCH')
    targets = {'HMDB0000073': (62, 'dopamine'), 'HMDB0000259': (63, 'serotonin')}
    rows = []
    for entry in inspection['inspected_entries']:
        if entry['database_id'] not in targets:
            continue
        ident, name = targets[entry['database_id']]
        rows.append({'id': ident, 'name': name, 'database_id': entry['database_id'], 'adduct': entry['metadata']['adduct'], 'method': entry['metadata']['frequency-method'], 'mode_count': entry['mode_count'], 'intensity_units': entry['metadata']['y-units'], 'einstein_a_present': entry['einstein_a_field_present'], 'resolved_transition_input_qualified': False, 'rejections': ['HARMONIC_FUNDAMENTALS', 'ION_IDENTITY_UNQUALIFIED', 'RESOLVED_STATE_FIELDS_ABSENT']})
    if {x['id'] for x in rows} != {62, 63}:
        raise RuntimeError('CANDIDATE_IDENTITIES_INCOMPLETE')
    return rows


def mode_limit(formula):
    atoms = sum(int(count or '1') for _, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula))
    return 0 if atoms == 1 else 1 if atoms == 2 else 3 * atoms - 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'local-data' / 'Chemosound-data' / 'hitran2024')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'manifests' / 'snapshot-frozen.json'
    frozen = json.loads(manifest.read_text())
    receipt = root / 'manifests' / 'snapshot-frozen.sha256'
    if receipt.read_text().split()[0] != digest(manifest):
        raise RuntimeError('FROZEN_MANIFEST_HASH_MISMATCH')
    if frozen.get('status') != 'frozen' or [x['molecule_id'] for x in frozen['items']] != list(range(1, 62)):
        raise RuntimeError('FROZEN_61_IDENTITIES_INVALID')
    start = time.monotonic()
    rows = []
    for entry in frozen['items']:
        row = inspect_item(root, entry)
        rows.append(row)
        print(json.dumps({'id': row['id'], 'conversion_complete': row['conversion_complete'], 'invalid_inputs': row['invalid_inputs']}), flush=True)
    candidates = inspect_candidates()
    excluded = [{'id': ident, 'formula': formula, 'harmonic_mode_limit': mode_limit(formula)} for ident, formula, _ in PROTOCOL_MOLECULES if mode_limit(formula) < 2]
    report = {'schema_version': 1, 'audit_complete': True, 'primary_inference_eligible': False, 'source_manifest_sha256': digest(manifest), 'script_sha256': digest(Path(__file__)), 'reference_temperature_K': TEMP_K, 'c2_cm_K': C2_CM_K, 'derivation': 'normalized_line_intensity_inversion_v1', 'input_columns': {'nu_cm_1': [3, 15], 'S296_cm_per_molecule': [15, 25], 'E_lower_cm_1': [45, 55], 'g_upper': [146, 153]}, 'direct_a_comparison_column': [25, 35], 'common_isotopologue_factor_cancelled': True, 'derived_absolute_A': False, 'complete_input_count': sum(x['conversion_complete'] for x in rows), 'required_count': 63, 'uniform_63_qualified': False, 'line_route_missing_ids': [62, 63], 'harmonic_route_pair_failures': excluded, 'sources': ['https://hitran.org/docs/definitions-and-units/', 'https://physics.nist.gov/cuu/pdf/wall_2022.pdf', 'https://doi.org/10.1021/acs.analchem.3c01078', 'https://doi.org/10.5281/zenodo.7706021', 'https://hj.hi.is/EE3/notesPESandNormalModes.pdf'], 'rows': rows, 'candidates': candidates, 'seconds': time.monotonic() - start}
    report['retained_lines'] = sum(x['retained'] for x in rows)
    report['invalid_line_inputs'] = sum(x['invalid_inputs'] for x in rows)
    target = ROOT / 'history' / 'conversion_audit.json'
    partial = target.with_suffix('.json.partial')
    with partial.open('wb') as stream:
        stream.write((json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, target)
    print(json.dumps({'report': str(target), 'complete_input_count': report['complete_input_count'], 'uniform_63_qualified': False}), flush=True)


if __name__ == '__main__':
    main()
