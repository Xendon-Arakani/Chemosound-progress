from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from .model import PERMISSION_SCOPE
PERMISSION_BASES = {'license', 'written_permission', 'purchase_terms', 'public_domain_review'}

@dataclass(frozen=True, slots=True)
class RightsRecord:
    status: str
    sha256: str | None
    document: dict | None
    error: str | None

def sidecar_path(audio_path: Path) -> Path:
    return Path(str(audio_path) + '.rights.json')

def load_rights(audio_path: Path) -> RightsRecord:
    path = sidecar_path(audio_path)
    if not path.is_file():
        return RightsRecord('missing', None, None, f'missing rights sidecar: {path}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return RightsRecord('invalid', digest, None, f'invalid rights JSON: {exc}')
    if not isinstance(document, dict):
        return RightsRecord('invalid', digest, None, 'rights JSON must be an object')
    errors: list[str] = []
    if document.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    if document.get('content_type') != 'complete_music_recording':
        errors.append('content_type must be complete_music_recording')
    if document.get('permission_basis') not in PERMISSION_BASES:
        errors.append('permission_basis is missing or unsupported')
    for field in ('title', 'artist', 'source_url', 'reviewed_by', 'reviewed_at'):
        if not isinstance(document.get(field), str) or not document[field].strip():
            errors.append(f'{field} is required')
    scope = document.get('permission_scope')
    if not isinstance(scope, list):
        errors.append('permission_scope must be a list')
    else:
        missing = PERMISSION_SCOPE.difference(scope)
        if missing:
            errors.append('missing permission scope: ' + ', '.join(sorted(missing)))
    evidence = document.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        errors.append('at least one evidence URL or path is required')
    if document.get('permission_basis') == 'license' and (not document.get('license_id') or not document.get('license_url')):
        errors.append('license_id and license_url are required for license basis')
    if errors:
        return RightsRecord('invalid', digest, document, '; '.join(errors))
    return RightsRecord('valid', digest, document, None)
