import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def field_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path.parent != root or Path(name).name != name:
        raise ValueError("FIELD_PATH_INVALID")
    return path


def read_fields(root, identity, token):
    root = Path(root)
    path = root / "fields.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_bytes())
    if document["identity"] != identity:
        raise RuntimeError("FIELD_IDENTITY_MISMATCH")
    offset = 0
    for record in document["chunks"]:
        token.checkpoint()
        if record["offset"] != offset or record["frames"] < 1:
            raise RuntimeError("FIELD_COVERAGE_INVALID")
        chunk = field_path(root, record["name"])
        if not chunk.is_file() or file_hash(chunk) != record["sha256"]:
            raise RuntimeError("FIELD_HASH_INVALID")
        offset += record["frames"]
    if offset != identity["frames"]:
        raise RuntimeError("FIELD_COVERAGE_INCOMPLETE")
    return document


def field_blocks(root, document, token):
    for record in document["chunks"]:
        token.checkpoint()
        with np.load(field_path(root, record["name"]), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        yield record["offset"], arrays


class FieldWriter:
    def __init__(self, root, identity, token):
        self.root = Path(root).resolve()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.stage = self.root.with_name(self.root.name + "-" + uuid.uuid4().hex + ".partial")
        self.stage.mkdir()
        self.identity = identity
        self.token = token
        self.chunks = []

    def append(self, offset, spectra):
        self.token.checkpoint()
        frames = spectra[0].shape[0]
        arrays = {"X_" + str(a): np.ascontiguousarray(values, dtype="<c16") for a, values in enumerate(spectra)}
        for a in range(len(spectra)):
            for b in range(a + 1, len(spectra)):
                arrays["Q_" + str(a) + "_" + str(b)] = arrays["X_" + str(a)] * np.conj(arrays["X_" + str(b)])
        name = str(offset).zfill(12) + ".npz"
        path = self.stage / name
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        self.chunks.append({"name": name, "offset": offset, "frames": frames, "sha256": file_hash(path), "bytes": path.stat().st_size})

    def commit(self):
        self.token.checkpoint()
        data = {"schema_version": 1, "identity": self.identity, "chunks": self.chunks}
        with (self.stage / "fields.json").open("wb") as stream:
            stream.write(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
        read_fields(self.stage, self.identity, self.token)
        self.token.checkpoint()
        if self.root.exists():
            raise RuntimeError("FIELD_TARGET_EXISTS")
        os.replace(self.stage, self.root)
        return data

    def close(self):
        if self.stage.exists():
            if self.stage.parent != self.root.parent or self.stage == self.root:
                raise RuntimeError("FIELD_STAGE_PATH_INVALID")
            shutil.rmtree(self.stage)
