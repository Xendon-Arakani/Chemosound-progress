import argparse
import hashlib
import importlib.util
import json
import platform
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="2026-10-03")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not args.batch or "/" in args.batch or "\\" in args.batch or args.batch in {".", ".."}:
        parser.error("Invalid batch")
    batch = ROOT / "local-data/song-acquisition" / args.batch
    with sqlite3.connect(f'{(batch / "ledger.sqlite3").as_uri()}?mode=ro', uri=True) as db:
        records = [json.loads(row[0]) for row in db.execute("SELECT record_json FROM records ORDER BY rowid")]
        decisions = [{"record_or_item_id": key, "state": state, **json.loads(detail)} for key, state, detail in db.execute("SELECT id,state,detail_json FROM decisions ORDER BY id")]
        integrity = db.execute("PRAGMA integrity_check").fetchall()
    spec = importlib.util.spec_from_file_location("surface_acquisition", ROOT / "tools/get_surface_audio.py")
    acquisition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acquisition)
    policy = json.loads((batch / "batch-policy.json").read_text(encoding="utf-8"))
    geometries = Counter((row["pcm_measurements"]["sample_rate"], row["pcm_measurements"]["bits_per_sample"], row["pcm_measurements"]["channels"]) for row in records)
    containers = Counter(Path(row["audio_relative_path"]).suffix.lower() for row in records)
    durations = [row["pcm_measurements"]["frames"] / row["pcm_measurements"]["sample_rate"] for row in records]
    counts = Counter(row["state"] for row in decisions)
    stats = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "recordings": len(records),
        "release_count": len({row["archive_identifier"] for row in records}),
        "audio_bytes": sum(row["download_receipt"]["bytes"] for row in records),
        "audio_hours": sum(durations) / 3600,
        "containers": dict(containers),
        "geometry_counts": [{"sample_rate": rate, "bits": bits, "channels": channels, "count": count} for (rate, bits, channels), count in sorted(geometries.items())],
        "decision_counts": dict(counts),
        "primary_eligible": False,
        "ledger_integrity": integrity,
        "runtime": {"python": platform.python_version(), "numpy": acquisition.audit.np.__version__, "soundfile": acquisition.audit.sf.__version__, "libsndfile": acquisition.audit.sf.__libsndfile_version__},
    }
    if args.finalize:
        if integrity != [("ok",)]:
            raise RuntimeError("Ledger integrity failed")
        if len({row["pcm_measurements"]["decoded_sha256"] for row in records}) != len(records):
            raise RuntimeError("Duplicate PCM identities")
        for row in records:
            source = batch / row["audio_relative_path"]
            if source.stat().st_size != row["download_receipt"]["bytes"] or digest(source) != row["source_sha256"]:
                raise RuntimeError("Audio source identity failed")
            if row["primary_inference_eligible"] is not False or row["reference_native_pcm_bytes_equal"] is not True:
                raise RuntimeError("Audio source state failed")
            sidecar = json.loads((batch / row["evidence_relative_path"]).read_text(encoding="utf-8"))
            if sidecar != row:
                raise RuntimeError("Audio receipt mismatch")
            for field in ("metadata_response_receipt", "discovery_receipt", "source_page_receipt"):
                receipt = row[field]
                if receipt.get("sha256") and digest(batch / "evidence" / receipt["sha256"]) != receipt["sha256"]:
                    raise RuntimeError("Source receipt identity failed")
        prior = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
        acquisition.export_reports(batch, records, target=policy["target"], transferred=prior["this_process_transfer_bytes"])
        inventory = []
        for row in records:
            inventory.extend([{"path": row["audio_relative_path"], "sha256": row["source_sha256"]}, {"path": row["evidence_relative_path"], "sha256": digest(batch / row["evidence_relative_path"])}])
        acquisition.atomic(batch / "completed-file-manifest.json", acquisition.canonical(inventory))
        stats["source_rehash_passed"] = True
        stats["target_met"] = len(records) >= policy["target"]
    artifacts = [{"path": str(path.relative_to(batch)), "bytes": path.stat().st_size, "completed": False} for folder in ("staging", "rejected") for path in sorted((batch / folder).glob("*")) if path.is_file()]
    acquisition.atomic(batch / "decisions.json", acquisition.canonical(decisions))
    acquisition.atomic(batch / "unaccepted-artifacts.json", acquisition.canonical(artifacts))
    acquisition.atomic(batch / "validation.json", acquisition.canonical(stats))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
