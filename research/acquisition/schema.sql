PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;

CREATE TABLE IF NOT EXISTS policy_versions (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256)=64),
    document_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_objects (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256)=64),
    bytes INTEGER NOT NULL CHECK (bytes>=0),
    object_path TEXT NOT NULL UNIQUE,
    source_url TEXT,
    captured_at TEXT NOT NULL,
    media_type TEXT NOT NULL,
    response_receipt_sha256 TEXT,
    classification TEXT NOT NULL CHECK (classification IN
        ('measurement','publisher_statement','license','transport_receipt','third_party_metadata','test_result'))
);
CREATE TABLE IF NOT EXISTS authorities (
    id TEXT PRIMARY KEY,
    official_domain TEXT NOT NULL,
    binding_evidence_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    manifest_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    status TEXT NOT NULL CHECK(status IN ('supported','withdrawn','conflict'))
);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK(source_kind IN ('internet_archive','artist_label_manifest','local_authorized_export')),
    source_item_id TEXT NOT NULL,
    exact_file_name TEXT NOT NULL,
    discovery_evidence_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    authority_id TEXT REFERENCES authorities(id),
    state TEXT NOT NULL CHECK(state IN
        ('discovered','evidence_missing','rights_supported','planned','downloading','downloaded',
         'pcm_verified','lineage_verified','audited','selected','published','excluded','quarantined','retry_exhausted')),
    source_sha256 TEXT,
    decoded_sha256 TEXT,
    sample_rate INTEGER CHECK(sample_rate>0),
    channels INTEGER CHECK(channels>0),
    frames INTEGER CHECK(frames>0),
    primary_eligible INTEGER NOT NULL DEFAULT 0 CHECK(primary_eligible IN (0,1)),
    UNIQUE(source_kind,source_item_id,exact_file_name)
);
CREATE TABLE IF NOT EXISTS evidence_claims (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    predicate TEXT NOT NULL,
    document_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    locator_json TEXT NOT NULL,
    authority_id TEXT REFERENCES authorities(id),
    statement_json TEXT NOT NULL,
    verification_code_sha256 TEXT NOT NULL CHECK(length(verification_code_sha256)=64),
    status TEXT NOT NULL CHECK(status IN ('supported','missing','conflict','failed')),
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256)
);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY CHECK(length(id)=64),
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    discovery_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    evidence_inventory_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256),
    verifier_sha256 TEXT NOT NULL CHECK(length(verifier_sha256)=64),
    status TEXT NOT NULL CHECK(status IN ('processing','complete','blocked','quarantined')),
    UNIQUE(id,candidate_id,policy_sha256,verifier_sha256),
    UNIQUE(candidate_id,discovery_sha256,evidence_inventory_sha256,policy_sha256,verifier_sha256)
);
CREATE TABLE IF NOT EXISTS download_attempts (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    partial_path TEXT NOT NULL,
    bytes_written INTEGER NOT NULL DEFAULT 0 CHECK(bytes_written>=0),
    expected_bytes INTEGER NOT NULL CHECK(expected_bytes>0),
    strong_etag TEXT,
    redirect_receipt_sha256 TEXT REFERENCES evidence_objects(sha256),
    status TEXT NOT NULL CHECK(status IN ('processing','complete','interrupted','failed')),
    last_error_code TEXT
);
CREATE TABLE IF NOT EXISTS lineage_nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('work','recording','master_family','shared_recorded_material','release','creator')),
    identity_document_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256)
);
CREATE TABLE IF NOT EXISTS candidate_lineage (
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    node_id TEXT NOT NULL REFERENCES lineage_nodes(id),
    PRIMARY KEY(candidate_id,node_id)
);
CREATE TABLE IF NOT EXISTS relation_edges (
    left_candidate_id TEXT NOT NULL REFERENCES candidates(id),
    right_candidate_id TEXT NOT NULL REFERENCES candidates(id),
    relation TEXT NOT NULL CHECK(relation IN
        ('same_file','same_pcm','same_work','same_master_family','shared_recorded_material','suspected_audio_relation','identity_conflict')),
    evidence_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    status TEXT NOT NULL CHECK(status IN ('confirmed','unresolved')),
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256),
    PRIMARY KEY(left_candidate_id,right_candidate_id,relation,evidence_sha256,policy_sha256),
    CHECK(left_candidate_id<right_candidate_id)
);
CREATE TABLE IF NOT EXISTS gate_decisions (
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    gate TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('passed','missing','failed','conflict')),
    evidence_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256),
    verifier_sha256 TEXT NOT NULL CHECK(length(verifier_sha256)=64),
    PRIMARY KEY(evaluation_id,gate),
    FOREIGN KEY(evaluation_id,candidate_id,policy_sha256,verifier_sha256)
        REFERENCES evaluations(id,candidate_id,policy_sha256,verifier_sha256)
);
CREATE TABLE IF NOT EXISTS corpus_manifests (
    sha256 TEXT PRIMARY KEY CHECK(length(sha256)=64),
    kind TEXT NOT NULL CHECK(kind IN ('development','primary','replication')),
    object_path TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256),
    discovery_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    relation_graph_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    backup_receipt_sha256 TEXT NOT NULL REFERENCES evidence_objects(sha256),
    song_count INTEGER NOT NULL CHECK(song_count>0),
    frozen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replenishment_requests (
    id TEXT PRIMARY KEY CHECK(length(id)=64),
    analysis_plan_sha256 TEXT NOT NULL CHECK(length(analysis_plan_sha256)=64),
    policy_sha256 TEXT NOT NULL REFERENCES policy_versions(sha256),
    required_primary INTEGER NOT NULL CHECK(required_primary BETWEEN 1 AND 9999),
    required_replication INTEGER NOT NULL CHECK(required_replication BETWEEN 1 AND 9999),
    parent_primary_sha256 TEXT REFERENCES corpus_manifests(sha256),
    parent_replication_sha256 TEXT REFERENCES corpus_manifests(sha256),
    status TEXT NOT NULL CHECK(status IN
        ('queued','processing','waiting_retry','supply_unavailable','blocked_policy',
         'blocked_storage','blocked_integrity','fulfilled','invalidated')),
    created_at TEXT NOT NULL,
    next_attempt_at TEXT,
    result_receipt_sha256 TEXT REFERENCES evidence_objects(sha256)
);
CREATE TABLE IF NOT EXISTS source_cursors (
    source_id TEXT PRIMARY KEY,
    authority_id TEXT REFERENCES authorities(id),
    cursor_json TEXT NOT NULL,
    last_inventory_sha256 TEXT REFERENCES evidence_objects(sha256),
    next_refresh_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready','waiting_retry','unsupported','exhausted'))
);
CREATE TABLE IF NOT EXISTS worker_jobs (
    id TEXT PRIMARY KEY CHECK(length(id)=64),
    request_id TEXT NOT NULL REFERENCES replenishment_requests(id),
    candidate_id TEXT REFERENCES candidates(id),
    stage TEXT NOT NULL CHECK(stage IN
        ('discover','verify_claims','download','verify_pcm','resolve_lineage','fingerprint',
         'compare_fingerprints','compile','backup','publish')),
    prerequisite_sha256 TEXT NOT NULL CHECK(length(prerequisite_sha256)=64),
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation>=0),
    status TEXT NOT NULL CHECK(status IN ('queued','running','waiting_retry','complete','blocked')),
    next_attempt_at TEXT,
    last_reason_code TEXT,
    UNIQUE(request_id,candidate_id,stage,prerequisite_sha256)
);
CREATE TABLE IF NOT EXISTS component_reservations (
    id TEXT PRIMARY KEY CHECK(length(id)=64),
    graph_component_sha256 TEXT NOT NULL CHECK(length(graph_component_sha256)=64),
    split TEXT NOT NULL CHECK(split IN ('development','primary','replication')),
    selected_candidate_id TEXT NOT NULL REFERENCES candidates(id),
    first_manifest_sha256 TEXT NOT NULL REFERENCES corpus_manifests(sha256)
);
CREATE TABLE IF NOT EXISTS reservation_members (
    reservation_id TEXT NOT NULL REFERENCES component_reservations(id),
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    PRIMARY KEY(reservation_id,candidate_id)
);
CREATE TABLE IF NOT EXISTS corpus_lifecycle (
    manifest_sha256 TEXT PRIMARY KEY REFERENCES corpus_manifests(sha256),
    parent_manifest_sha256 TEXT REFERENCES corpus_manifests(sha256),
    status TEXT NOT NULL CHECK(status IN ('valid','invalidated')),
    reason_receipt_sha256 TEXT REFERENCES evidence_objects(sha256)
);
