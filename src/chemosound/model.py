from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import Event
AUDIO_EXTENSIONS = frozenset({'.flac', '.wav', '.wave', '.aif', '.aiff'})
PERMISSION_SCOPE = frozenset({'local_storage', 'computational_analysis', 'research_archiving', 'publish_nonreconstructive_results'})

class Cancelled(RuntimeError):
    pass

@dataclass(slots=True)
class CancellationToken:
    event: Event

    @classmethod
    def create(cls) -> CancellationToken:
        return cls(Event())

    def cancel(self) -> None:
        self.event.set()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise Cancelled('cancellation requested')

@dataclass(frozen=True, slots=True)
class Paths:
    state_dir: Path

    @property
    def database(self) -> Path:
        return self.state_dir / 'state.sqlite3'

    @property
    def staging(self) -> Path:
        return self.state_dir / 'staging'

    @property
    def validation_results(self) -> Path:
        return self.state_dir / 'results' / 'validate_audio'

    @property
    def relation_results(self) -> Path:
        return self.state_dir / 'results' / 'music_relation_v1'

    @property
    def similarity_results(self) -> Path:
        return self.state_dir / 'results' / 'exploratory_similarity_v1'

    def ensure(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.validation_results.mkdir(parents=True, exist_ok=True)
        self.relation_results.mkdir(parents=True, exist_ok=True)
        self.similarity_results.mkdir(parents=True, exist_ok=True)
