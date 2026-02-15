"""Batch processing engine with checkpointing."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


class BatchCheckpoint:
    """Checkpoint for crash-safe batch resume."""

    def __init__(self, batch_id: str, checkpoint_dir: Path) -> None:
        self.batch_id = batch_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self._completed: dict[str, dict[str, Any]] = {}

    def mark_completed(self, doc_name: str, result: dict[str, Any]) -> None:
        self._completed[doc_name] = result

    def is_completed(self, doc_name: str) -> bool:
        return doc_name in self._completed

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def save(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.batch_id}.json"
        data = {"batch_id": self.batch_id, "completed": self._completed,
                "saved_at": datetime.now().isoformat()}
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> BatchCheckpoint:
        data = json.loads(Path(path).read_text())
        cp = cls(batch_id=data["batch_id"], checkpoint_dir=Path(path).parent)
        cp._completed = data.get("completed", {})
        return cp


class BatchProcessor:
    """Parallel document processor with error isolation."""

    def __init__(self, workers: int = 4, checkpoint_every: int = 50) -> None:
        self.workers = workers
        self.checkpoint_every = checkpoint_every

    def process_directory(self, input_dir: Path, output_dir: Path,
                          process_fn: Callable, template: str = "auto",
                          resume_id: Optional[str] = None) -> dict[str, Any]:
        """Process all documents in a directory. Stub — full impl in later phase."""
        return {"status": "not_yet_implemented", "input_dir": str(input_dir)}
