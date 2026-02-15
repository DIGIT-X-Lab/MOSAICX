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

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        process_fn: Callable[[str], dict[str, Any]],
        resume_id: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        load_fn: Optional[Callable[[Path], str]] = None,
    ) -> dict[str, Any]:
        """Process all documents in a directory.

        Args:
            input_dir: Directory of input documents.
            output_dir: Directory for output JSON files.
            process_fn: Function that takes document text and returns a dict.
            resume_id: If set, resume from checkpoint with this ID.
            checkpoint_dir: Directory for checkpoint files.
            load_fn: Function that takes a Path and returns text. Defaults to
                      the document loader (handles PDFs, images, text files).

        Returns:
            Summary dict with keys: total, succeeded, failed, skipped, errors.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover documents
        supported = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        docs = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        )

        # Load checkpoint if resuming
        checkpoint: BatchCheckpoint | None = None
        skipped = 0
        if resume_id:
            ckpt_dir = checkpoint_dir or output_dir
            cp_path = ckpt_dir / f"{resume_id}.json"
            if cp_path.exists():
                checkpoint = BatchCheckpoint.load(cp_path)
                remaining = []
                for d in docs:
                    if checkpoint.is_completed(d.name):
                        skipped += 1
                    else:
                        remaining.append(d)
                docs = remaining
            else:
                checkpoint = BatchCheckpoint(batch_id=resume_id, checkpoint_dir=ckpt_dir)

        succeeded = 0
        failed = 0
        errors: list[dict[str, str]] = []

        if load_fn is None:
            from .documents.loader import load_document
            def _default_load(path: Path) -> str:
                return load_document(path).text
            load_fn = _default_load

        def _process_one(doc_path: Path) -> tuple[str, dict | None, str | None]:
            try:
                text = load_fn(doc_path)
                result = process_fn(text)
                out_path = output_dir / f"{doc_path.stem}.json"
                out_path.write_text(
                    json.dumps(result, indent=2, default=str), encoding="utf-8"
                )
                return doc_path.name, result, None
            except Exception as exc:
                return doc_path.name, None, str(exc)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_process_one, d): d for d in docs}
            processed = 0
            for future in as_completed(futures):
                name, result, error = future.result()
                processed += 1
                if error:
                    failed += 1
                    errors.append({"file": name, "error": error})
                else:
                    succeeded += 1
                    if checkpoint:
                        checkpoint.mark_completed(name, result or {})
                        if processed % self.checkpoint_every == 0:
                            checkpoint.save()

        if checkpoint:
            checkpoint.save()

        return {
            "total": len(docs) + skipped,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
        }
