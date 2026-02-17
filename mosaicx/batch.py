"""Batch processing engine with checkpointing."""

from __future__ import annotations

import json
import threading
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
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BatchCheckpoint:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cp = cls(batch_id=data["batch_id"], checkpoint_dir=Path(path).parent)
        cp._completed = data.get("completed", {})
        return cp


class BatchProcessor:
    """Parallel document processor with error isolation."""

    MAX_WORKERS = 32

    def __init__(self, workers: int = 4, checkpoint_every: int = 50) -> None:
        self.workers = min(max(1, workers), self.MAX_WORKERS)
        self.checkpoint_every = checkpoint_every

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        process_fn: Callable[[str], dict[str, Any]],
        resume_id: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        load_fn: Optional[Callable[[Path], str]] = None,
        on_progress: Optional[Callable[[str, bool], None]] = None,
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
            on_progress: Callback(doc_name, success) called after each document.

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
                        if on_progress:
                            on_progress(d.name, True)
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

        # Load documents sequentially — pypdfium2 is not thread-safe.
        # Only LLM extraction is parallelized.
        loaded: list[tuple[Path, str | None, str | None]] = []
        for doc_path in docs:
            try:
                text = load_fn(doc_path)
                loaded.append((doc_path, text, None))
            except Exception as exc:
                loaded.append((doc_path, None, str(exc)))

        # Skip docs that failed to load
        to_process = [(p, t) for p, t, e in loaded if e is None and t]
        for doc_path, _, err_msg in loaded:
            if err_msg is not None:
                failed += 1
                errors.append({"file": doc_path.name, "error": err_msg})
                if on_progress:
                    on_progress(doc_path.name, False)

        _write_lock = threading.Lock()

        def _process_one(doc_path: Path, text: str) -> tuple[str, dict | None, str | None]:
            try:
                result = process_fn(text)
                out_path = output_dir / f"{doc_path.stem}.json"
                data = json.dumps(result, indent=2, default=str, ensure_ascii=False)
                with _write_lock:
                    out_path.write_text(data, encoding="utf-8")
                return doc_path.name, result, None
            except Exception as exc:
                return doc_path.name, None, f"{type(exc).__name__}: {exc}"

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(_process_one, p, t): p for p, t in to_process}
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
                    if on_progress:
                        on_progress(name, error is None)
        finally:
            # Always save checkpoint on exit (including Ctrl+C)
            if checkpoint:
                checkpoint.save()

        return {
            "total": len(docs) + skipped,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
        }
