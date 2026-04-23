"""Batch processing engine with checkpointing.

Uses a pipelined architecture: OCR producers feed a queue that extraction
workers (parallel consumers) drain concurrently. With ``ocr_workers >= 2``
multiple documents are OCR'd in parallel, which is important when the OCR
backend (e.g. Chandra on a dedicated GPU) has headroom to serve several
documents' worth of pages at once. PDFium calls are serialized internally
via a module-wide lock (``mosaicx.documents._pdfium.PDFIUM_LOCK``) since
the underlying library is not thread-safe; the lock is held only while
rasterizing, not across the OCR HTTP calls.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SENTINEL = None  # signals OCR producer is done


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
    """Pipelined document processor: OCR producers -> extraction consumers."""

    MAX_WORKERS = 32
    MAX_OCR_WORKERS = 16

    def __init__(
        self,
        workers: int = 4,
        checkpoint_every: int = 50,
        ocr_workers: int = 1,
    ) -> None:
        self.workers = min(max(1, workers), self.MAX_WORKERS)
        self.ocr_workers = min(max(1, ocr_workers), self.MAX_OCR_WORKERS)
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

        OCR runs in a pool of up to ``ocr_workers`` threads (PDFium calls
        are serialized internally via a lock; OCR HTTP calls overlap).
        Results feed a queue that ``workers`` extraction consumers drain
        in parallel, so OCR, LLM extraction, and JSON writing all overlap.

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

        # --- Pipeline: OCR producers -> extraction consumers ---

        # Queue holds (doc_path, text) tuples; _SENTINEL signals end.
        # Size the queue against the *slower* stage so neither starves:
        # the consumer pool (extraction workers) is usually larger.
        ocr_queue: queue.Queue[tuple[Path, str] | None] = queue.Queue(
            maxsize=max(self.workers, self.ocr_workers) * 2,
        )
        ocr_errors: list[tuple[Path, str]] = []
        _errors_lock = threading.Lock()

        def _ocr_one(doc_path: Path) -> None:
            """Load and OCR a single document, then enqueue for extraction.

            Called from multiple OCR producer threads concurrently.
            Exceptions are collected, not raised, so one bad document
            does not take down the whole batch.
            """
            try:
                text = load_fn(doc_path)
                if text:
                    ocr_queue.put((doc_path, text))
                else:
                    with _errors_lock:
                        ocr_errors.append((doc_path, "Empty document"))
            except Exception as exc:
                with _errors_lock:
                    ocr_errors.append((doc_path, str(exc)))

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

        # Fan OCR across up to self.ocr_workers threads; a single
        # coordinator thread waits for all OCR tasks to finish and
        # then signals end-of-stream with _SENTINEL.
        ocr_pool = ThreadPoolExecutor(
            max_workers=self.ocr_workers,
            thread_name_prefix="mosaicx-ocr",
        )
        ocr_futures = [ocr_pool.submit(_ocr_one, d) for d in docs]

        def _ocr_coordinator() -> None:
            for f in as_completed(ocr_futures):
                # _ocr_one captures its own exceptions; .result() is safe.
                f.result()
            ocr_queue.put(_SENTINEL)

        coordinator = threading.Thread(
            target=_ocr_coordinator,
            daemon=True,
            name="mosaicx-ocr-coordinator",
        )
        coordinator.start()

        # Extraction consumers pull from queue as OCR results arrive
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures: dict[Any, Path] = {}
                processed = 0
                producer_done = False

                while not producer_done or futures:
                    # Submit new work from OCR queue while we have capacity
                    while not producer_done and len(futures) < self.workers:
                        try:
                            item = ocr_queue.get(timeout=0.1)
                        except queue.Empty:
                            break
                        if item is _SENTINEL:
                            producer_done = True
                            break
                        doc_path, text = item
                        fut = pool.submit(_process_one, doc_path, text)
                        futures[fut] = doc_path

                    # Collect completed extraction results (non-blocking)
                    done = [f for f in futures if f.done()]
                    for future in done:
                        name, result, error = future.result()
                        processed += 1
                        del futures[future]
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

            # Report OCR failures
            with _errors_lock:
                ocr_error_snapshot = list(ocr_errors)
            for doc_path, err_msg in ocr_error_snapshot:
                failed += 1
                errors.append({"file": doc_path.name, "error": err_msg})
                if on_progress:
                    on_progress(doc_path.name, False)

        finally:
            coordinator.join(timeout=5)
            ocr_pool.shutdown(wait=False)
            if checkpoint:
                checkpoint.save()

        return {
            "total": len(docs) + skipped,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
        }
