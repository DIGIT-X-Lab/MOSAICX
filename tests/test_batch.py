"""Tests for the batch processing engine."""
import pytest
import json
from pathlib import Path


class TestBatchCheckpoint:
    def test_checkpoint_save_load(self, tmp_path):
        from mosaicx.batch import BatchCheckpoint
        cp = BatchCheckpoint(batch_id="test_batch", checkpoint_dir=tmp_path)
        cp.mark_completed("doc1.pdf", {"status": "ok"})
        cp.mark_completed("doc2.pdf", {"status": "ok"})
        cp.save()

        cp2 = BatchCheckpoint.load(tmp_path / "test_batch.json")
        assert cp2.is_completed("doc1.pdf")
        assert cp2.is_completed("doc2.pdf")
        assert not cp2.is_completed("doc3.pdf")

    def test_checkpoint_resume_skips_completed(self, tmp_path):
        from mosaicx.batch import BatchCheckpoint
        cp = BatchCheckpoint(batch_id="test", checkpoint_dir=tmp_path)
        cp.mark_completed("doc1.pdf", {"status": "ok"})
        cp.save()
        all_docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        remaining = [d for d in all_docs if not cp.is_completed(d)]
        assert remaining == ["doc2.pdf", "doc3.pdf"]

    def test_completed_count(self, tmp_path):
        from mosaicx.batch import BatchCheckpoint
        cp = BatchCheckpoint(batch_id="test", checkpoint_dir=tmp_path)
        assert cp.completed_count == 0
        cp.mark_completed("doc1.pdf", {"status": "ok"})
        assert cp.completed_count == 1


class TestBatchProcessor:
    def test_processor_creation(self):
        from mosaicx.batch import BatchProcessor
        proc = BatchProcessor(workers=2)
        assert proc.workers == 2

    def test_processor_has_process_method(self):
        from mosaicx.batch import BatchProcessor
        proc = BatchProcessor(workers=1)
        assert hasattr(proc, "process_directory")
