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


class TestBatchProcessorReal:
    def test_process_empty_directory(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=lambda text: {"extracted": "test"},
        )
        assert result["total"] == 0
        assert result["succeeded"] == 0

    def test_process_txt_files(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        (input_dir / "doc1.txt").write_text("Patient report 1")
        (input_dir / "doc2.txt").write_text("Patient report 2")
        (input_dir / "not_a_doc.xyz").write_text("skip this")

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=lambda text: {"summary": "extracted"},
        )
        assert result["total"] == 2
        assert result["succeeded"] == 2
        assert output_dir.exists()

    def test_process_with_error_isolation(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        (input_dir / "good.txt").write_text("Good report")
        (input_dir / "bad.txt").write_text("Bad report")

        def flaky_fn(text):
            if "Bad" in text:
                raise ValueError("Simulated extraction failure")
            return {"result": "ok"}

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=flaky_fn,
        )
        assert result["succeeded"] == 1
        assert result["failed"] == 1

    def test_process_with_resume(self, tmp_path):
        from mosaicx.batch import BatchProcessor, BatchCheckpoint
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        checkpoint_dir = tmp_path / "checkpoints"

        (input_dir / "doc1.txt").write_text("Report 1")
        (input_dir / "doc2.txt").write_text("Report 2")

        cp = BatchCheckpoint(batch_id="resume", checkpoint_dir=checkpoint_dir)
        cp.mark_completed("doc1.txt", {"status": "ok"})
        cp.save()

        call_count = 0
        def counting_fn(text):
            nonlocal call_count
            call_count += 1
            return {"result": "ok"}

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=counting_fn,
            resume_id="resume",
            checkpoint_dir=checkpoint_dir,
        )
        assert call_count == 1
        assert result["skipped"] == 1
