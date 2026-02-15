# tests/test_quality_scorer.py
"""Tests for the OCR quality scorer."""

import pytest


class TestMedicalVocabScore:
    def test_medical_text_scores_high(self):
        from mosaicx.documents.quality import medical_vocab_score

        text = "CT chest shows 5mm nodule in the right upper lobe. No pleural effusion. Heart is normal size."
        score = medical_vocab_score(text)
        assert score > 0.3

    def test_garbled_text_scores_low(self):
        from mosaicx.documents.quality import medical_vocab_score

        text = "Th3 p@t13nt pr3s3nts w1th c0ugh. Xr@y sh0ws n0 @bn0rm@l1ty."
        score = medical_vocab_score(text)
        assert score < 0.1

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import medical_vocab_score

        assert medical_vocab_score("") == 0.0


class TestCharSanityScore:
    def test_clean_text_scores_high(self):
        from mosaicx.documents.quality import char_sanity_score

        text = "The lungs are clear bilaterally. No consolidation or effusion."
        score = char_sanity_score(text)
        assert score > 0.8

    def test_garbled_text_scores_low(self):
        from mosaicx.documents.quality import char_sanity_score

        text = "T#3 l@ng$ @r3 cl3@r b!l@t3r@lly. N0 c0ns0l!d@t!0n."
        score = char_sanity_score(text)
        assert score < 0.7

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import char_sanity_score

        assert char_sanity_score("") == 0.0


class TestWordStructureScore:
    def test_normal_text_scores_high(self):
        from mosaicx.documents.quality import word_structure_score

        text = "Patient presents with persistent cough for three weeks."
        score = word_structure_score(text)
        assert score > 0.7

    def test_fragmented_text_scores_low(self):
        from mosaicx.documents.quality import word_structure_score

        text = "P a t i e n t p r e s e n t s w i t h c o u g h"
        score = word_structure_score(text)
        assert score < 0.5

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import word_structure_score

        assert word_structure_score("") == 0.0


class TestQualityScorer:
    def test_good_medical_text(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        text = (
            "CT chest with contrast. Indication: cough.\n"
            "Findings: 5mm ground glass nodule in the right upper lobe.\n"
            "No pleural effusion. Heart is normal size.\n"
            "Impression: Pulmonary nodule, recommend follow-up."
        )
        score = scorer.score(text)
        assert score > 0.5

    def test_garbled_text_low_score(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        text = "@#$ %^& *() !@# $%^ &*( )!@ #$% ^&*"
        score = scorer.score(text)
        assert score < 0.3

    def test_empty_text_zero(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        assert scorer.score("") == 0.0

    def test_score_in_range(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        for text in ["hello world", "CT scan normal", "", "@@@@"]:
            score = scorer.score(text)
            assert 0.0 <= score <= 1.0
