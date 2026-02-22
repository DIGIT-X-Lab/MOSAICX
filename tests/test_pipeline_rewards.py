# tests/test_pipeline_rewards.py
from __future__ import annotations


class TestFindingsReward:
    def test_empty_findings_scores_low(self):
        from mosaicx.pipelines.rewards import findings_reward

        score = findings_reward(findings=[])
        assert score < 0.3

    def test_findings_with_anatomy_scores_higher(self):
        from mosaicx.pipelines.rewards import findings_reward

        findings = [
            {"anatomy": "right upper lobe", "observation": "nodule", "description": "5mm nodule"},
            {"anatomy": "left lower lobe", "observation": "atelectasis", "description": "mild"},
        ]
        score = findings_reward(findings=findings)
        assert score > 0.5

    def test_findings_with_measurements_get_bonus(self):
        from mosaicx.pipelines.rewards import findings_reward

        no_measure = [{"anatomy": "RUL", "observation": "nodule", "description": "nodule"}]
        with_measure = [{"anatomy": "RUL", "observation": "nodule", "description": "5mm nodule"}]
        assert findings_reward(findings=with_measure) > findings_reward(findings=no_measure)

    def test_reward_capped_at_one(self):
        from mosaicx.pipelines.rewards import findings_reward

        many_findings = [
            {"anatomy": f"loc{i}", "observation": "finding", "description": f"{i}mm mass"}
            for i in range(20)
        ]
        score = findings_reward(findings=many_findings)
        assert score <= 1.0


class TestImpressionReward:
    def test_empty_impression_scores_zero(self):
        from mosaicx.pipelines.rewards import impression_reward

        assert impression_reward(impressions=[]) == 0.0

    def test_nonempty_impression_scores_positive(self):
        from mosaicx.pipelines.rewards import impression_reward

        impressions = [{"statement": "Pulmonary nodule, recommend follow-up.", "actionable": True}]
        assert impression_reward(impressions=impressions) > 0.0


class TestDiagnosisReward:
    def test_empty_diagnosis_scores_zero(self):
        from mosaicx.pipelines.rewards import diagnosis_reward

        assert diagnosis_reward(diagnoses=[]) == 0.0

    def test_diagnosis_with_detail_scores_high(self):
        from mosaicx.pipelines.rewards import diagnosis_reward

        diagnoses = [{"diagnosis": "Adenocarcinoma", "grade": "well-differentiated", "margin": "negative"}]
        assert diagnosis_reward(diagnoses=diagnoses) > 0.5


class TestSchemaComplianceReward:
    def test_all_required_fields_present(self):
        from mosaicx.pipelines.rewards import schema_compliance_reward

        schema_fields = ["indication", "findings", "impression"]
        extraction = {"indication": "cough", "findings": "nodule", "impression": "follow-up"}
        assert schema_compliance_reward(extraction=extraction, required_fields=schema_fields) == 1.0

    def test_missing_required_field_penalized(self):
        from mosaicx.pipelines.rewards import schema_compliance_reward

        schema_fields = ["indication", "findings", "impression"]
        extraction = {"indication": "cough", "findings": "nodule"}
        score = schema_compliance_reward(extraction=extraction, required_fields=schema_fields)
        assert score < 1.0
