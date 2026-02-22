from __future__ import annotations


class TestDeterministicVerify:
    def test_measurement_found_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 12.0, "unit": "mm"}}]}
        source = "Right external iliac node enlarged (short axis 12 mm)."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"
        assert len(report.issues) == 0

    def test_measurement_not_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 22.0, "unit": "mm"}}]}
        source = "Right external iliac node enlarged (short axis 12 mm)."
        report = verify_deterministic(extraction, source)
        assert report.verdict != "verified"
        assert any(i.type == "value_not_found" for i in report.issues)

    def test_empty_extraction_passes(self):
        from mosaicx.verify.deterministic import verify_deterministic

        report = verify_deterministic({}, "Some source text.")
        assert report.verdict == "verified"

    def test_invalid_finding_ref(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {
            "findings": [{"anatomy": "RUL"}],
            "impressions": [{"finding_refs": [5]}],  # index 5 doesn't exist
        }
        source = "Some source."
        report = verify_deterministic(extraction, source)
        assert any(i.type == "invalid_reference" for i in report.issues)

    def test_valid_finding_ref(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {
            "findings": [{"anatomy": "RUL"}, {"anatomy": "LLL"}],
            "impressions": [{"finding_refs": [0, 1]}],
        }
        source = "Some source."
        report = verify_deterministic(extraction, source)
        assert not any(i.type == "invalid_reference" for i in report.issues)

    def test_confidence_decreases_with_issues(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {
            "findings": [
                {"measurement": {"value": 99.0, "unit": "mm"}},
                {"measurement": {"value": 88.0, "unit": "mm"}},
            ],
        }
        source = "No numbers here."
        report = verify_deterministic(extraction, source)
        assert report.confidence < 1.0
        assert report.level == "deterministic"

    def test_measurement_none_value_skipped(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": None, "unit": "mm"}}]}
        source = "Some source text."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"

    def test_measurement_non_dict_skipped(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": "12 mm"}]}
        source = "Some source text."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"

    def test_finding_refs_non_list_skipped(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {
            "findings": [{"anatomy": "RUL"}],
            "impressions": [{"finding_refs": "not a list"}],
        }
        source = "Some source."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"

    def test_integer_value_found_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 5, "unit": "cm"}}]}
        source = "Mass measures 5 cm in greatest dimension."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"

    def test_float_value_found_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 3.5, "unit": "cm"}}]}
        source = "Lesion measuring 3.5 cm."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"
