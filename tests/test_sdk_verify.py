from __future__ import annotations


class TestSDKVerify:
    def test_verify_function_exists(self):
        from mosaicx.sdk import verify

        assert callable(verify)

    def test_verify_quick_no_llm(self):
        """Quick verification should work without DSPy configured."""
        from mosaicx.sdk import verify

        result = verify(
            extraction={"findings": []},
            source_text="Normal report.",
            level="quick",
        )
        assert "verdict" in result
        assert result["verdict"] == "verified"

    def test_verify_claim(self):
        from mosaicx.sdk import verify

        result = verify(
            claim="Blood pressure was 120/80",
            source_text="Vitals: BP 120/80 mmHg.",
            level="quick",
        )
        assert "verdict" in result

    def test_verify_returns_dict(self):
        from mosaicx.sdk import verify

        result = verify(
            extraction={"findings": [{"measurement": {"value": 5, "unit": "mm"}}]},
            source_text="5mm nodule in RUL.",
            level="quick",
        )
        assert isinstance(result, dict)
        assert "confidence" in result
        assert "level" in result
        assert "issues" in result
