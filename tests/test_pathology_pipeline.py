"""Tests for the pathology report structurer pipeline."""
import pytest


class TestPathReportBaseModels:
    def test_path_sections_construction(self):
        from mosaicx.schemas.pathreport.base import PathSections
        s = PathSections(
            clinical_history="56-year-old male with rectal mass",
            gross_description="Received in formalin...",
            microscopic="Sections show moderately differentiated adenocarcinoma",
            diagnosis="Adenocarcinoma of the rectum",
        )
        assert s.clinical_history == "56-year-old male with rectal mass"
        assert s.ancillary_studies == ""

    def test_path_finding_construction(self):
        from mosaicx.schemas.pathreport.base import PathFinding
        f = PathFinding(
            description="Moderately differentiated adenocarcinoma",
            histologic_type="adenocarcinoma",
            grade="G2",
            margins="negative, closest margin 0.3 cm",
            lymphovascular_invasion="present",
        )
        assert f.histologic_type == "adenocarcinoma"
        assert f.perineural_invasion is None

    def test_biomarker_construction(self):
        from mosaicx.schemas.pathreport.base import Biomarker
        b = Biomarker(name="ER", result="positive (95%)", method="IHC")
        assert b.name == "ER"
        assert b.method == "IHC"

    def test_path_diagnosis_construction(self):
        from mosaicx.schemas.pathreport.base import PathDiagnosis
        d = PathDiagnosis(
            diagnosis="Invasive ductal carcinoma, grade 2",
            who_classification="Invasive carcinoma of no special type",
            tnm_stage="pT2 pN1a",
            biomarkers=[],
        )
        assert d.tnm_stage == "pT2 pN1a"
        assert d.icd_o_morphology is None

    def test_path_diagnosis_with_biomarkers(self):
        from mosaicx.schemas.pathreport.base import Biomarker, PathDiagnosis
        d = PathDiagnosis(
            diagnosis="Breast carcinoma",
            biomarkers=[
                Biomarker(name="ER", result="positive"),
                Biomarker(name="PR", result="positive"),
                Biomarker(name="HER2", result="negative"),
                Biomarker(name="Ki-67", result="30%"),
            ],
        )
        assert len(d.biomarkers) == 4
        assert d.biomarkers[0].name == "ER"
