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


class TestPathologyPipelineSignatures:
    def test_classify_specimen_type_signature(self):
        from mosaicx.pipelines.pathology import ClassifySpecimenType
        assert "report_header" in ClassifySpecimenType.input_fields
        assert "specimen_type" in ClassifySpecimenType.output_fields

    def test_parse_path_sections_signature(self):
        from mosaicx.pipelines.pathology import ParsePathSections
        assert "report_text" in ParsePathSections.input_fields
        assert "sections" in ParsePathSections.output_fields

    def test_extract_specimen_details_signature(self):
        from mosaicx.pipelines.pathology import ExtractSpecimenDetails
        assert "gross_text" in ExtractSpecimenDetails.input_fields
        assert "site" in ExtractSpecimenDetails.output_fields

    def test_extract_microscopic_findings_signature(self):
        from mosaicx.pipelines.pathology import ExtractMicroscopicFindings
        assert "microscopic_text" in ExtractMicroscopicFindings.input_fields
        assert "findings" in ExtractMicroscopicFindings.output_fields

    def test_extract_path_diagnosis_signature(self):
        from mosaicx.pipelines.pathology import ExtractPathDiagnosis
        assert "diagnosis_text" in ExtractPathDiagnosis.input_fields
        assert "diagnoses" in ExtractPathDiagnosis.output_fields


class TestPathologyStructurerModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.pathology import PathologyReportStructurer
        pipeline = PathologyReportStructurer()
        assert hasattr(pipeline, "classify_specimen")
        assert hasattr(pipeline, "parse_sections")
        assert hasattr(pipeline, "extract_specimen_details")
        assert hasattr(pipeline, "extract_findings")
        assert hasattr(pipeline, "extract_diagnosis")
