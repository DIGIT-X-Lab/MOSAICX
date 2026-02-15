# mosaicx/schemas/ontology.py
"""Ontology resolver with local RadLex lookup table.

Provides a lightweight, zero-dependency ontology resolution layer for
mapping free-text anatomical and pathological terms to standardised
vocabulary codes.  Currently supports RadLex; additional vocabularies
(SNOMED CT, ICD-10, LOINC) and LLM-assisted fuzzy resolution are
planned for future releases.

Key components:
    - OntologyResult: Pydantic model for a resolved term.
    - OntologyResolver: Lookup-table resolver with case-insensitive
      matching against a curated RadLex subset.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class OntologyResult(BaseModel):
    """A resolved ontology term with its code and vocabulary."""

    term: str
    code: str
    vocabulary: str
    confidence: float = 1.0


class OntologyResolver:
    """Local lookup table resolver. LLM fallback to be added later."""

    def __init__(self) -> None:
        self._radlex: dict[str, OntologyResult] = {
            # Pulmonary lobes
            "right upper lobe": OntologyResult(term="right upper lobe", code="RID1303", vocabulary="radlex"),
            "right middle lobe": OntologyResult(term="right middle lobe", code="RID1310", vocabulary="radlex"),
            "right lower lobe": OntologyResult(term="right lower lobe", code="RID1315", vocabulary="radlex"),
            "left upper lobe": OntologyResult(term="left upper lobe", code="RID1327", vocabulary="radlex"),
            "left lower lobe": OntologyResult(term="left lower lobe", code="RID1338", vocabulary="radlex"),
            # Thoracic structures
            "lung": OntologyResult(term="lung", code="RID1301", vocabulary="radlex"),
            "heart": OntologyResult(term="heart", code="RID1385", vocabulary="radlex"),
            "aorta": OntologyResult(term="aorta", code="RID480", vocabulary="radlex"),
            "trachea": OntologyResult(term="trachea", code="RID1247", vocabulary="radlex"),
            "esophagus": OntologyResult(term="esophagus", code="RID95", vocabulary="radlex"),
            "mediastinum": OntologyResult(term="mediastinum", code="RID1384", vocabulary="radlex"),
            "diaphragm": OntologyResult(term="diaphragm", code="RID1362", vocabulary="radlex"),
            # Abdominal organs
            "liver": OntologyResult(term="liver", code="RID58", vocabulary="radlex"),
            "spleen": OntologyResult(term="spleen", code="RID86", vocabulary="radlex"),
            "kidney": OntologyResult(term="kidney", code="RID205", vocabulary="radlex"),
            "pancreas": OntologyResult(term="pancreas", code="RID170", vocabulary="radlex"),
            "gallbladder": OntologyResult(term="gallbladder", code="RID187", vocabulary="radlex"),
            "adrenal gland": OntologyResult(term="adrenal gland", code="RID88", vocabulary="radlex"),
            "stomach": OntologyResult(term="stomach", code="RID155", vocabulary="radlex"),
            # Head / neuro
            "brain": OntologyResult(term="brain", code="RID6434", vocabulary="radlex"),
            "cerebellum": OntologyResult(term="cerebellum", code="RID6489", vocabulary="radlex"),
            "pituitary gland": OntologyResult(term="pituitary gland", code="RID6649", vocabulary="radlex"),
            # Endocrine / other
            "thyroid": OntologyResult(term="thyroid", code="RID7578", vocabulary="radlex"),
            # Musculoskeletal
            "vertebral body": OntologyResult(term="vertebral body", code="RID29011", vocabulary="radlex"),
            "rib": OntologyResult(term="rib", code="RID28569", vocabulary="radlex"),
            "sternum": OntologyResult(term="sternum", code="RID2531", vocabulary="radlex"),
            "clavicle": OntologyResult(term="clavicle", code="RID2510", vocabulary="radlex"),
            "scapula": OntologyResult(term="scapula", code="RID2524", vocabulary="radlex"),
            "humerus": OntologyResult(term="humerus", code="RID2619", vocabulary="radlex"),
            "femur": OntologyResult(term="femur", code="RID2660", vocabulary="radlex"),
            # Pathological findings
            "pleural effusion": OntologyResult(term="pleural effusion", code="RID4872", vocabulary="radlex"),
            "pulmonary nodule": OntologyResult(term="pulmonary nodule", code="RID3875", vocabulary="radlex"),
            "lymph node": OntologyResult(term="lymph node", code="RID34233", vocabulary="radlex"),
            "pericardial effusion": OntologyResult(term="pericardial effusion", code="RID4874", vocabulary="radlex"),
            "pneumothorax": OntologyResult(term="pneumothorax", code="RID4862", vocabulary="radlex"),
            "atelectasis": OntologyResult(term="atelectasis", code="RID4856", vocabulary="radlex"),
            "consolidation": OntologyResult(term="consolidation", code="RID43256", vocabulary="radlex"),
            "ground glass opacity": OntologyResult(term="ground glass opacity", code="RID43259", vocabulary="radlex"),
            "mass": OntologyResult(term="mass", code="RID3874", vocabulary="radlex"),
            "calcification": OntologyResult(term="calcification", code="RID5196", vocabulary="radlex"),
            "fracture": OntologyResult(term="fracture", code="RID4712", vocabulary="radlex"),
        }

    @property
    def supported_vocabularies(self) -> list[str]:
        """Return the list of vocabularies this resolver supports."""
        return ["radlex"]

    def resolve(
        self, term: str, vocabulary: str = "radlex"
    ) -> Optional[OntologyResult]:
        """Resolve a free-text term to a vocabulary code.

        Performs case-insensitive, whitespace-trimmed lookup against the
        local lookup table for the specified vocabulary.

        Parameters
        ----------
        term:
            The free-text term to resolve (e.g. ``"Right Upper Lobe"``).
        vocabulary:
            The target vocabulary.  Currently only ``"radlex"`` is
            supported.

        Returns
        -------
        OntologyResult | None
            The resolved result, or ``None`` if no match is found.
        """
        if vocabulary == "radlex":
            return self._radlex.get(term.lower().strip())
        return None
