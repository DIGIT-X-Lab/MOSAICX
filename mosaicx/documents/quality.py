# mosaicx/documents/quality.py
"""Deterministic quality scorer for OCR output.

Evaluates text quality using four metrics: medical vocabulary hit rate,
character sanity, word structure, and text length. No LLM dependency.
"""

from __future__ import annotations

import re
import string

# Curated medical vocabulary (~200 terms). Covers anatomy, findings,
# modalities, procedures, and common clinical language. Lowercase for
# case-insensitive matching.
MEDICAL_VOCAB: frozenset[str] = frozenset({
    # Anatomy - thorax
    "lung", "lungs", "lobe", "lobes", "bronchus", "bronchi", "trachea",
    "mediastinum", "pleura", "pleural", "diaphragm", "hilum", "hilar",
    "pericardium", "pericardial", "cardiac", "heart", "aorta", "aortic",
    "pulmonary", "thorax", "thoracic", "sternum", "rib", "ribs",
    # Anatomy - abdomen
    "liver", "hepatic", "spleen", "splenic", "kidney", "kidneys", "renal",
    "pancreas", "pancreatic", "gallbladder", "adrenal", "bowel", "colon",
    "intestine", "stomach", "gastric", "duodenum", "appendix", "mesentery",
    "peritoneum", "peritoneal", "retroperitoneal", "bladder", "uterus",
    "ovary", "prostate",
    # Anatomy - neuro
    "brain", "cerebral", "cerebellum", "cerebellar", "ventricle",
    "ventricular", "cortex", "cortical", "hippocampus", "thalamus",
    "brainstem", "pons", "midbrain", "skull", "calvarium", "meninges",
    "meningeal", "dura", "dural", "spinal", "spine", "vertebra",
    "vertebral", "disc", "foramen",
    # Anatomy - MSK
    "femur", "tibia", "fibula", "humerus", "radius", "ulna", "pelvis",
    "acetabulum", "scapula", "clavicle", "patella", "meniscus", "tendon",
    "ligament", "cartilage", "joint", "muscle", "bone", "marrow",
    # Anatomy - vascular
    "artery", "arterial", "vein", "venous", "vessel", "vascular",
    "carotid", "jugular", "subclavian", "iliac", "femoral", "portal",
    # Findings / pathology
    "nodule", "nodular", "mass", "lesion", "tumor", "tumour", "cyst",
    "calcification", "opacity", "opacification", "consolidation",
    "atelectasis", "effusion", "edema", "oedema", "hemorrhage",
    "haemorrhage", "infarct", "infarction", "thrombosis", "thrombus",
    "embolism", "stenosis", "occlusion", "aneurysm", "dissection",
    "fracture", "dislocation", "erosion", "sclerosis", "fibrosis",
    "necrosis", "abscess", "inflammation", "metastasis", "metastatic",
    "lymphadenopathy", "cardiomegaly", "hepatomegaly", "splenomegaly",
    "pneumothorax", "pneumonia", "emphysema", "hernia", "herniation",
    "hydrocephalus", "demyelination", "enhancement", "enhancing",
    # Modalities / procedures
    "ct", "mri", "mra", "pet", "spect", "ultrasound", "radiograph",
    "x-ray", "xray", "fluoroscopy", "mammography", "angiography",
    "echocardiography", "doppler", "scan", "imaging", "contrast",
    "gadolinium", "iodine",
    # Clinical terms
    "patient", "diagnosis", "impression", "findings", "indication",
    "history", "clinical", "chronic", "acute", "bilateral", "unilateral",
    "anterior", "posterior", "lateral", "medial", "superior", "inferior",
    "proximal", "distal", "diffuse", "focal", "mild", "moderate",
    "severe", "benign", "malignant", "normal", "abnormal", "stable",
    "unchanged", "interval", "follow-up", "followup", "prior",
    "comparison", "technique", "protocol", "axial", "coronal", "sagittal",
    # Measurements / descriptors
    "mm", "cm", "diameter", "size", "volume", "density", "signal",
    "intensity", "attenuation", "hyperintense", "hypointense",
    "hyperdense", "hypodense", "heterogeneous", "homogeneous",
    "circumscribed", "irregular", "spiculated", "ground-glass",
})

# Characters considered "sane" in medical text
_SANE_CHARS = set(string.ascii_letters + string.digits + string.whitespace + ".,;:!?'-/()+=%\u00b0")


def medical_vocab_score(text: str) -> float:
    """Score 0-1 based on fraction of words matching medical vocabulary."""
    if not text.strip():
        return 0.0
    words = re.findall(r"[a-zA-Z][\w'-]*", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in MEDICAL_VOCAB)
    # Normalize: a good medical report has ~15-30% medical terms
    raw_ratio = hits / len(words)
    # Scale so that 0.15 ratio -> 0.7 score, 0.30 -> 1.0
    return min(1.0, raw_ratio / 0.30)


def char_sanity_score(text: str) -> float:
    """Score 0-1 based on fraction of characters that are 'normal'.

    Also penalises digits embedded inside alphabetic tokens (e.g. ``cl3@r``,
    ``p@t13nt``), which is a strong signal of OCR corruption.
    """
    if not text:
        return 0.0
    sane = sum(1 for c in text if c in _SANE_CHARS)
    base = sane / len(text)
    # Penalise mixed alpha-digit tokens (leet-speak style OCR errors)
    tokens = text.split()
    if tokens:
        mixed = sum(
            1 for t in tokens
            if re.search(r"[a-zA-Z]", t) and re.search(r"\d", t)
        )
        mixed_ratio = mixed / len(tokens)
        # Heavy penalty: each mixed token drags the score down
        base *= max(0.0, 1.0 - mixed_ratio)
    return base


def word_structure_score(text: str) -> float:
    """Score 0-1 based on word length distribution and whitespace ratio.

    Also penalises text where most tokens lack alphabetic characters,
    which indicates non-text content (pure symbols / garbled OCR).
    """
    if not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    # Ideal average word length for English medical text: 4-8 chars
    if 3.0 <= avg_len <= 10.0:
        len_score = 1.0
    elif avg_len < 2.0 or avg_len > 15.0:
        len_score = 0.2
    else:
        len_score = 0.6
    # Whitespace ratio: should be ~15-25% of total chars
    ws_ratio = sum(1 for c in text if c in " \t\n") / len(text) if text else 0
    if 0.10 <= ws_ratio <= 0.35:
        ws_score = 1.0
    elif ws_ratio < 0.05 or ws_ratio > 0.50:
        ws_score = 0.2
    else:
        ws_score = 0.6
    # Alpha-content: fraction of tokens that contain at least one letter
    alpha_tokens = sum(1 for w in words if re.search(r"[a-zA-Z]", w))
    alpha_ratio = alpha_tokens / len(words)
    return (len_score + ws_score) / 2.0 * alpha_ratio


def text_length_score(text: str, expected_pages: int = 1) -> float:
    """Score 0-1 based on text length relative to page count.

    A typical medical report page has ~300-800 words.
    """
    if not text.strip():
        return 0.0
    word_count = len(text.split())
    expected_words = expected_pages * 400  # midpoint estimate
    if expected_words == 0:
        return 0.5
    ratio = word_count / expected_words
    # 0.5-2.0x expected is good, outside that degrades
    if 0.3 <= ratio <= 2.5:
        return 1.0
    elif ratio < 0.1:
        return 0.1
    else:
        return 0.5


class QualityScorer:
    """Deterministic scorer for OCR output quality.

    Combines medical vocabulary, character sanity, word structure,
    and text length into a single 0-1 score.
    """

    WEIGHTS = {
        "medical_vocab": 0.35,
        "char_sanity": 0.25,
        "word_structure": 0.20,
        "text_length": 0.20,
    }

    def score(self, text: str, expected_pages: int = 1) -> float:
        """Return overall quality score 0-1."""
        if not text.strip():
            return 0.0
        return (
            self.WEIGHTS["medical_vocab"] * medical_vocab_score(text)
            + self.WEIGHTS["char_sanity"] * char_sanity_score(text)
            + self.WEIGHTS["word_structure"] * word_structure_score(text)
            + self.WEIGHTS["text_length"] * text_length_score(text, expected_pages)
        )

    def score_detailed(self, text: str, expected_pages: int = 1) -> dict[str, float]:
        """Return per-metric scores plus overall."""
        detail = {
            "medical_vocab": medical_vocab_score(text),
            "char_sanity": char_sanity_score(text),
            "word_structure": word_structure_score(text),
            "text_length": text_length_score(text, expected_pages),
        }
        detail["overall"] = sum(
            self.WEIGHTS[k] * v for k, v in detail.items() if k != "overall"
        )
        return detail
