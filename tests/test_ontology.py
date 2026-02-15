"""Tests for the ontology resolver."""
import pytest

class TestOntologyResult:
    def test_construction(self):
        from mosaicx.schemas.ontology import OntologyResult
        r = OntologyResult(term="right upper lobe", code="RID1303", vocabulary="radlex", confidence=0.95)
        assert r.code == "RID1303"
        assert r.vocabulary == "radlex"

class TestOntologyResolver:
    def test_exact_match(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        result = resolver.resolve("right upper lobe", vocabulary="radlex")
        assert result is not None
        assert result.code is not None

    def test_unknown_term(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        result = resolver.resolve("xyzzy_nonexistent_abc123", vocabulary="radlex")
        assert result is None

    def test_supported_vocabularies(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        vocabs = resolver.supported_vocabularies
        assert "radlex" in vocabs

    def test_case_insensitive(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        r1 = resolver.resolve("Right Upper Lobe", vocabulary="radlex")
        r2 = resolver.resolve("right upper lobe", vocabulary="radlex")
        assert r1 is not None and r2 is not None
        assert r1.code == r2.code
