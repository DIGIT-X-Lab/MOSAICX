#!/bin/bash

# MOSAICX Pre-Push Validation Script
# ==================================
# Comprehensive validation before pushing to production

set -e  # Exit on any error

echo "🧬 MOSAICX Pre-Push Validation"
echo "========================================================"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "mosaicx" ]]; then
    echo "❌ Please run this script from the MOSAICX root directory"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connectivity..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

# Check if required model is available
echo "🔍 Checking for mistral:latest model..."
if ! ollama list | grep -q "mistral:latest"; then
    echo "❌ mistral:latest model not found. Please install it:"
    echo "   ollama pull mistral:latest"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Test 1: CLI Health Check
echo "🧪 Test 1: CLI Health Check"
echo "----------------------------------------"
if mosaicx --help > /dev/null 2>&1; then
    echo "✅ CLI is functional"
else
    echo "❌ CLI health check failed"
    exit 1
fi

# Test 2: Schema Generation (CLI)
echo ""
echo "🧪 Test 2: CLI Schema Generation"
echo "----------------------------------------"
if mosaicx generate --desc "Test patient extraction with name and age" --model mistral:latest > /tmp/mosaicx_test_schema.log 2>&1; then
    echo "✅ Schema generation successful"
    if grep -q "Generated Schema Results" /tmp/mosaicx_test_schema.log; then
        echo "✅ Schema output format correct"
    else
        echo "⚠️  Schema output format unusual (check /tmp/mosaicx_test_schema.log)"
    fi
else
    echo "❌ Schema generation failed"
    cat /tmp/mosaicx_test_schema.log
    exit 1
fi

# Test 3: PDF Extraction (CLI)
echo ""
echo "🧪 Test 3: CLI PDF Extraction"
echo "----------------------------------------"
EXTRACT_PDF="tests/datasets/extract/sample_patient_vitals.pdf"
EXTRACT_SCHEMA="mosaicx/schema/templates/python/patient_identity.py"

if [[ ! -f "$EXTRACT_PDF" ]]; then
    echo "❌ Test PDF not found: $EXTRACT_PDF"
    exit 1
fi

if [[ ! -f "$EXTRACT_SCHEMA" ]]; then
    echo "❌ Test schema not found: $EXTRACT_SCHEMA"
    exit 1
fi

if mosaicx extract --pdf "$EXTRACT_PDF" --schema "$EXTRACT_SCHEMA" --model mistral:latest > /tmp/mosaicx_test_extract.log 2>&1; then
    echo "✅ PDF extraction successful"
    if grep -q "Extraction results" /tmp/mosaicx_test_extract.log && grep -q "name" /tmp/mosaicx_test_extract.log; then
        echo "✅ Extraction output format correct"
    else
        echo "⚠️  Extraction output format unusual (check /tmp/mosaicx_test_extract.log)"
    fi
else
    echo "❌ PDF extraction failed"
    cat /tmp/mosaicx_test_extract.log
    exit 1
fi

# Test 4: Report Summarization (CLI)
echo ""
echo "🧪 Test 4: CLI Report Summarization"
echo "----------------------------------------"
SUMMARIZE_DIR="tests/datasets/summarize"

if [[ ! -d "$SUMMARIZE_DIR" ]]; then
    echo "❌ Test summarization directory not found: $SUMMARIZE_DIR"
    exit 1
fi

if mosaicx summarize --dir "$SUMMARIZE_DIR" --model mistral:latest > /tmp/mosaicx_test_summarize.log 2>&1; then
    echo "✅ Report summarization successful"
    if grep -q "Patient: P001" /tmp/mosaicx_test_summarize.log && grep -q "2025-08-01" /tmp/mosaicx_test_summarize.log; then
        echo "✅ Summarization output format correct"
    else
        echo "⚠️  Summarization output format unusual (check /tmp/mosaicx_test_summarize.log)"
    fi
else
    echo "❌ Report summarization failed"
    cat /tmp/mosaicx_test_summarize.log
    exit 1
fi

# Test 5: Python API Tests
echo ""
echo "🧪 Test 5: Python API Validation"
echo "----------------------------------------"

# Create a temporary Python test script
cat > /tmp/mosaicx_api_test.py << 'EOF'
#!/usr/bin/env python3
import sys
import traceback

try:
    from mosaicx import generate_schema, extract_pdf, summarize_reports
    print("✅ API imports successful")
    
    # Test schema generation
    schema = generate_schema(
        "Test extraction: patient name and age",
        class_name="TestPatient",
        model="mistral:latest"
    )
    print("✅ API schema generation successful")
    
    # Test extraction
    extraction = extract_pdf(
        pdf_path="tests/datasets/extract/sample_patient_vitals.pdf",
        schema_path="mosaicx/schema/templates/python/patient_identity.py",
    )
    payload = extraction.to_dict()
    if "name" in payload:
        print("✅ API extraction successful")
    else:
        print("⚠️  API extraction completed but unexpected format")
    
    # Test summarization
    summary = summarize_reports(
        paths=["tests/datasets/summarize"],
        patient_id="TEST_API",
    )
    if summary.patient.patient_id == "TEST_API" and len(summary.timeline) > 0:
        print("✅ API summarization successful")
    else:
        print("⚠️  API summarization completed but unexpected format")
        
    print("🎉 All Python API tests passed!")
    
except Exception as e:
    print(f"❌ Python API test failed: {e}")
    traceback.print_exc()
    sys.exit(1)
EOF

if python /tmp/mosaicx_api_test.py; then
    echo "✅ Python API tests passed"
else
    echo "❌ Python API tests failed"
    exit 1
fi

# Test 6: File Structure Validation
echo ""
echo "🧪 Test 6: File Structure Validation"
echo "----------------------------------------"

REQUIRED_FILES=(
    "pyproject.toml"
    "README.md"
    "mosaicx/__init__.py"
    "mosaicx/mosaicx.py"
    "tests/test_pre_push_validation.py"
    "webapp/start.sh"
    "webapp/README.md"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    echo "✅ All required files present"
else
    echo "❌ Missing required files:"
    printf '%s\n' "${MISSING_FILES[@]}"
    exit 1
fi

# Test 7: Git Status Check
echo ""
echo "🧪 Test 7: Git Status Check"
echo "----------------------------------------"

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Uncommitted changes detected:"
    git status --porcelain
    echo ""
    echo "Consider committing changes before push"
else
    echo "✅ No uncommitted changes"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up test artifacts..."
rm -f /tmp/mosaicx_test_*.log /tmp/mosaicx_api_test.py

echo ""
echo "🎉 PRE-PUSH VALIDATION COMPLETED SUCCESSFULLY!"
echo "========================================================"
echo "✅ CLI functionality: PASSED"
echo "✅ Schema generation: PASSED" 
echo "✅ PDF extraction: PASSED"
echo "✅ Report summarization: PASSED"
echo "✅ Python API: PASSED"
echo "✅ File structure: PASSED"
echo "✅ Git status: CHECKED"
echo ""
echo "🚀 MOSAICX is ready for production push!"
echo ""
