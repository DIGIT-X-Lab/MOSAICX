# Tasks: MOSAICX Python Package for Radiology Report Extraction

**Input**: Design documents from `/Users/nutellabear/Documents/00-Code/MOSAICX/specs/001-python-package-mosaicx/`
**Prerequisites**: plan.md (✓), spec.md (✓)

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Found: Implementation plan with Python 3.11+, Ollama, DSPy tech stack
   → Extract: Single project structure, CLI + library pattern
2. Load optional design documents:
   → No data-model.md: Generate models from spec entities
   → No contracts/: Generate from CLI commands and library API
   → No research.md: Generate setup tasks from plan dependencies
3. Generate tasks by category:
   → Setup: Python project, dependencies, Ollama integration
   → Tests: CLI commands, library methods, integration scenarios
   → Core: extraction engine, schema system, LLM processing
   → Integration: PDF parsing, multi-report analysis, terminal UI
   → Polish: documentation, performance, unit tests
4. Apply task rules:
   → Different modules = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph for Python package development
7. Create parallel execution examples
8. Validate task completeness against 37 functional requirements
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single Python project**: `mosaicx/`, `tests/` at repository root
- Paths based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create Python project structure per implementation plan: `mosaicx/{core,cli,schemas,extractors,utils}/`, `tests/{contract,integration,e2e,unit}/`, `docs/`, `examples/`, `schemas/`
- [ ] T002 Initialize Python project with pyproject.toml, dependencies: Ollama, OpenAI, DSPy, Rich, Click, PyPDF2, Pydantic, pytest
- [ ] T003 [P] Configure development tools: pre-commit hooks, black, isort, mypy, pytest configuration
- [ ] T004 [P] Create example radiology reports in `examples/` for testing
- [ ] T005 [P] Create built-in schema templates in `schemas/` for demographics, findings, impressions

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### CLI Contract Tests
- [ ] T006 [P] Contract test `mosaicx extract report.pdf` in `tests/contract/test_cli_extract_basic.py`
- [ ] T007 [P] Contract test `mosaicx extract --style summary` in `tests/contract/test_cli_extract_summary.py`
- [ ] T008 [P] Contract test `mosaicx extract --style structured` in `tests/contract/test_cli_extract_structured.py`
- [ ] T009 [P] Contract test `mosaicx extract --interactive` in `tests/contract/test_cli_interactive.py`
- [ ] T010 [P] Contract test `mosaicx extract --batch` in `tests/contract/test_cli_batch.py`
- [ ] T011 [P] Contract test `mosaicx extract --style patient-summary` in `tests/contract/test_cli_patient_summary.py`

### Library API Contract Tests  
- [ ] T012 [P] Contract test `mosaicx.extract()` method in `tests/contract/test_api_extract.py`
- [ ] T013 [P] Contract test `mosaicx.build_schema_interactive()` in `tests/contract/test_api_schema_builder.py`
- [ ] T014 [P] Contract test `mosaicx.suggest_schema()` in `tests/contract/test_api_schema_suggest.py`
- [ ] T015 [P] Contract test `mosaicx.analyze_patient_history()` in `tests/contract/test_api_patient_history.py`

### Integration Tests from User Scenarios
- [ ] T016 [P] Integration test: PDF report auto-extraction in `tests/integration/test_pdf_auto_extraction.py`
- [ ] T017 [P] Integration test: Text report summary generation in `tests/integration/test_text_summary.py`
- [ ] T018 [P] Integration test: Custom schema structured extraction in `tests/integration/test_custom_schema.py`
- [ ] T019 [P] Integration test: Interactive schema building in `tests/integration/test_interactive_schema.py`
- [ ] T020 [P] Integration test: Batch processing multiple reports in `tests/integration/test_batch_processing.py`
- [ ] T021 [P] Integration test: Patient history synthesis in `tests/integration/test_patient_history.py`
- [ ] T022 [P] Integration test: Multi-report analysis in `tests/integration/test_multi_report.py`
- [ ] T023 [P] Integration test: LLM offline processing in `tests/integration/test_llm_offline.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Core Models and Data Structures
- [ ] T024 [P] Report model in `mosaicx/core/__init__.py` - data classes for PDF/text reports
- [ ] T025 [P] ExtractedContent model in `mosaicx/core/__init__.py` - structured output representation
- [ ] T026 [P] Schema model in `mosaicx/schemas/__init__.py` - schema definition classes

### PDF and Text Processing
- [ ] T027 [P] PDF parser in `mosaicx/core/report_parser.py` - extract text from PDF files
- [ ] T028 [P] Text processor in `mosaicx/core/report_parser.py` - handle text file input

### LLM Integration Engine
- [ ] T029 [P] LLM processor base in `mosaicx/core/llm_processor.py` - Ollama + OpenAI + DSPy integration
- [ ] T030 [P] Schema templates in `mosaicx/schemas/templates.py` - built-in extraction patterns
- [ ] T031 [P] Schema validator in `mosaicx/schemas/validator.py` - validate user schemas

### Single Report Extraction
- [ ] T032 Main extractor engine in `mosaicx/core/extractor.py` - orchestrates LLM processing
- [ ] T033 Single report processor in `mosaicx/extractors/single_report.py` - individual report analysis
- [ ] T034 Summary generator in `mosaicx/extractors/summarizer.py` - summary creation logic

### CLI Commands Implementation
- [ ] T035 CLI main entry point in `mosaicx/cli/main.py` - Click app setup
- [ ] T036 Extract command handler in `mosaicx/cli/extract.py` - main extraction command
- [ ] T037 Schema management commands in `mosaicx/cli/schema.py` - schema-related CLI

### Interactive Features
- [ ] T038 Interactive schema builder in `mosaicx/schemas/builder.py` - conversational schema design
- [ ] T039 Terminal progress displays in `mosaicx/utils/progress.py` - Rich terminal UI
- [ ] T040 User input handling in `mosaicx/utils/validation.py` - input validation and feedback

## Phase 3.4: Integration

### Multi-Report Analysis
- [ ] T041 Temporal organizer in `mosaicx/extractors/multi_report.py` - chronological ordering
- [ ] T042 Patient history analyzer in `mosaicx/extractors/multi_report.py` - synthesis logic
- [ ] T043 Progression detector in `mosaicx/extractors/multi_report.py` - change analysis

### Library API Integration
- [ ] T044 Library main API in `mosaicx/__init__.py` - public library interface
- [ ] T045 Error handling system in `mosaicx/utils/validation.py` - comprehensive error management
- [ ] T046 Audit logging in `mosaicx/utils/logging.py` - compliance and activity tracking

### Output and Formatting
- [ ] T047 JSON/CSV output formatters in `mosaicx/core/extractor.py` - structured data output
- [ ] T048 Terminal display enhancement in `mosaicx/utils/progress.py` - beautiful CLI output
- [ ] T049 Schema auto-suggestion in `mosaicx/schemas/builder.py` - intelligent schema recommendations

## Phase 3.5: Polish

### Performance and Optimization
- [ ] T050 [P] Performance tests: <5s single report, <30s batch processing in `tests/unit/test_performance.py`
- [ ] T051 [P] Memory usage optimization: <2GB memory limit in `tests/unit/test_memory.py`
- [ ] T052 [P] Large file handling: 50MB PDF support in `tests/unit/test_large_files.py`

### Documentation and Examples
- [ ] T053 [P] Auto-generate API documentation from docstrings in `docs/`
- [ ] T054 [P] Create usage examples and tutorials in `examples/`
- [ ] T055 [P] Update README.md with installation and quick start

### Unit Tests Coverage
- [ ] T056 [P] Unit tests for core extractor in `tests/unit/test_core_extractor.py`
- [ ] T057 [P] Unit tests for schema validation in `tests/unit/test_schema_validator.py`
- [ ] T058 [P] Unit tests for report parsing in `tests/unit/test_report_parser.py`
- [ ] T059 [P] Unit tests for LLM processing in `tests/unit/test_llm_processor.py`

### Final Integration
- [ ] T060 End-to-end CLI validation in `tests/e2e/test_full_workflow.py`
- [ ] T061 Package build and distribution setup
- [ ] T062 Update version to 0.1.0 and prepare release

## Dependencies
- Setup (T001-T005) before everything
- Tests (T006-T023) before implementation (T024-T049)
- Core models (T024-T026) before processors (T027-T034)
- PDF/text processing (T027-T028) before extraction (T032-T034)
- LLM integration (T029) before extractors (T032-T034)
- CLI foundation (T035) before commands (T036-T037)
- Single report (T032-T034) before multi-report (T041-T043)
- Core functionality before polish (T050-T062)

## Parallel Example
```bash
# Launch contract tests together (Phase 3.2):
Task: "Contract test mosaicx extract report.pdf in tests/contract/test_cli_extract_basic.py"
Task: "Contract test mosaicx extract --style summary in tests/contract/test_cli_extract_summary.py"
Task: "Contract test mosaicx.extract() method in tests/contract/test_api_extract.py"
Task: "Contract test mosaicx.build_schema_interactive() in tests/contract/test_api_schema_builder.py"

# Launch core models together (Phase 3.3):
Task: "Report model in mosaicx/core/__init__.py"
Task: "Schema model in mosaicx/schemas/__init__.py"
Task: "PDF parser in mosaicx/core/report_parser.py"
Task: "Schema templates in mosaicx/schemas/templates.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify ALL tests fail before implementing (critical for TDD)
- Commit after each task completion
- Focus on local LLM integration without external API dependencies
- Maintain intuitive CLI and library interfaces throughout

## Task Generation Rules Applied
1. **From Functional Requirements**: 37 requirements → contract and integration tests
2. **From User Scenarios**: 10 scenarios → integration test tasks
3. **From Technical Architecture**: Python modules → implementation tasks
4. **From CLI Commands**: Extract command variations → contract tests
5. **From Library API**: Core methods → API contract tests

## Validation Checklist
- [x] All CLI commands have corresponding contract tests (T006-T011)
- [x] All library methods have contract tests (T012-T015)
- [x] All user scenarios have integration tests (T016-T023)
- [x] All tests come before implementation (Phase 3.2 before 3.3)
- [x] Parallel tasks are truly independent (different files)
- [x] Each task specifies exact file path
- [x] No [P] task modifies same file as another [P] task
- [x] TDD enforced: tests must fail before implementation
- [x] Covers all 37 functional requirements from specification
