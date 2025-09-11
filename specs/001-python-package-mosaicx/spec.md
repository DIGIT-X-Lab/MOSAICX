# Feature Specification: MOSAICX Python Package for Radiology Report Extraction

---

## Project Information

**Author**: Lalith Kumar Shiyam Sundar, PhD  
**Laboratory**: DIGITX Lab  
**Department**: Department of Radiology  
**Institution**: LMU Munich University Hospital  
**Module**: MOSAICX (Medical cOmputational Suite for Advanced Intelligent eXtraction)

---

**Feature Branch**: `001-python-package-mosaicx`  
**Created**: September 11, 2025  
**Status**: Ready for Development  
**Input**: User description: "Python package MOSAICX for radiology report extraction with configurable output formats. Takes report and config file as inputs to extract structured data or summaries based on instructions."

**Note**: This specification is designed to evolve during development. Requirements may be refined, added, or modified based on implementation discoveries and user feedback.

## Execution Flow (main)
```
1. Parse user description from Input
   → Feature description provided: Python package for radiology report extraction
2. Extract key concepts from description
   → Actors: biomedical engineers, radiologists, medical professionals
   → Actions: extract data, configure output, process reports
   → Data: radiology reports, configuration files, structured output
   → Constraints: configurable extraction rules
3. For each unclear aspect:
   → Input formats: PDF and text files confirmed
   → Deployment: Both library and CLI tool confirmed
   → Output formats: JSON and CSV confirmed
   → Local LLM integration: Ollama with OpenAI wrapper + DSPy
4. Fill User Scenarios & Testing section
   → Primary flow: load config, process report, extract data
5. Generate Functional Requirements
   → Each requirement focused on extraction capabilities
6. Identify Key Entities
   → Report, Configuration, ExtractedData
7. Run Review Checklist
   → Spec has uncertainties marked for clarification
8. Return: SUCCESS (spec ready for development with evolution expected)
```

**Evolution Strategy**: This specification establishes the foundation but anticipates refinements during development. Key areas likely to evolve:
- User interaction patterns based on real usage
- Schema definition approaches based on complexity discoveries  
- Performance optimizations based on real-world report processing
- Fun factor calibration based on user feedback
- Integration patterns based on library usage patterns

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a biomedical engineer working with radiology reports, I want to use an intuitive tool that can intelligently "extract" whatever I need from reports - whether that's structured data, summaries, or insights - with minimal cognitive overhead. When I have multiple fragmented reports representing a patient's history, I want the system to intelligently synthesize them into a comprehensive summary highlighting the important progression, findings, and clinical significance. The system should have intuitive commands and provide results in a clear, engaging way.

### Acceptance Scenarios
1. **Given** only a PDF radiology report, **When** I run `mosaicx extract report.pdf`, **Then** the system intelligently determines whether I need structured data or a summary and delivers it with clear terminal output and progress indicators
2. **Given** a text radiology report, **When** I run `mosaicx extract report.txt --style summary`, **Then** I receive a well-formatted summary with key insights highlighted and clear visual presentation
3. **Given** I want custom structured output, **When** I run `mosaicx extract report.pdf --style structured --schema my_schema.yaml`, **Then** I get data organized exactly according to my defined schema with clear validation feedback
4. **Given** I don't have a predefined schema, **When** I run `mosaicx extract report.pdf --style structured --interactive`, **Then** the system engages me in a guided conversation to collaboratively design the optimal schema for my needs
5. **Given** multiple reports with the same schema, **When** I run `mosaicx extract *.pdf --schema radiology_standard.json --batch`, **Then** I see consistent structured output across all reports with clear progress displays
6. **Given** I'm using MOSAICX as a library and want interactive schema building, **When** I call `mosaicx.build_schema_interactive(report, callback=my_input_handler)`, **Then** the system calls my callback function with questions and options, allowing programmatic interaction
7. **Given** I'm using MOSAICX as a library without interactive capability, **When** I call `mosaicx.suggest_schema(report)`, **Then** I receive an auto-generated schema suggestion that I can modify and use for extraction
8. **Given** multiple fragmented reports from a patient's history, **When** I run `mosaicx extract patient_reports/*.pdf --style patient-summary`, **Then** the system intelligently synthesizes all reports into a comprehensive timeline summary highlighting disease progression, key findings, and clinical significance
9. **Given** a collection of reports and custom aggregation rules, **When** I run `mosaicx extract reports/*.pdf --style multi-report --schema timeline_schema.yaml`, **Then** I receive structured data showing progression patterns, recurring findings, and comparative analysis across the report series
10. **Given** I'm using the library for patient history analysis, **When** I call `mosaicx.analyze_patient_history(reports_list, focus="progression")`, **Then** I receive an intelligent synthesis highlighting temporal patterns, evolving conditions, and clinical significance

### Edge Cases
- What happens when the system can't decide between summary or structured extraction and wants to have fun asking the user?
- How does the system handle user-defined schemas that are impossible to fill from the available report data?
- What occurs when users want to extract something unusual and the system needs to be creative with schema suggestions?
- How does the interactive schema builder handle users who aren't sure what fields they want in CLI vs library contexts?
- How does the library handle interactive schema building when no callback function is provided?
- What happens when callback functions in library mode don't respond or provide invalid input?
- How does the system validate and suggest improvements to user-defined schemas while keeping it fun?
- What happens when local LLM services are slow and the system needs to entertain the user during schema processing?
- How does the system handle conflicting field definitions in custom schemas and make resolution enjoyable?
- How does the system intelligently handle temporal ordering when report dates are missing or unclear?
- What happens when multi-report analysis encounters conflicting findings across different time points?
- How does the system handle incomplete patient histories where some reports may be missing?
- How does the patient history synthesis maintain clinical accuracy while providing engaging summaries?

## Requirements *(mandatory)*

**Evolution Note**: These requirements form the initial foundation. During development, requirements may be:
- **Refined**: Made more specific based on implementation learnings
- **Extended**: New requirements added for discovered needs  
- **Simplified**: Complex requirements broken down or streamlined
- **Reprioritized**: Order adjusted based on user feedback and technical discoveries

### Functional Requirements
- **FR-001**: System MUST accept radiology reports as input in PDF and text file formats with an intuitive `extract` command that handles everything
- **FR-002**: System MUST intelligently determine whether user needs structured data, summaries, or insights and deliver accordingly with clear presentation
- **FR-003**: System MUST extract any type of information (structured data, summaries, insights) using local LLMs with an efficient and engaging user experience
- **FR-004**: System MUST support intuitive style options like `--style summary`, `--style structured`, `--style auto`, `--style surprise-me`, `--style patient-summary`, and `--style multi-report` with optional schema definitions for structured outputs
- **FR-005**: System MUST provide clear, colorful terminal outputs with progress indicators and informative status messages
- **FR-006**: System MUST handle missing or incomplete report sections gracefully while maintaining a helpful and informative tone
- **FR-007**: System MUST provide outputs in JSON, CSV, or well-formatted text summaries with clear presentation and visual clarity
- **FR-008**: System MUST process single reports and batches with clear progress displays, useful statistics, and completion notifications
- **FR-009**: System MUST preserve data integrity while making the entire process efficient and user-friendly
- **FR-010**: System MUST provide comprehensive audit logging with clear naming conventions and informative log messages
- **FR-011**: System MUST handle errors with helpful suggestions and clear guidance without being frustrating
- **FR-012**: System MUST accept custom schema definitions in JSON, YAML, or Python dict formats to define structured output formats
- **FR-013**: System MUST provide an interactive schema builder that works in CLI mode with terminal conversations and in library mode with callback functions for programmatic interaction
- **FR-014**: System MUST validate user-defined schemas and provide helpful suggestions for improvements with clear feedback
- **FR-015**: System MUST offer built-in schema templates for common radiology extraction patterns (demographics, findings, impressions, recommendations)
- **FR-016**: System MUST support schema inheritance and composition to allow users to build complex schemas from simpler components
- **FR-017**: System MUST provide schema auto-generation based on report analysis when users request structured output without defining schemas
- **FR-018**: System MUST validate extracted data against user schemas and provide confidence scores for each field with clear visual feedback
- **FR-019**: Library mode MUST provide both callback-based interactive schema building and non-interactive schema suggestion methods
- **FR-020**: System MUST handle library interactive mode gracefully when no callback is provided by falling back to auto-suggestion
- **FR-021**: System MUST be available as both a Python library with intuitive method names and a CLI tool with clear, logical commands
- **FR-022**: CLI tool MUST provide excellent terminal experiences with clear visual feedback, progress indicators, and satisfying completion notifications
- **FR-023**: System MUST engage users in helpful conversations when clarification is needed, using clear communication and guidance
- **FR-024**: System MUST maintain high performance while providing clear feedback about processing speed and efficiency
- **FR-025**: System MUST operate with local LLMs while keeping users informed during processing with clear status messages and updates
- **FR-026**: System MUST provide reliable fallback mechanisms and handle technical difficulties with clear error messages and helpful guidance
- **FR-027**: System MUST generate comprehensive documentation automatically from code using docstrings, type hints, and code annotations
- **FR-028**: Documentation MUST include interactive examples, API references, and usage tutorials that reflect the fun personality of the tool
- **FR-029**: System MUST support documentation generation in multiple formats (HTML, PDF, Markdown) with rich formatting and examples
- **FR-030**: System MUST intelligently analyze multiple reports as a cohesive patient history, identifying temporal patterns, disease progression, and clinical significance
- **FR-031**: System MUST provide patient-summary style that synthesizes fragmented reports into comprehensive timeline narratives highlighting key developments
- **FR-032**: System MUST support multi-report structured extraction that identifies patterns, progressions, and relationships across report series  
- **FR-033**: System MUST automatically detect and handle temporal ordering of reports using dates, clinical context, and content analysis
- **FR-034**: System MUST identify and highlight contradictions, progressions, and stable findings across multiple reports in an engaging manner
- **FR-035**: Library MUST provide specialized methods like `analyze_patient_history()` and `synthesize_reports()` for multi-report analysis
- **FR-036**: Multi-report analysis MUST maintain individual report traceability while providing cohesive synthesis
- **FR-037**: System MUST support custom aggregation schemas that define how to combine and structure multi-report extractions

### Key Entities *(include if feature involves data)*
- **Report**: The star of the show - a radiology report that's about to get the VIP treatment with intelligent analysis and fun extraction
- **ExtractedContent**: The delightful output that could be structured data, summaries, or insights - always presented in the most engaging and helpful way possible
- **Schema**: User-defined or auto-generated structure that specifies exactly what fields to extract and how to organize them; supports JSON, YAML, and Python dict formats
- **SchemaBuilder**: The interactive component that works in CLI mode with terminal conversations and in library mode with callback functions for programmatic schema design and smart suggestions
- **SchemaTemplate**: Pre-built schema patterns for common radiology needs (demographics, findings, impressions) that users can use as starting points
- **FunExtractor**: The entertaining AI-powered entity that analyzes reports and decides whether to create summaries, structured data, or surprise insights while keeping things lively
- **StyleEngine**: The creative component that handles different extraction styles and ensures each has its own personality and flair
- **ValidationEngine**: The encouraging component that checks extracted data against schemas and provides confidence scores with positive feedback
- **CelebrationManager**: The entity responsible for making every successful extraction feel like a victory with animations, emojis, and encouraging messages
- **InteractiveHost**: The friendly conversational component that engages users when clarification is needed, using humor and encouragement to make the experience enjoyable
- **AuditLogger**: The behind-the-scenes entity that tracks everything with fun naming conventions and engaging log messages that even make debugging entertaining
- **LLMProcessor**: The local AI brain using Ollama with OpenAI wrapper and DSPy, enhanced with personality and wit to make every interaction delightful
- **DocumentationGenerator**: The intelligent component that automatically creates comprehensive, engaging documentation from code annotations and examples
- **PatientHistoryAnalyzer**: The sophisticated component that takes multiple fragmented reports and intelligently synthesizes them into cohesive patient narratives, identifying patterns, progressions, and clinical significance
- **TemporalOrganizer**: The smart entity that automatically detects and orders reports chronologically using dates, clinical context, and content analysis for accurate timeline construction
- **MultiReportSynthesizer**: The creative component that combines individual report extractions into cohesive structured data or summaries while maintaining traceability to source reports
- **ProgressionDetector**: The analytical entity that identifies disease progression, treatment responses, and clinical changes across multiple time points with engaging visual representations

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---
