# Sample Medical PDFs for MOSAICX Demo

This directory contains sample medical documents for testing MOSAICX capabilities.

## Files

### 1. sample_echo_report.pdf
**Type**: Echocardiography Report  
**Content**: Typical cardiac ultrasound report with LVEF measurements, valve assessments, and clinical impressions  
**Best Schema**: Use "Echocardiography report with patient demographics, LVEF, valve grades, impression"

### 2. sample_cbc_report.pdf  
**Type**: Complete Blood Count Lab Report  
**Content**: Standard CBC with differential counts, reference ranges, and patient demographics  
**Best Schema**: Use "Complete blood count with patient ID, test date, hemoglobin, hematocrit, WBC count, differential counts, and reference ranges"

### 3. sample_radiology_series/
**Type**: Series of CT reports for longitudinal analysis  
**Content**: Three sequential CT chest reports showing disease progression  
**Use Case**: Perfect for the Report Summarizer to demonstrate timeline generation

## Usage Instructions

1. **Schema Generation**: Start by creating schemas with the suggested descriptions above
2. **PDF Extraction**: Upload the PDFs and select the appropriate schema  
3. **Report Summarization**: Use the radiology series to see timeline-based summaries

## Data Privacy

These are synthetic/anonymized medical reports created for demonstration purposes only. No real patient data is included.