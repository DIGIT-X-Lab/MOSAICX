# Sample Schema Templates for MOSAICX Demo

## 🔬 Common Medical Schema Descriptions

Copy and paste these into the Schema Generator for quick testing:

### Cardiovascular

**Echocardiography Report**
```
Echocardiography report with patient demographics, exam date, LVEF percentage, mitral valve grade, aortic valve grade, tricuspid valve grade, pulmonary valve grade, wall motion abnormalities, and clinical impression
```

**ECG Report**  
```
Electrocardiogram report with patient ID, recording date, heart rate, rhythm interpretation, PR interval, QRS duration, QT interval, axis deviation, and cardiologist impression
```

### Laboratory

**Complete Blood Count**
```
Complete blood count with patient ID, test date, hemoglobin, hematocrit, WBC count, neutrophils percentage, lymphocytes percentage, monocytes percentage, eosinophils percentage, basophils percentage, and reference ranges
```

**Basic Metabolic Panel**
```
Basic metabolic panel with patient identifier, collection date, sodium, potassium, chloride, CO2, BUN, creatinine, glucose, eGFR, and critical value flags
```

### Radiology

**CT Chest Report**
```
CT chest report with patient demographics, study date, indication, technique, contrast usage, lung findings, mediastinal findings, pleural findings, chest wall findings, and radiologist impression
```

**MRI Brain Report**
```
MRI brain report with patient ID, scan date, sequences performed, brain parenchyma findings, ventricular system, extra-axial spaces, vascular findings, and neuroradiologist conclusion
```

### Pathology

**Surgical Pathology**
```
Surgical pathology report with patient information, specimen type, gross description, microscopic findings, tumor staging, margins status, immunohistochemistry results, and pathologist diagnosis
```

**Cytology Report**
```
Cytology report with patient demographics, specimen source, adequacy assessment, cellular findings, background findings, and cytopathologist interpretation
```

## 📊 Usage Tips

1. **Start Simple**: Begin with basic schemas and add complexity
2. **Medical Terminology**: Use standard medical terms for better results  
3. **Validation Constraints**: The AI will automatically add appropriate field validations
4. **Field Descriptions**: Each generated field will have descriptive help text

## 🎯 Model Recommendations

- **Complex schemas**: Use `gpt-oss:120b` for best accuracy
- **Quick testing**: Use `llama3.1:8b-instruct` for faster generation
- **Batch processing**: Use `qwen2.5:7b-instruct` for efficiency