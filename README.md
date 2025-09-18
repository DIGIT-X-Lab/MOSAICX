
<div align="center">
  <img src="assets/mosaicx_logo.png" alt="MOSAICX Logo" width="800"/>
</div>

### Medical cOmputational Suite for Advanced Intelligent eXtraction

[![PyPI version](https://badge.fury.io/py/mosaicx.svg)](https://badge.fury.io/py/mosaicx)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> *"We built this because manually extracting data from thousands of medical reports was slowly killing our souls."*  
> — The DIGIT-X Team, after another late night of copy-pasting patient data

---

## 🎯 **What MOSAICX Actually Does**

MOSAICX turns this nightmare:
```
"Pat.-Nr.: 111111111, geb. 13.03.1940, Müller, Jane
Transthorakale Echokardiographie vom 06.10.2020 10:45
Befund: Mitralklappe physiologische Insuffizienz..."
```

Into this blessing:
```json
{
  "patient_id": "111111111",
  "age": 80,
  "sex": "Female", 
  "mitral_valve_grade": "Normal",
  "tricuspid_valve_grade": "Mild"
}
```

**The honest truth:** This tool was born out of pure desperation at DIGIT-X Lab when we realized we had 50,000+ radiology reports to process and our research budget couldn't afford a small army of medical students with Red Bull addictions.

---

## 🚀 **Quick Start (Because Time is Money)**

### Installation

**Option 1: Standard Installation**
```bash
pip install mosaicx
```

**Option 2: With UV (Faster & Better)**
```bash
uv add mosaicx
```

### Basic Usage
```bash
# 1. Generate a schema from natural language
mosaicx generate --desc "Patient demographics with valve conditions"

# 2. Extract data from PDF reports  
mosaicx extract --pdf report.pdf --schema PatientValveReport

# 3. Profit (literally, in research publications)
```

That's it. Seriously. We spent months making this as simple as possible because we're researchers, not software engineers, and we have better things to do than debug YAML files.

---

## 🏥 **Why We Built This (The Real Story)**

### **The Problem**
At DIGIT-X Lab (LMU University Hospital), we had:
- 📄 **50,000+ medical reports** in PDF format
- 🧠 **Brilliant researchers** who shouldn't be doing data entry
- ⏰ **Deadlines** that don't care about your manual extraction process
- 💰 **Limited budgets** (welcome to academic research)

### **Existing Solutions Were...**
- � **Too expensive** (enterprise NLP solutions cost more than our coffee budget)
- 🎯 **Too generic** (built for business documents, not medical reports)  
- 🔒 **Too cloud-dependent** (patient data doesn't leave our servers, period)
- 🤖 **Too rigid** (required predefined schemas that never match reality)

### **Our Approach**
We said "screw it" and built something that actually works for medical researchers:

- 🏠 **Runs locally** (your patient data stays in your building)
- 🧠 **Uses local LLMs** (Ollama + your own models)
- 📝 **Generates schemas from plain English** (describe what you want, get code)
- 🔧 **Actually handles real medical text** (German medical terms, inconsistent formats, coffee stains)
- 🎨 **Pretty terminal output** (because we're human beings who appreciate beauty)

---

## 🛠 **How It Actually Works**

### **The Magic Pipeline**
```
📄 PDF → 📝 Text (Docling) → 🤖 LLM + Schema → ✨ Structured Data
```

### **Schema Generation** 
```bash
mosaicx generate --desc "Echocardiography report with valve assessments"
```
- Uses local LLMs to understand your requirements
- Generates proper Pydantic models with validation
- Saves both Python classes and JSON schemas
- No more manually writing data models!

### **Data Extraction**
```bash  
mosaicx extract --pdf echo_report.pdf --schema PatientValveReport --model mistral
```
- Robust PDF text extraction (handles scanned docs, tables, weird formatting)
- Schema-driven extraction with validation
- Falls back gracefully when models get creative
- Silent error handling (no more spam in your terminal)

---

## 🎨 **Features We're Actually Proud Of**

### **🧠 Smart Schema Coercion**
- Handles German medical terms → English schema values
- "physiologische Insuffizienz" → "Normal" (because we live in Germany)
- Case-insensitive matching (because doctors don't follow style guides)

### **🛡️ Bulletproof Error Handling**
- Multiple fallback strategies when models fail
- JSON repair attempts (because GPT sometimes gets creative)
- Graceful degradation (something is better than nothing)

### **🎭 Clean Terminal Experience**
```
✨ Schema Model: PatientValveReport ✨

📋 Extraction Results: PatientValveReport
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field                    ┃ Extracted Value                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ patient_id               │ 0022768653                      │
│ valve_condition          │ Mild insufficiency              │
└──────────────────────────┴─────────────────────────────────┘
```

### **🔐 Privacy-First Architecture**
- All processing happens on your hardware
- No cloud APIs (your data never leaves your network)
- GDPR compliant by design (because we're in Europe)

---

## 📊 **Real-World Performance**

**What we've tested it on:**
- ✅ **German echocardiography reports** (our bread and butter)
- ✅ **Mixed-language medical documents** (German/English clinical notes)
- ✅ **Scanned PDFs** (with OCR quality ranging from "perfect" to "help me")
- ✅ **50,000+ reports** (and counting)

**Models that work well:**
- 🥇 **Mistral** (fast, reliable, good with medical terminology)  
- 🥈 **DeepSeek R1 70B** (slower but handles complex cases)
- 🥉 **Llama 3** (solid baseline performance)

**Honest accuracy rates:**
- 📊 **~85-90%** field extraction accuracy on clean reports
- 📊 **~70-80%** on challenging scanned documents
- 📊 **~95%** when you fine-tune the schema descriptions

*(These numbers are from actual usage, not cherry-picked benchmarks)*

---

## 🤝 **Contributing (We Need Your Help)**

### **What We'd Love Help With:**
- 🌍 **More language support** (French medical terms, anyone?)
- 🏥 **New medical domains** (pathology, radiology, lab reports)
- 🐛 **Bug reports** (especially weird edge cases we haven't seen)
- 📚 **Documentation** (making this more accessible to non-programmers)

### **How to Contribute:**
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-medical-nlp`
3. **Test** on real medical data (anonymized, please!)
4. **Submit** a pull request with examples

We're academics, so we appreciate proper citations and detailed explanations of your improvements.

---

## 📜 **License & Citation**

### **License**
AGPL-3.0 (GNU Affero General Public License v3.0)

*Translation: You can use it, modify it, and distribute it freely. If you improve it and share your improvements publicly, you need to share your code too. Fair's fair.*

### **Citation**
If MOSAICX helps with your research, we'd appreciate a citation:

```bibtex
@software{mosaicx2025,
  title={MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction},
  author={Shiyam Sundar, Lalith Kumar and DIGIT-X Lab Team},
  year={2025},
  url={https://github.com/LalithShiyam/MOSAICX},
  institution={DIGIT-X Lab, LMU Radiology, LMU University Hospital}
}
```

---

## 👥 **The Team Behind This**

### **DIGIT-X Lab @ LMU University Hospital**
- 🧠 **Lalith Kumar Shiyam Sundar, PhD** - *Lead Developer & Chief Coffee Consumer*
- 👥 **DIGIT-X Lab Team** - *The people who actually test this stuff*

**Contact:** lalith.shiyam@med.uni-muenchen.de  
**Lab:** https://www.digit-x-lab.com  
**Location:** Munich, Germany 🇩🇪

---

## 🙏 **Acknowledgments**

**Thanks to:**
- ☕ **Coffee** (the real MVP of this project)
- 🦙 **Ollama team** (for making local LLMs actually usable)
- 📄 **Docling team** (for solving PDF extraction so we didn't have to)
- 🐍 **Pydantic team** (for making data validation not terrible)
- 🎨 **Rich library** (for making our terminals beautiful)
- 🏥 **Our clinical collaborators** (for providing endless edge cases)
- 🎓 **LMU University Hospital** (for letting us build cool stuff)

---

## 🔮 **What's Next?**

### **Roadmap:**
- 🌐 **Web interface** (for the point-and-click crowd)
- 📊 **Batch processing tools** (because one PDF at a time is for amateurs)  
- 🤖 **Fine-tuned medical models** (when we get more GPU budget)
- 🔌 **API endpoints** (for the developers among us)
- � **Mobile app** (just kidding, we're not monsters)

### **Help Us Prioritize:**
Open an issue with your use case. We build what people actually need, not what sounds cool in academic papers.

---

## 💡 **Final Thoughts**

MOSAICX isn't perfect. It's not going to solve all your medical data problems overnight. But it's honest, it's practical, and it was built by people who actually use it every day.

We built this tool because we needed it, and we're sharing it because we think you might need it too. If it saves you even half the time it's saved us, we've done our job.

Happy extracting! 🚀

---

*Built with ❤️, ☕, and occasional frustration at DIGIT-X Lab, Munich*
