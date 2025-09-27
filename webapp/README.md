# MOSAICX WebApp - Smart Contract Generator

🌐 **Interactive Web Interface for Medical AI Data Structuring**

A modern web application that showcases MOSAICX capabilities through an intuitive interface. Generate smart contracts (Pydantic schemas) from natural language, extract structured data from medical PDFs, and create timeline summaries from clinical reports.

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  nginx          │    │  FastAPI        │    │  Ollama         │
│  Frontend       │────│  Backend        │────│  LLM Server     │
│  (Port 3000)    │    │  (Port 8000)    │    │  (Port 11434)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐            ┌──────────┐           ┌──────────┐
    │ Glass   │            │ MOSAICX  │           │ LLM      │
    │ Morphism│            │ CLI      │           │ Models   │
    │ UI      │            │ Direct   │           │          │
    └─────────┘            └──────────┘           └──────────┘
```

### **Tech Stack:**
- **Frontend**: Modern HTML5, Vanilla JavaScript, CSS3 Glass Morphism
- **Backend**: FastAPI, Python 3.11+, MOSAICX CLI Integration  
- **AI Engine**: Ollama with multiple LLM models (gpt-oss:120b)
- **Styling**: Electric cyan theme with glass morphism effects
- **Deployment**: Docker Compose with optimized containers

---

## 🚀 **Complete Setup Guide**

### **Step 1: Install Prerequisites**

#### **Install Docker Desktop**
```bash
# macOS (using Homebrew)
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop/
```

#### **Install Ollama (Recommended for Best Performance)**
```bash
# macOS/Linux - Automatic installation
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from: https://ollama.com/download/windows

# Verify installation
ollama --version
```

### **Step 2: Start Ollama & Download Models**
```bash
# Start Ollama service (keep this terminal open)
ollama serve

# In another terminal, download a model
ollama pull mistral:latest     # Good quality, ~4GB
# OR
ollama pull gpt-oss:120b      # Best quality, ~70GB (requires 32GB+ RAM)

# Verify model is downloaded
ollama list
```

### **Step 3: Start MOSAICX WebApp**

#### **Option A: Quick Start (Most Users)**
```bash
cd webapp
./start.sh
```
- ✅ Automatically detects your setup
- ✅ Chooses best configuration  
- ✅ Provides guidance if anything is missing

#### **Option B: Full System Check**
```bash
./start-full-check.sh
```
- ✅ Analyzes your system (RAM, disk, ports)
- ✅ Validates all requirements
- ✅ Perfect for first-time setup or troubleshooting

#### **Option C: Fully Containerized (No Ollama Install Needed)**
```bash
./start-containerized.sh
```
- ✅ Everything runs in Docker containers
- ✅ No need to install Ollama separately
- ✅ Download models after startup:
  ```bash
  docker exec mosaicx-ollama ollama pull mistral:latest
  ```

### **Step 4: Access Your WebApp**
Once started, open your browser:
- **🌐 Web Interface:** http://localhost:3000
- **📚 API Documentation:** http://localhost:8000/docs
- **🤖 Ollama API:** http://localhost:11434

### **Access the Application:**
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Ollama API**: http://localhost:11434 (Option 1) or via container (Option 2)

## ❓ **Troubleshooting**

### **"Docker not found" or "Docker not running"**
```bash
# Check if Docker is installed
docker --version

# If not installed, install Docker Desktop:
# macOS: brew install --cask docker
# Windows/Linux: https://www.docker.com/products/docker-desktop/

# Start Docker Desktop and try again
```

### **"Ollama not found" or "No models available"**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Download a model (in another terminal)
ollama pull mistral:latest

# Verify model is available
ollama list
```

### **"Port already in use" errors**
```bash
# Check what's using the ports
lsof -i :3000  # Frontend port
lsof -i :8000  # Backend port
lsof -i :11434 # Ollama port

# Stop conflicting services or use different ports
```

### **"Insufficient memory" warnings**
- **Minimum:** 8GB RAM (use `mistral:latest`)
- **Recommended:** 16GB+ RAM 
- **Optimal:** 32GB+ RAM (can use `gpt-oss:120b`)

### **Still having issues?**
Run the full diagnostic:
```bash
./start-full-check.sh
```
This will analyze your system and provide specific recommendations.

## 📋 **System Requirements**
- **Docker:** Desktop or Engine 20.10+
- **Memory:** 8GB minimum, 16GB+ recommended, 32GB+ optimal
- **Storage:** 10GB+ free space for models and containers
- **OS:** macOS, Linux, or Windows with WSL2

---

## 🎯 **Features & Usage**

### **🔬 Schema Generator**
Transform natural language into validated Pydantic models:

1. **Enter Description**: "Echocardiography report with LVEF and valve assessments"
2. **Select Model**: Choose from available LLMs based on speed/accuracy needs
3. **Generate**: Creates complete Python/Pydantic model with validation
4. **Download**: Save generated schemas for reuse

**Sample Descriptions:**
- `Complete blood count with patient ID, test date, hemoglobin, hematocrit, WBC count, differential counts, and reference ranges`
- `CT chest report with patient demographics, study date, lung findings, mediastinal findings, and radiologist impression`
- `Surgical pathology report with specimen type, microscopic findings, tumor staging, and pathologist diagnosis`

### **📄 PDF Extractor**
Extract structured data from medical documents:

1. **Upload PDF**: Drag-and-drop medical reports (supports .pdf only)
2. **Select Schema**: Choose from generated schemas or schema registry
3. **Configure Model**: Adjust LLM and temperature settings
4. **Extract**: Processes PDF and returns structured JSON data
5. **Export**: Download results as JSON or view in-browser

**Supported Document Types:**
- Laboratory reports (CBC, BMP, liver function tests)
- Radiology reports (CT, MRI, X-ray, ultrasound)
- Cardiology reports (echo, ECG, stress tests)
- Pathology reports (surgical, cytology)

### **📊 Report Summarizer**
Generate timeline-based clinical summaries:

1. **Upload Multiple Reports**: Select related clinical documents
2. **Set Patient ID**: Organize reports by patient identifier
3. **Configure Processing**: Choose model and temperature
4. **Generate Timeline**: Creates chronological summary with key findings
5. **Export Summary**: Download as JSON or formatted report

**Best Use Cases:**
- Longitudinal patient tracking
- Disease progression monitoring
- Clinical research data compilation
- Care transition summaries

---

## ⚙️ **Configuration**

### **Environment Variables:**
```bash
# Backend Configuration
OPENAI_BASE_URL=http://ollama:11434/v1
OPENAI_API_KEY=ollama
MOSAICX_DEFAULT_MODEL=gpt-oss:120b
MOSAICX_LOG_LEVEL=INFO

# Frontend Configuration  
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **Model Selection Guide:**

| Model | Size | Speed | Accuracy | Memory | Use Case |
|-------|------|-------|----------|---------|----------|
| `gpt-oss:120b` | ~120B | Slow | ★★★★★ | 64GB+ | Complex schemas, high accuracy |
| `llama3.1:8b-instruct` | ~8B | Fast | ★★★★☆ | 16GB+ | Balanced performance |
| `qwen2.5:7b-instruct` | ~7B | Fastest | ★★★☆☆ | 12GB+ | Quick testing, batch processing |

### **Hardware Optimization:**
```bash
# For limited memory, use quantized models
docker exec mosaicx-ollama ollama pull llama3.1:8b-instruct-q4_0

# Monitor GPU usage
docker exec mosaicx-ollama nvidia-smi

# Check Ollama model status
curl http://localhost:11434/api/tags
```

---

## 🛠️ **Development Setup**

### **Local Development (without Docker):**

**1. Backend Setup:**
```bash
cd webapp/backend

# Install dependencies
pip install -r requirements.txt

# Install MOSAICX in development mode
cd ../../
pip install -e .

# Start Ollama separately
ollama serve &

# Run FastAPI server
cd webapp/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**2. Frontend Setup:**
```bash
cd webapp/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### **API Testing:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Generate schema
curl -X POST http://localhost:8000/api/v1/generate-schema \
  -H "Content-Type: application/json" \
  -d '{"description": "Simple patient record with name and age", "class_name": "PatientRecord"}'

# List available schemas
curl http://localhost:8000/api/v1/schemas
```

---

## 📋 **Sample Data & Templates**

### **Pre-built Schema Templates:**
Located in `webapp/shared/schema_templates.md`:
- Cardiovascular (Echo, ECG)
- Laboratory (CBC, BMP)
- Radiology (CT, MRI)
- Pathology (Surgical, Cytology)

### **Sample Medical PDFs:**
Located in `webapp/shared/sample_data/`:
- `sample_echo_report.pdf` - Cardiac ultrasound report
- `sample_cbc_report.pdf` - Complete blood count lab results
- `sample_radiology_series/` - Sequential CT reports for timeline demo

### **Demo Workflow:**
1. **Start with Schema Generation**: Use provided templates
2. **Test PDF Extraction**: Upload sample PDFs with matching schemas
3. **Try Summarization**: Use the radiology series for timeline generation

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**Services won't start:**
```bash
# Check Docker resources
docker system df
docker system prune -f

# Restart services
docker compose down    # or: docker-compose down
docker compose up -d   # or: docker-compose up -d
```

**Model download fails:**
```bash
# Manually download models
docker exec -it mosaicx-ollama ollama pull gpt-oss:120b
docker exec -it mosaicx-ollama ollama pull llama3.1:8b-instruct

# Check model status
docker exec -it mosaicx-ollama ollama list
```

**PDF extraction errors:**
```bash
# Check backend logs
docker-compose logs mosaicx-backend

# Verify PDF is text-based (not scanned image)
# Use OCR preprocessing if needed: tesseract input.pdf output.pdf
```

**Frontend build issues:**
```bash
# Clear Next.js cache
docker exec -it mosaicx-frontend rm -rf .next

# Rebuild frontend
docker compose build mosaicx-frontend  # or: docker-compose build mosaicx-frontend
```

### **Performance Optimization:**

**Memory Issues:**
- Use quantized models (`q4_0` suffix)
- Reduce batch sizes
- Close unused applications

**Slow Processing:**
- Switch to smaller models (llama3.1:8b-instruct)
- Enable GPU acceleration
- Increase available RAM

### **Debugging Commands:**
```bash
# Check all service logs
docker compose logs -f  # or: docker-compose logs -f

# Monitor resource usage
docker stats

# Test API endpoints
curl -f http://localhost:8000/api/v1/health
curl -f http://localhost:3000

# Access container shells
docker exec -it mosaicx-backend bash
docker exec -it mosaicx-frontend sh
```

---

## 🏥 **Use Cases & Applications**

### **Clinical Research:**
- **Cohort Studies**: Extract standardized data from diverse clinical reports
- **Outcomes Research**: Structure endpoints from heterogeneous documents
- **Quality Metrics**: Automate clinical quality measure extraction

### **Clinical Decision Support:**
- **Risk Stratification**: Extract and structure risk factors
- **Care Pathways**: Standardize clinical workflow documentation
- **Adverse Events**: Structure safety data from clinical narratives

### **Healthcare Operations:**
- **Revenue Cycle**: Extract billable procedures and diagnoses
- **Compliance**: Structure regulatory reporting requirements
- **Care Coordination**: Generate standardized handoff summaries

---

## 🎓 **From DIGIT-X Lab**

**MOSAICX WebApp** is developed by the [DIGIT-X Lab](https://www.linkedin.com/company/digitx-lmu/) at LMU Munich University, extending our mission to democratize clinical AI through practical, privacy-preserving tools.

### **Key Principles:**
- **Privacy-First**: All processing happens locally, no PHI leaves your environment
- **Clinician-Friendly**: Designed by researchers who understand clinical workflows
- **Production-Ready**: Built for real-world deployment in healthcare settings
- **Open Science**: Supporting reproducible research and clinical AI development

---

## 📞 **Support & Contributing**

### **Getting Help:**
- **Issues**: [GitHub Issues](https://github.com/LalithShiyam/MOSAICX/issues)
- **Research Inquiries**: lalith.shiyam@med.uni-muenchen.de
- **Commercial Support**: lalith@zenta.solutions

### **Contributing:**
- **Bug Reports**: Include logs, system info, and reproduction steps
- **Feature Requests**: Describe clinical use cases and requirements
- **Code Contributions**: Follow existing patterns and include tests
- **Documentation**: Help improve examples and clinical guidance

---

*Built with ❤️ for the medical community. **Structure first. Insight follows.***

**MOSAICX WebApp**: Making clinical AI accessible through intuitive interfaces while maintaining the rigor and privacy requirements of healthcare environments.