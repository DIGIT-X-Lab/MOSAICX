#!/bin/bash
# ==============================================================================
# MOSAICX Docker Entrypoint
# Starts Ollama server and runs MOSAICX commands
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║   ███╗   ███╗  ██████╗  ███████╗  █████╗  ██╗  ██████╗ ██╗  ██╗     ║"
echo "║   ████╗ ████║ ██╔═══██╗ ██╔════╝ ██╔══██╗ ██║ ██╔════╝ ╚██╗██╔╝     ║"
echo "║   ██╔████╔██║ ██║   ██║ ███████╗ ███████║ ██║ ██║       ╚███╔╝      ║"
echo "║   ██║╚██╔╝██║ ██║   ██║ ╚════██║ ██╔══██║ ██║ ██║       ██╔██╗      ║"
echo "║   ██║ ╚═╝ ██║ ╚██████╔╝ ███████║ ██║  ██║ ██║ ╚██████╗ ██╔╝ ██╗    ║"
echo "║   ╚═╝     ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═════╝ ╚═╝  ╚═╝     ║"
echo "║                                                                      ║"
echo "║   Medical cOmputational Suite for Advanced Intelligent eXtraction   ║"
echo "║   DIGIT-X Lab | LMU University Hospital                             ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if Ollama is ready
wait_for_ollama() {
    echo -e "${YELLOW}⏳ Waiting for Ollama server to start...${NC}"
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama server is ready${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ Ollama server failed to start${NC}"
    return 1
}

# Function to check/pull models
ensure_models() {
    echo -e "${YELLOW}📦 Checking available models...${NC}"
    
    local models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    
    # Check for gpt-oss:20b
    if echo "$models" | grep -q "gpt-oss:20b"; then
        echo -e "${GREEN}  ✓ gpt-oss:20b available${NC}"
    else
        echo -e "${YELLOW}  ⏳ Pulling gpt-oss:20b...${NC}"
        ollama pull gpt-oss:20b
    fi
    
    # Check for gpt-oss:120b
    if echo "$models" | grep -q "gpt-oss:120b"; then
        echo -e "${GREEN}  ✓ gpt-oss:120b available${NC}"
    else
        echo -e "${YELLOW}  ⏳ Pulling gpt-oss:120b...${NC}"
        ollama pull gpt-oss:120b
    fi
    
    echo -e "${GREEN}✓ All models ready${NC}"
}

# Start Ollama server in background
echo -e "${CYAN}🚀 Starting Ollama inference server...${NC}"
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
wait_for_ollama || exit 1

# Ensure models are available
ensure_models

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MOSAICX is ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Run MOSAICX with passed arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ -z "$1" ]; then
    mosaicx --help
elif [ "$1" = "shell" ]; then
    # Interactive shell mode
    exec /bin/bash
elif [ "$1" = "ollama" ]; then
    # Direct ollama commands
    shift
    exec ollama "$@"
else
    # Run mosaicx command
    exec mosaicx "$@"
fi
