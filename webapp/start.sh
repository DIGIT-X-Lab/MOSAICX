#!/bin/bash

# MOSAICX WebApp Simple Startup Script
# Clean and reliable startup with basic checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🧬 MOSAICX Smart Contract Generator WebApp${NC}"
echo "======================================================="

# Basic system checks
function basic_checks() {
    echo -e "${BLUE}🔍 Basic system checks...${NC}"
    
    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
        echo -e "${GREEN}✅ Docker Compose available${NC}"
    elif command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker-compose"
        echo -e "${GREEN}✅ Docker Compose (standalone) available${NC}"
    else
        echo -e "${RED}❌ Docker Compose not available${NC}"
        exit 1
    fi
    
    # Check config file
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}❌ docker-compose.yml not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Configuration file found${NC}"
}

# Check Ollama setup
function check_ollama() {
    echo -e "${BLUE}🔍 Checking Ollama...${NC}"
    
    if curl -s --connect-timeout 3 http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama is running${NC}"
        
        # Check for models
        local models_response=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
        if echo "$models_response" | grep -q '"models"'; then
            echo -e "${GREEN}✅ Models are available${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  No models found${NC}"
            echo -e "${CYAN}   Suggestion: ollama pull mistral:latest${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}⚠️  Ollama not running on localhost:11434${NC}"
        return 1
    fi
}

# Navigate to script directory
cd "$(dirname "$0")"

# Run basic checks
basic_checks

echo ""

# Check Ollama and decide configuration
if check_ollama; then
    CONFIG_FILE="docker-compose.yml"
    echo -e "${GREEN}✅ Using external Ollama setup${NC}"
else
    echo -e "${YELLOW}Ollama not detected on localhost:11434${NC}"
    echo ""
    echo -e "${BLUE}🔧 Setup Options:${NC}"
    echo -e "   ${CYAN}1) Install Ollama (recommended for best performance)${NC}"
    echo -e "   ${CYAN}2) Use fully containerized setup${NC}"
    echo ""
    
    # Check if Ollama is installed but not running
    if command -v ollama >/dev/null 2>&1; then
        echo -e "${YELLOW}💡 Ollama is installed but not running. Start it with:${NC}"
        echo -e "   ${CYAN}ollama serve${NC}"
        echo -e "   ${CYAN}ollama pull mistral:latest${NC}"
        echo ""
    else
        echo -e "${YELLOW}💡 To install Ollama:${NC}"
        echo -e "   ${CYAN}curl -fsSL https://ollama.com/install.sh | sh${NC}"
        echo -e "   ${CYAN}ollama serve${NC}"
        echo -e "   ${CYAN}ollama pull mistral:latest${NC}"
        echo ""
    fi
    
    echo -n -e "${BLUE}Use containerized setup instead? (y/N): ${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if [ -f "docker-compose.full.yml" ]; then
            CONFIG_FILE="docker-compose.full.yml"
            echo -e "${GREEN}✅ Using containerized setup${NC}"
            echo -e "${YELLOW}Note: You'll need to download models after startup${NC}"
        else
            echo -e "${RED}❌ docker-compose.full.yml not found${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Setup Ollama first, then run this script again${NC}"
        exit 1
    fi
fi

echo ""

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
echo -e "Configuration: ${CYAN}$CONFIG_FILE${NC}"

if ! $DOCKER_COMPOSE_CMD -f "$CONFIG_FILE" up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    echo -e "${YELLOW}Try: docker system prune -f${NC}"
    exit 1
fi

# Wait for services
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 3

# Check service status
echo -e "${BLUE}📊 Service Status:${NC}"
$DOCKER_COMPOSE_CMD -f "$CONFIG_FILE" ps

# Simple health check
echo -e "${BLUE}🏥 Health Check:${NC}"

echo -n "Frontend: "
if curl -s --connect-timeout 5 http://localhost:3000 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Starting up${NC}"
fi

echo -n "Backend: "
if curl -s --connect-timeout 5 http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Starting up${NC}"
fi

# Success message
echo ""
echo -e "${GREEN}🎉 MOSAICX WebApp Started!${NC}"
echo ""
echo -e "${CYAN}📱 Web Interface: ${GREEN}http://localhost:3000${NC}"
echo -e "${CYAN}📚 API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "${CYAN}🤖 Ollama API:    ${GREEN}http://localhost:11434${NC}"
echo ""

# Containerized Ollama instructions
if [[ "$CONFIG_FILE" == "docker-compose.full.yml" ]]; then
    echo -e "${YELLOW}📥 Next Steps - Download Models into Container:${NC}"
    echo -e "   ${CYAN}# For good quality (~4GB):${NC}"
    echo -e "   ${CYAN}docker exec mosaicx-ollama ollama pull mistral:latest${NC}"
    echo ""
    echo -e "   ${CYAN}# For best quality (~70GB, needs 32GB+ RAM):${NC}"
    echo -e "   ${CYAN}docker exec mosaicx-ollama ollama pull gpt-oss:120b${NC}"
    echo ""
    echo -e "   ${CYAN}# List downloaded models:${NC}"
    echo -e "   ${CYAN}docker exec mosaicx-ollama ollama list${NC}"
    echo ""
    echo -e "${GREEN}💡 Tip: Download a model before using the webapp!${NC}"
    echo ""
fi

# Management commands
echo -e "${BLUE}📋 Management Commands:${NC}"
echo -e "   ${CYAN}Stop:    $DOCKER_COMPOSE_CMD -f $CONFIG_FILE down${NC}"
echo -e "   ${CYAN}Logs:    $DOCKER_COMPOSE_CMD -f $CONFIG_FILE logs -f${NC}"
echo -e "   ${CYAN}Restart: $DOCKER_COMPOSE_CMD -f $CONFIG_FILE restart${NC}"