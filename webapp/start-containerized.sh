#!/bin/bash

# MOSAICX WebApp - Fully Containerized Setup
# Everything runs in Docker containers (including Ollama)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🧬 MOSAICX - Fully Containerized Setup${NC}"
echo "======================================================="

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# Check Docker Compose
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}❌ Docker Compose not available${NC}"
    exit 1
fi

# Check for containerized config
if [ ! -f "docker-compose.full.yml" ]; then
    echo -e "${RED}❌ docker-compose.full.yml not found${NC}"
    echo -e "${YELLOW}This script requires the full containerized configuration${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 Starting fully containerized MOSAICX...${NC}"
echo -e "${YELLOW}This includes Ollama container - may take longer on first run${NC}"

# Start all services
if ! $DOCKER_COMPOSE_CMD -f docker-compose.full.yml up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

echo -e "${YELLOW}⏳ Waiting for Ollama container to initialize...${NC}"
sleep 10

# Check if Ollama container is ready
echo -e "${BLUE}📊 Container Status:${NC}"
$DOCKER_COMPOSE_CMD -f docker-compose.full.yml ps

echo ""
echo -e "${GREEN}🎉 Containerized MOSAICX Started!${NC}"
echo ""
echo -e "${CYAN}📱 Web Interface: ${GREEN}http://localhost:3000${NC}"
echo -e "${CYAN}📚 API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "${CYAN}🤖 Ollama API:    ${GREEN}http://localhost:11434${NC}"
echo ""

echo -e "${YELLOW}📥 Download models into Ollama container:${NC}"
echo -e "   ${CYAN}docker exec mosaicx-ollama ollama pull mistral:latest${NC}"
echo -e "   ${CYAN}docker exec mosaicx-ollama ollama pull gpt-oss:120b${NC}"
echo -e "   ${CYAN}docker exec mosaicx-ollama ollama list${NC}"
echo ""

echo -e "${BLUE}📋 Management:${NC}"
echo -e "   ${CYAN}Stop:    $DOCKER_COMPOSE_CMD -f docker-compose.full.yml down${NC}"
echo -e "   ${CYAN}Logs:    $DOCKER_COMPOSE_CMD -f docker-compose.full.yml logs -f${NC}"