#!/bin/bash

# MOSAICX WebApp Comprehensive Startup Script with Precheck
# Analyzes system readiness and starts webapp services

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

# Comprehensive System Precheck
function run_precheck() {
    echo -e "${BLUE}🔍 Running comprehensive system precheck...${NC}"
    local errors=0
    local warnings=0
    
    # Check 1: Operating System
    echo -n "Operating System: "
    case "$(uname -s)" in
        Darwin) echo -e "${GREEN}✅ macOS${NC}"; ;;
        Linux) echo -e "${GREEN}✅ Linux${NC}"; ;;
        MINGW*|CYGWIN*|MSYS*) echo -e "${GREEN}✅ Windows${NC}"; ;;
        *) echo -e "${YELLOW}⚠️  Unknown OS: $(uname -s)${NC}"; ((warnings++)); ;;
    esac
    
    # Check 2: Available RAM
    echo -n "Memory: "
    local ram_gb=0
    if command -v free >/dev/null 2>&1; then
        ram_gb=$(($(free -m | awk 'NR==2{print $2}') / 1024))
    elif [ -f /proc/meminfo ]; then
        ram_gb=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
    elif command -v sysctl >/dev/null 2>&1; then
        ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
    fi
    
    if [ "$ram_gb" -ge 32 ]; then
        echo -e "${GREEN}✅ ${ram_gb}GB (Excellent)${NC}"
    elif [ "$ram_gb" -ge 16 ]; then
        echo -e "${GREEN}✅ ${ram_gb}GB (Good)${NC}"
    elif [ "$ram_gb" -ge 8 ]; then
        echo -e "${YELLOW}⚠️  ${ram_gb}GB (Minimum - may be slow)${NC}"
        ((warnings++))
    elif [ "$ram_gb" -gt 0 ]; then
        echo -e "${RED}❌ ${ram_gb}GB (Insufficient)${NC}"
        ((errors++))
    else
        echo -e "${YELLOW}⚠️  Cannot determine RAM${NC}"
        ((warnings++))
    fi
    
    # Check 3: Disk Space  
    echo -n "Disk Space: "
    if command -v df >/dev/null 2>&1; then
        local disk_info=$(df -h . 2>/dev/null | tail -1 | awk '{print $4}')
        if echo "$disk_info" | grep -q 'T'; then
            echo -e "${GREEN}✅ ${disk_info} available (Excellent)${NC}"
        elif echo "$disk_info" | grep -qE '[5-9][0-9]G|[0-9]{3,}G'; then
            echo -e "${GREEN}✅ ${disk_info} available${NC}"
        elif echo "$disk_info" | grep -qE '[2-4][0-9]G'; then
            echo -e "${YELLOW}⚠️  ${disk_info} available${NC}"
            ((warnings++))
        elif echo "$disk_info" | grep -qE '[0-9]+G'; then
            echo -e "${RED}❌ ${disk_info} available (low)${NC}"
            ((errors++))
        else
            echo -e "${GREEN}✅ ${disk_info} available${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Cannot check disk space${NC}"
        ((warnings++))
    fi
    
    # Check 4: Docker Installation
    echo -n "Docker Engine: "
    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        echo -e "${GREEN}✅ v${docker_version}${NC}"
    else
        echo -e "${RED}❌ Not installed${NC}"
        ((errors++))
    fi
    
    # Check 5: Docker Service
    echo -n "Docker Service: "
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Running${NC}"
    else
        echo -e "${RED}❌ Not running${NC}"
        ((errors++))
    fi
    
    # Check 6: Docker Compose
    echo -n "Docker Compose: "
    if docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
        echo -e "${GREEN}✅ Integrated${NC}"
    elif command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker-compose"
        echo -e "${GREEN}✅ Standalone${NC}"
    else
        echo -e "${RED}❌ Not available${NC}"
        ((errors++))
    fi
    
    # Check 7: Port Availability
    local ports_in_use=()
    if command -v lsof >/dev/null 2>&1; then
        echo -n "Port 3000 (Frontend): "
        if lsof -i :3000 >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  In use${NC}"
            ((warnings++))
        else
            echo -e "${GREEN}✅ Available${NC}"
        fi
        
        echo -n "Port 8000 (Backend): "
        if lsof -i :8000 >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  In use${NC}"
            ((warnings++))
        else
            echo -e "${GREEN}✅ Available${NC}"
        fi
    fi
    
    # Check 8: Configuration Files
    echo -n "Docker Compose Config: "
    if [ -f "docker-compose.yml" ]; then
        echo -e "${GREEN}✅ Present${NC}"
    else
        echo -e "${RED}❌ Missing${NC}"
        ((errors++))
    fi
    
    # Summary
    echo ""
    echo -e "${BLUE}📊 Precheck Summary:${NC}"
    if [ "$errors" -eq 0 ] && [ "$warnings" -eq 0 ]; then
        echo -e "${GREEN}🎉 All checks passed!${NC}"
        return 0
    elif [ "$errors" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  ${warnings} warning(s) - should work${NC}"
        return 0
    else
        echo -e "${RED}❌ ${errors} error(s), ${warnings} warning(s)${NC}"
        return 1
    fi
}

# Ollama Detection and Analysis
function check_ollama() {
    echo -e "${BLUE}🔍 Checking Ollama setup...${NC}"
    
    echo -n "Ollama CLI: "
    if command -v ollama >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Installed${NC}"
    else
        echo -e "${YELLOW}⚠️  Not installed${NC}"
    fi
    
    echo -n "Ollama Service: "
    if curl -s --connect-timeout 3 http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Running${NC}"
        
        echo -n "Available Models: "
        if command -v curl >/dev/null 2>&1; then
            local models_json=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
            if echo "$models_json" | grep -q '"models"' 2>/dev/null; then
                local model_count=$(echo "$models_json" | grep -o '"name":' | wc -l | tr -d ' ')
                if [ "$model_count" -gt 0 ]; then
                    echo -e "${GREEN}✅ ${model_count} model(s)${NC}"
                    return 0
                else
                    echo -e "${YELLOW}⚠️  No models${NC}"
                    return 1
                fi
            else
                echo -e "${YELLOW}⚠️  Cannot check${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}⚠️  Cannot check${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Not running${NC}"
        return 1
    fi
}

# Main execution
echo ""

# Run precheck
if ! run_precheck; then
    echo -e "${RED}System precheck failed. Please fix errors before continuing.${NC}"
    exit 1
fi

echo ""

# Check Ollama setup
if check_ollama; then
    CONFIG_FILE="docker-compose.yml"
    echo -e "${GREEN}✅ Will use external Ollama${NC}"
else
    echo -e "${YELLOW}Ollama not ready on host${NC}"
    echo -n -e "${BLUE}Use containerized setup? (y/N): ${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if [ -f "docker-compose.full.yml" ]; then
            CONFIG_FILE="docker-compose.full.yml"
            echo -e "${GREEN}✅ Using containerized setup${NC}"
        else
            echo -e "${RED}❌ docker-compose.full.yml not found${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Please start Ollama and try again:${NC}"
        echo -e "   ${CYAN}ollama serve${NC}"
        echo -e "   ${CYAN}ollama pull mistral:latest${NC}"
        exit 1
    fi
fi

echo ""

# Final validation
echo -e "${BLUE}🔧 Final validation...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

if ! $DOCKER_COMPOSE_CMD -f "$CONFIG_FILE" config >/dev/null 2>&1; then
    echo -e "${RED}❌ Invalid Docker Compose config${NC}"
    exit 1
fi

echo -e "${GREEN}✅ System ready${NC}"

# Start services
echo ""
echo -e "${BLUE}🚀 Starting MOSAICX WebApp...${NC}"
echo -e "Configuration: ${CYAN}$CONFIG_FILE${NC}"

if ! $DOCKER_COMPOSE_CMD -f "$CONFIG_FILE" up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait and check
echo -e "${YELLOW}⏳ Waiting for services...${NC}"
sleep 3

# Show status
echo -e "${BLUE}📊 Service Status:${NC}"
$DOCKER_COMPOSE_CMD -f "$CONFIG_FILE" ps

# Health checks
echo -e "${BLUE}🏥 Health Checks:${NC}"

echo -n "Frontend: "
if curl -s --connect-timeout 3 http://localhost:3000 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Starting${NC}"
fi

echo -n "Backend: "
if curl -s --connect-timeout 3 http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Starting${NC}"
fi

# Instructions
echo ""
echo -e "${GREEN}🎉 MOSAICX Smart Contract Generator Started!${NC}"
echo ""
echo -e "${CYAN}📱 Web Interface: ${GREEN}http://localhost:3000${NC}"
echo -e "${CYAN}📚 API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "${CYAN}🤖 Ollama API:    ${GREEN}http://localhost:11434${NC}"
echo ""

if [[ "$CONFIG_FILE" == "docker-compose.full.yml" ]]; then
    echo -e "${YELLOW}📥 Containerized Ollama Commands:${NC}"
    echo -e "   ${CYAN}docker exec mosaicx-ollama ollama pull mistral:latest${NC}"
    echo -e "   ${CYAN}docker exec mosaicx-ollama ollama list${NC}"
    echo ""
fi

echo -e "${BLUE}📋 Management:${NC}"
echo -e "   ${CYAN}Stop:     $DOCKER_COMPOSE_CMD -f $CONFIG_FILE down${NC}"
echo -e "   ${CYAN}Logs:     $DOCKER_COMPOSE_CMD -f $CONFIG_FILE logs -f${NC}"
echo -e "   ${CYAN}Restart:  $DOCKER_COMPOSE_CMD -f $CONFIG_FILE restart${NC}"