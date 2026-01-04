#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}🎬 YouTube Streaming API Server v2.0.0${NC}"
echo -e "${BLUE}==================================================${NC}"

# Check Python version
echo -e "${YELLOW}🔍 Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}⚡ Activating virtual environment...${NC}"
source venv/bin/activate

# Install/upgrade pip
echo -e "${YELLOW}📦 Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Check for FFmpeg
echo -e "${YELLOW}🔍 Checking for FFmpeg...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}⚠️  FFmpeg is not installed! Audio conversion may not work.${NC}"
    echo -e "${YELLOW}Install FFmpeg:${NC}"
    echo -e "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo -e "  MacOS: brew install ffmpeg"
    echo -e "  Windows: Download from ffmpeg.org"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p downloads logs static

# Check for cookies file
if [ -f "cookies.txt" ]; then
    echo -e "${GREEN}🍪 Cookies file detected${NC}"
else
    echo -e "${YELLOW}⚠️  No cookies.txt file found${NC}"
    echo -e "${YELLOW}   Create cookies.txt for age-restricted videos${NC}"
fi

# Start server
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}🚀 Starting YouTube Streaming API Server...${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "${YELLOW}🌐 Server will run on: http://0.0.0.0:8000${NC}"
echo -e "${YELLOW}📚 API Docs: http://0.0.0.0:8000/docs${NC}"
echo -e "${YELLOW}📊 Health: http://0.0.0.0:8000/health${NC}"
echo -e "${YELLOW}📝 Logs: logs/streaming_api.log${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
echo -e "${BLUE}==================================================${NC}"

# Run the server
python main.py