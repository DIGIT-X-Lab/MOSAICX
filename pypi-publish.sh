#!/bin/bash
# Quick PyPI Publisher for MOSAICX
# Usage: ./pypi-publish.sh

set -e

echo "🚀 Publishing MOSAICX to PyPI..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ No .env file found!"
    echo "Please create .env with your PyPI token:"
    echo "export TWINE_USERNAME=__token__"
    echo "export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE"
    exit 1
fi

# Load environment variables
source .env

# Check if credentials are set
if [ -z "$TWINE_PASSWORD" ]; then
    echo "❌ TWINE_PASSWORD not set in .env file!"
    echo "Please add your PyPI token to .env"
    exit 1
fi

# Activate virtual environment and publish
source .venv/bin/activate
twine upload dist/mosaicx-1.0.2*

echo "🎉 Successfully published to PyPI!"
echo "📍 Available at: https://pypi.org/project/mosaicx/"
