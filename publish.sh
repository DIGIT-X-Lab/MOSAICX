#!/bin/bash
# MOSAICX PyPI Publishing Script
# 
# This script automates the publishing process to PyPI
# Make sure you have your API token configured first!

set -e  # Exit on any error

echo "🚀 MOSAICX PyPI Publishing Script"
echo "================================="

# Check if .env file exists and source it
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env"
    source .env
else
    echo "⚠️  No .env file found. Make sure TWINE_USERNAME and TWINE_PASSWORD are set."
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source .venv/bin/activate

# Build the package
echo "📦 Building package..."
uv build

# Check the package
echo "✅ Checking package integrity..."
twine check dist/mosaicx-*

# Upload to PyPI
echo "🌐 Uploading to PyPI..."
twine upload dist/mosaicx-*

echo "🎉 Publishing complete!"
echo "📍 Your package is available at: https://pypi.org/project/mosaicx/"
