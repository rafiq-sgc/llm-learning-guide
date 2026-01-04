#!/bin/bash

# Script to generate PowerPoint presentation
# Usage: ./generate_ppt.sh

echo "🚀 Generating PowerPoint Presentation..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ Error: pip is not installed"
    exit 1
fi

# Install python-pptx if not installed
echo "📦 Checking for python-pptx..."
python3 -c "import pptx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing python-pptx..."
    if command -v pip3 &> /dev/null; then
        pip3 install python-pptx --quiet
    else
        pip install python-pptx --quiet
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install python-pptx"
        echo "💡 Try manually: pip install python-pptx"
        exit 1
    fi
    echo "✅ python-pptx installed"
else
    echo "✅ python-pptx already installed"
fi

echo ""
echo "📊 Creating presentation..."
python3 create_presentation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Presentation generated successfully!"
    echo "📁 File: LLM_Presentation.pptx"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Open LLM_Presentation.pptx in PowerPoint"
    echo "   2. Review and customize as needed"
    echo "   3. Add your name and date to slide 1"
    echo "   4. Practice your presentation!"
else
    echo ""
    echo "❌ Failed to generate presentation"
    exit 1
fi

