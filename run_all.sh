#!/bin/bash

# run_all.sh - Run the entire counterfactual generation pipeline

echo "=========================================="
echo "LLM-VT-AL Pipeline Execution"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "WARNING: Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found!"
    echo "Please create config.yaml from config.yaml.example"
    exit 1
fi

# Check if input data exists
if [ ! -d "input_data" ] || [ -z "$(ls -A input_data)" ]; then
    echo "WARNING: No data found in input_data/"
    echo "Please add your train and test CSV files to input_data/"
    exit 1
fi

echo "Environment ready"
echo ""

# Check if required Python scripts exist
REQUIRED_SCRIPTS=("01_data_formatting.py" "02_counterfactual_over_generation.py" "03_counterfactual_filtering.py")
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Required script $script not found!"
        exit 1
    fi
done
echo ""

# Optional: Check if Ollama is running (if provider is ollama)
PROVIDER=$(grep "provider:" config.yaml | awk '{print $2}')
if [ "$PROVIDER" == "ollama" ]; then
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "WARNING: Ollama is not running!"
        echo "Please start Ollama in another terminal: ollama serve"
        exit 1
    fi
    echo "Ollama is running"
    echo ""
fi

# Run the pipeline
echo "=========================================="
echo "Step 1: Data Formatting"
echo "=========================================="
python 01_data_formatting.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 01 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Counterfactual Over-Generation"
echo "=========================================="
python 02_counterfactual_over_generation.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 02 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Counterfactual Filtering"
echo "=========================================="
python 03_counterfactual_filtering.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 03 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Check your results in output_data/"
echo ""
echo "To run evaluation:"
echo "  python 05_counterfactual_evaluation.py"
echo ""
