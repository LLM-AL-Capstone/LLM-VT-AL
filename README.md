# LLM-VT-AL: Enhanced Counterfactual Data Augmentation

A scalable, LLM-based pipeline for generating high-quality counterfactual examples to improve few-shot text classification performance.

## Quick Start

### 1. Setup
```bash
source venv/bin/activate
```

### 2. Configure LLM Provider

**Option A: Use Gemini**
```yaml
# config.yaml
llm:
  provider: gemini
  gemini:
    api_key: YOUR_GEMINI_API_KEY
    model: gemini-2.5-flash  
```

**Option B: Use Azure OpenAI**
```yaml
# config.yaml
llm:
  provider: openai
  openai:
    api_key: YOUR_AZURE_OPENAI_API_KEY
    azure_endpoint: https://your-endpoint.openai.azure.com/openai
    model: gpt-4o  # Or gpt-4, gpt-3.5-turbo
```
**Option C: Use Ollama (Local)**
```bash
ollama serve
ollama pull qwen2.5:7b
```
```yaml
# config.yaml
llm:
  provider: ollama
  ollama:
    model: qwen2.5:7b
```

### 3. Prepare Dataset
Place your CSV files in `input_data/`:
- `emotions_train.csv` (training data)
- `emotions_test.csv` (test data)

Required columns: `id`, `example`, `Label`

### 4. Run Pipeline
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all scripts sequentially
chmod +x run_all.sh
./run_all.sh

# Or run individually
python 01_data_formatting.py
python 02_counterfactual_over_generation.py
python 03_counterfactual_filtering.py
python 04_counterfactual_evaluation.py
```

## 🔄 Pipeline Overview

```
Input: 500 Training Examples
        ↓
Script 01: Enhanced Pattern Identification
    • Processes x examples per label (can be configured)
    • Batch processing for efficiency
    • Generates patterns
        ↓
Script 02: Counterfactual Generation  
    • counterfactuals generated
        ↓
Script 03: Three-Stage Quality Filtering
    • Filter 1 (Heuristic)
    • Filter 2 (Semantic) 
    • Filter 3 (Discriminator)
        ↓
Script 04: Enhanced Evaluation
    • Tests up to 150+ shot classification
    • Complete test dataset evaluation 
    • Robust metrics: Precision, Recall, F-Score, Accuracy
        ↓
Output: f-1,accuracy scores
```
## 📁 Project Structure

```
LLM-VT-AL/
├── config.yaml                          # Main configuration
├── run_all.sh                          # Pipeline automation
├── requirements.txt                     # Dependencies
│
├── utils/                               # Core utilities
│   ├── llm_provider.py                  # LLM abstraction layer
│   └── __init__.py                      # Utility exports
│
├── 01_data_formatting.py                # Enhanced pattern identification
├── 02_counterfactual_over_generation.py # Counterfactual generation  
├── 03_counterfactual_filtering.py       # Three-stage filtering
├── 04_counterfactual_evaluation.py      # Enhanced evaluation
│
├── input_data/                          # Your datasets
│   ├── emotions_train.csv
│   └── emotions_test.csv
│
└── output_data/                         # Generated results
    ├── [42]filtered_emotions_train.csv  # high-quality counterfactuals
    └── archive/gpt/                     # Evaluation results
```

## 🔧 Script Details

### Script 01: Enhanced Data Formatting
**Purpose**: Pattern identification and candidate phrase generation
**Enhancements**: 
- Processes x examples per label (can be configured)
- Batch processing for efficiency

### Script 02: Counterfactual Over-Generation  
**Purpose**: Generate complete counterfactual sentences

### Script 03: Counterfactual Filtering
**Purpose**: Three-stage quality control
**Results**: high-quality counterfactuals
- Stage 1: Remove meta-responses  
- Stage 2: Validate semantic consistency
- Stage 3: Verify label transformation

### Script 04: Enhanced Evaluation
**Purpose**: Measure counterfactual effectiveness
**Enhancements**:
- Few-shot classification capability
- Complete test dataset evaluation
- Multiple evaluation metrics

## Troubleshooting

### Common Issues

**Gemini Rate Limits**
```bash
# Check if thinking mode is properly disabled
grep "thinking_budget=0" utils/llm_provider.py
```

**Configuration Issues**
```bash
# Verify config structure
python -c "from utils import load_config; print(load_config())"
```

**File Not Found**
```bash
# Check input data exists
ls -la input_data/
```

**Script Dependencies**
Scripts must run in order: 01 → 02 → 03 → 04

## 🎨 Customization

### Switch LLM Providers
```yaml
# Use Ollama instead of Gemini
llm:
  provider: ollama
  ollama:
    model: qwen2.5:7b
```

### Adjust Processing Scale
```yaml
processing:
  max_examples_per_label: 30    # Reduce for faster processing
  token_limit: 10000000         # Lower budget
  evaluation_shots: [10, 30, 50]  # Fewer evaluation points
```

### Use Different Dataset
```yaml
dataset:
  train_file: your_data_train.csv
  test_file: your_data_test.csv
  columns:
    id: sample_id
    text: sentence  
    label: category
```
## 📝 License

This enhanced LLM-VT-AL pipeline builds upon counterfactual data augmentation research and implements optimizations for scalable, production-ready text classification.

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requierments.txt
```

### Configuration

1. **Create/Edit `config.ini`:**

```ini
[settings]
openai_api = YOUR_OPENAI_API_KEY_HERE
data_file = emotions.csv
test_file = emotions_test.csv
seed = 42
```

2. **Prepare Input Data:**

Place these files in `input_data/`:
- `emotions.csv` - Training data (500 rows)
- `emotions_test.csv` - Test data (100 rows)

**Required schema for training data:**
```csv
id,example,Label
ss104440,"i feel hopeless because i don't know how to fix this","sadness"
ss224831,"i feel so blessed to have such wonderful friends","love"
```

**Required schema for test data:**
```csv
id,example,Label
test001,"i am so excited about this","joy"
test002,"i feel really sad today","sadness"
```

## Expected Outputs

### Script 01 Output Example

**File:** `output_data/[42]annotated_data_with_pattern_emotions.csv`

```csv
id,example,Label,pattern,hihglight
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE] because [CONTEXT]","hopeless"
ss224831,"i feel blessed...","love","i feel [CANDIDATE] because [CONTEXT]","blessed"
```

**File:** `output_data/interim_output/[42]candidate_phrases_emotions.csv`

```csv
id,ori_text,ori_label,pattern,highlight,candidate_phrases,target_label
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE]...","hopeless","hopeful","joy"
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE]...","hopeless","frustrated","anger"
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE]...","hopeless","terrified","fear"
```

### Script 02 Output Example

**File:** `output_data/[42]counterfactuals_emotions.csv`

```csv
id,ori_text,ori_label,pattern,highlight,candidate_phrases,target_label,counterfactual
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE]...","hopeless","hopeful","joy","i feel hopeful because i can't wait to celebrate this"
```

### Script 03 Output Example

**File:** `output_data/[42]filtered_emotions.csv`

```csv
id,ori_text,ori_label,pattern,highlight,candidate_phrases,target_label,counterfactual,heuristic_filtered,matched_pattern,is_ori,is_target
ss104440,"i feel hopeless...","sadness","i feel [CANDIDATE]...","hopeless","hopeful","joy","i feel hopeful...","True","True","False","True"
```

**File:** `output_data/[42]fine_tuneset_emotions.csv` (subset where all filters = True)

### Script 04 Output Example

**File:** `output_data/archive/gpt/[42][GPT]_counter_emotions_prf.csv`

```csv
shots,precision,recall,fscore
10,0.52,0.49,0.50
15,0.61,0.58,0.59
30,0.73,0.70,0.71
50,0.81,0.78,0.79
70,0.85,0.82,0.83
90,0.87,0.84,0.85
120,0.89,0.86,0.87
```

---