# LLM-VT-AL: Enhanced Counterfactual Data Augmentation

A scalable, LLM-based pipeline for generating high-quality counterfactual examples to improve few-shot text classification performance.

## Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/LLM-AL-Capstone/LLM-VT-AL.git
cd LLM-VT-AL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM Provider

Create a `config.yaml` file in the project root (or copy from `config.yaml.example`):

**Option A: Azure OpenAI (Recommended)**
```yaml
llm:
  provider: openai
  openai:
    api_key: YOUR_AZURE_OPENAI_API_KEY
    azure_endpoint: https://your-endpoint.openai.azure.com/
    api_version: "2024-08-01-preview"
    model: gpt-4o  # Or gpt-5-nano-2025-08-07, gpt-4, gpt-3.5-turbo
  
  models:
    pattern_identification:
      temperature: 1.0
      max_tokens: 4096
    candidate_generation:
      temperature: 1.0
      max_tokens: 2048
    counterfactual_generation:
      temperature: 1.0
      max_tokens: 1024
    semantic_filtering:
      temperature: 1.0
      max_tokens: 512
    discriminator_filtering:
      temperature: 1.0
      max_tokens: 256

dataset:
  train_file: emotions_train.csv
  test_file: emotions_test.csv
  columns:
    id: id
    text: example
    label: Label
  exclude_labels: []

processing:
  seed: 42
  max_examples_per_label: 50
  evaluation_shots: [10, 15, 30, 50, 70, 90, 120]
```

**Option B: Google Gemini**
```yaml
llm:
  provider: gemini
  gemini:
    api_key: YOUR_GEMINI_API_KEY
    model: gemini-2.5-flash
```

**Option C: Ollama (Local)**
```bash
# First, install and start Ollama
ollama serve
ollama pull qwen2.5:7b
```

```yaml
llm:
  provider: ollama
  ollama:
    base_url: http://localhost:11434
    model: qwen2.5:7b
```

### 3. Prepare Your Dataset

Place your CSV files in the `input_data/` directory:

```
input_data/
├── emotions_train.csv
├── emotions_test.csv
├── yelp_train.csv
├── yelp_test.csv
└── massive_train.csv
```

**Required CSV format:**
```csv
id,example,Label
1,"I feel so happy today!","joy"
2,"This makes me really sad","sadness"
3,"I'm terrified of heights","fear"
```

**Column requirements:**
- `id`: Unique identifier for each example
- `example`: Text content to classify
- `Label`: Classification label

### 4. Run the Pipeline

**Option A: Run Complete Pipeline**
```bash
# Make script executable
chmod +x run_all.sh

# Run all steps
./run_all.sh
```

**Option B: Run Scripts Individually**
```bash
# Step 1: Pattern identification and candidate generation
python 01_data_formatting.py

# Step 2: Generate counterfactuals
python 02_counterfactual_over_generation.py

# Step 3: Filter counterfactuals (3-stage filtering)
python 03_counterfactual_filtering.py

# Step 4: Evaluate with few-shot classification
python 04_counterfactual_evaluation.py
```


## Pipeline Overview

The LLM-VT-AL pipeline consists of four main stages:

```
Input: Training Data (CSV with id, example, Label columns)
        ↓
┌─────────────────────────────────────────────────────┐
│  Script 01: Pattern Identification & Candidates     │
│  • Identifies key phrases in labeled examples       │
│  • Generates alternative phrases for each label     │
│  • Batch processing for efficiency                  │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Script 02: Counterfactual Over-Generation          │
│  • Transforms sentences using candidate phrases     │
│  • Creates counterfactuals for all target labels    │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Script 03: Three-Stage Quality Filtering           │
│  • Filter 1: Remove meta-responses (heuristic)      │
│  • Filter 2: Validate phrase usage (LLM)            │
│  • Filter 3: Verify label transformation (LLM)      │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Script 04: Few-Shot Evaluation                     │
│  • Tests across multiple shot configurations        │
│  • Evaluates entire test dataset                    │
│  • Outputs: Precision, Recall, F1-Score, Accuracy   │
└─────────────────────────────────────────────────────┘
        ↓
Output: Evaluation Metrics & High-Quality Counterfactuals
```

## Project Structure

```
LLM-VT-AL/
├── config.yaml                          # Main configuration file
├── config.yaml.example                  # Configuration template
├── run_all.sh                          # Pipeline automation script
├── requirements.txt                     # Python dependencies
│
├── utils/                               # Utility modules
│   ├── llm_provider.py                  # LLM provider abstraction
│   └── __init__.py                      # Utility exports
│
├── 01_data_formatting.py                # Pattern & candidate generation
├── 02_counterfactual_over_generation.py # Counterfactual generation  
├── 03_counterfactual_filtering.py       # Quality filtering pipeline
├── 04_counterfactual_evaluation.py      # Few-shot evaluation
│
├── input_data/                          # Input datasets
│   ├── emotions_train.csv
│   ├── emotions_test.csv
│   ├── yelp_train.csv
│   ├── yelp_test.csv
│   └── massive_train.csv
│
└── output_data/                         # Generated outputs
    ├── [seed][model]annotated_data_with_pattern_*.csv
    ├── [seed][model]*_candidate_phrases_annotated_data.csv
    ├── [seed][model]counterfactuals_*.csv
    ├── [seed][model]filtered_*.csv
    ├── [seed][model]fine_tuneset_*.csv
    │
    ├── interim_output/                  # Intermediate files
    └── archive/gpt/                     # Evaluation results
        └── [seed][model]_counter_*_prf.csv
```

##  Script Details

### Script 01: Pattern Identification & Candidate Generation

**Purpose:** Identifies key phrases in training examples and generates alternative phrases for counterfactual generation.

**Features:**
- Batch processing (20 examples per LLM call)
- Dataset-agnostic prompting
- Enhanced text normalization for robust parsing
- Configurable sampling per label

**Outputs:**
- `annotated_data_with_pattern_*.csv`: Identified patterns for each example
- `*_candidate_phrases_annotated_data.csv`: Alternative phrases for each target label

**Configuration:**
```yaml
processing:
  max_examples_per_label: 50  # Number of examples to process per label
```

---

### Script 02: Counterfactual Over-Generation

**Purpose:** Generates complete counterfactual sentences by transforming original examples using candidate phrases.

**Features:**
- Transforms each example to all target labels
- Rate limiting to respect API quotas
- Handles API errors gracefully

**Outputs:**
- `counterfactuals_*.csv`: Generated counterfactual sentences

**Configuration:**
```yaml
processing:
  max_counterfactuals_per_example: 4  # Counterfactuals per original example
```

---

### Script 03: Three-Stage Quality Filtering

**Purpose:** Filters generated counterfactuals through a rigorous three-stage pipeline to ensure quality.

**Filtering Stages:**

1. **Heuristic Filter (Rule-based)**
   - Removes LLM meta-responses (e.g., "given the constraints")
   - Fast, no LLM calls required

2. **Semantic Filter (LLM-based)**
   - Validates that counterfactual uses one of the candidate phrases
   - Ensures semantic coherence with original context

3. **Discriminator Filter (LLM-based)**
   - Verifies counterfactual is NOT about original label
   - Confirms counterfactual IS about target label

**Outputs:**
- `filtered_*.csv`: All counterfactuals with filter results (columns: heuristic_filtered, matched_pattern, is_ori, is_target)
- `fine_tuneset_*.csv`: High-quality subset (all filters passed)
- `interim_output/`: Intermediate filtering results

---

### Script 04: Few-Shot Evaluation

**Purpose:** Evaluates the effectiveness of counterfactual augmentation using few-shot classification.

**Features:**
- Tests multiple shot configurations (default: 10, 15, 30, 50, 70, 90, 120)
- Evaluates entire test dataset (no sampling)
- Uses label masking ("concept A/B/C") to prevent label leakage
- Computes comprehensive metrics

**Outputs:**
- `archive/gpt/[seed][model]_counter_*_prf.csv`: Performance metrics

**Output Format:**
```csv
shots,precision,recall,fscore,accuracy,test_size
10,0.52,0.49,0.50,0.54,152
15,0.61,0.58,0.59,0.63,152
30,0.73,0.70,0.71,0.75,152
```

**Configuration:**
```yaml
processing:
  evaluation_shots: [10, 15, 30, 50, 70, 90, 120]
```

## Configuration Guide

### Full Configuration Example

```yaml
# LLM Provider Configuration
llm:
  provider: openai  # Options: openai, gemini, ollama
  
  openai:
    api_key: YOUR_API_KEY
    azure_endpoint: https://your-endpoint.openai.azure.com/
    api_version: "2024-08-01-preview"
    model: gpt-4o
  
  # Model-specific parameters for each pipeline stage
  models:
    pattern_identification:
      temperature: 1.0
      max_tokens: 4096
    
    candidate_generation:
      temperature: 1.0
      max_tokens: 2048
    
    counterfactual_generation:
      temperature: 1.0
      max_tokens: 1024
    
    semantic_filtering:
      temperature: 1.0
      max_tokens: 512
    
    discriminator_filtering:
      temperature: 1.0
      max_tokens: 256

# Dataset Configuration
dataset:
  train_file: emotions_train.csv
  test_file: emotions_test.csv
  
  columns:
    id: id
    text: example
    label: Label
  
  exclude_labels: []  # Labels to skip (e.g., ['neutral', 'other'])

# Processing Configuration
processing:
  seed: 42
  max_examples_per_label: 50
  max_counterfactuals_per_example: 4
  token_limit: 10000000
  evaluation_shots: [10, 15, 30, 50, 70, 90, 120]

# Directory Configuration
directories:
  input_data: input_data
  output_data: output_data
  interim_output: output_data/interim_output
  archive: output_data/archive/gpt
```

### Switching Datasets

To run the pipeline on a different dataset, update `config.yaml`:

```yaml
dataset:
  train_file: yelp_train.csv
  test_file: yelp_test.csv
```

Or for MASSIVE dataset:

```yaml
dataset:
  train_file: massive_train.csv
  test_file: massive_test.csv
```

## Troubleshooting

### Common Issues

**1. Virtual environment not activated**
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**2. Missing API key**
```
ERROR: API key not configured
```
→ Add your API key to `config.yaml`

**3. Input file not found**
```
ERROR: Training file not found: input_data/emotions_train.csv
```
→ Ensure CSV files are in `input_data/` directory

**4. Ollama not running (if using Ollama)**
```
ERROR: Could not connect to Ollama
```
→ Start Ollama: `ollama serve`

**5. API rate limits**
```
429 Error: Too Many Requests
```
→ Scripts include rate limiting; wait for quota to reset or upgrade API tier

**6. Script dependencies**
→ Scripts must run in order: 01 → 02 → 03 → 04

### Validation Commands

```bash
# Check Python version (requires 3.8+)
python --version

# Verify dependencies installed
pip list | grep -E "openai|google-generativeai|pandas|scikit-learn"

# Test configuration loading
python -c "from utils import load_config; print(load_config())"

# Check input data
ls -la input_data/

# Verify Ollama (if using)
curl http://localhost:11434/api/tags
```

## Expected Results

### Emotions Dataset (6 labels, ~180 examples)

**Script 01:**
- Patterns identified
- Candidate phrase sets generated

**Script 02:**
- Raw counterfactuals generated

**Script 03:**
- Heuristic filter pass rate
- Semantic filter pass rate
- Discriminator filter pass rate
- Final high-quality counterfactuals

**Script 04:**
- Evaluation across 7 shot configurations
- Performance typically improves with more shots

## Contributing

Contributions are welcome! Areas for improvement:

1. Additional LLM provider support (Claude, Llama, etc.)
2. Enhanced parsing strategies for different model outputs
3. Cost optimization techniques
4. Additional evaluation metrics
5. Support for multilingual datasets

## License

MIT License - See LICENSE file for details

## Acknowledgments

This enhanced LLM-VT-AL pipeline builds upon counterfactual data augmentation research with improvements in:
- Dataset-agnostic prompting
- Robust text normalization and parsing
- Three-stage quality filtering
- Comprehensive few-shot evaluation

---

**Version:** 2.0 (Enhanced with multi-provider support, improved filtering, and scalable evaluation)