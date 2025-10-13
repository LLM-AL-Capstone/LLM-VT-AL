# Output Data Directory

This directory contains all generated outputs from the pipeline.

## Output Files

### From Script 01 (Data Formatting)
- `annotated_data_with_pattern_{dataset}.csv` - Key phrases identified per example
- `[{seed}]{dataset}_candidate_phrases_annotated_data.csv` - Alternative phrases for transformation

### From Script 02 (Counterfactual Generation)
- `[{seed}]counterfactuals_{dataset}.csv` - Generated counterfactual sentences

### From Script 03 (Filtering)
- `[{seed}]filtered_{dataset}.csv` - Complete dataset with filter results
- `[{seed}]fine_tuneset_{dataset}.csv` - **High-quality counterfactuals ready for training**

### Interim Outputs (`interim_output/`)
- `[{seed}]heuristic_filtered_{dataset}.csv` - After Filter 1
- `[{seed}]semantic_filtered_{dataset}.csv` - After Filter 2

### Evaluation Results (`archive/gpt/`)
- `[{seed}][GPT]_counter_{dataset}_prf.csv` - Precision/Recall/F-Score metrics per shot size

## Directory Structure

```
output_data/
├── annotated_data_with_pattern_*.csv
├── [seed]*_candidate_phrases_annotated_data.csv
├── [seed]counterfactuals_*.csv
├── [seed]filtered_*.csv
├── [seed]fine_tuneset_*.csv          ← Use this for training!
│
├── interim_output/
│   ├── [seed]heuristic_filtered_*.csv
│   └── [seed]semantic_filtered_*.csv
│
└── archive/
    └── gpt/
        └── [seed][GPT]_counter_*_prf.csv
```

## Using the Fine-Tune Dataset

The most important output is `[{seed}]fine_tuneset_{dataset}.csv`. This contains:
- Original examples
- High-quality counterfactual examples
- Both with their respective labels

You can use this to:
1. Augment your training set
2. Fine-tune classification models
3. Improve model robustness

Example usage:
```python
import pandas as pd

# Load fine-tune set
df_finetune = pd.read_csv('output_data/[42]fine_tuneset_emotions_train.csv')

# Combine with original training data
df_train = pd.read_csv('input_data/emotions_train.csv')

# Use both for training
df_augmented = pd.concat([df_train, df_finetune[['counter_text', 'counter_label']].rename(
    columns={'counter_text': 'example', 'counter_label': 'Label'}
)])
```
