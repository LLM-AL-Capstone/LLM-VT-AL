#!/usr/bin/env python3
"""
Script 01: Data Formatting for LLM-Based Counterfactual Generation

This script:
1. Identifies key emotional phrases in labeled text using LLM
2. Generates alternative phrases to transform emotions

Output:
- annotated_data_with_pattern_{dataset}.csv: Key phrases per example
- [{seed}]{dataset}_candidate_phrases_annotated_data.csv: Alternative phrases per target emotion
"""

import sys
import pandas as pd
import json
import ast
from typing import List, Dict
from utils import (
    load_config, 
    ensure_directories,
    load_dataset,
    get_unique_labels,
    shuffle_dataframe,
    get_llm_provider
)


def get_llm_patterns(config: dict, llm_provider):
    """
    Identify key emotional phrases using LLM.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Pattern Identification ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['pattern_identification']
    
    seed = processing['seed']
    max_examples_per_label = processing.get('max_examples_per_label', 50)  # Enhanced: More examples per label
    
    # Load dataset
    file_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    df = load_dataset(file_path, config)
    
    # Shuffle with seed
    df = shuffle_dataframe(df, seed)
    
    # Get column names
    col_id = dataset_config['columns']['id']
    col_text = dataset_config['columns']['text']
    col_label = dataset_config['columns']['label']
    
    # Enhanced: Process ALL available examples, not limited to 150
    # Only limit by max_examples_per_label for computational efficiency
    selected_df = df
    
    # Get unique labels (excluding specified labels)
    unique_labels = get_unique_labels(
        df, 
        col_label, 
        dataset_config.get('exclude_labels', [])
    )
    
    # Filter out null/NaN values and 'none' (case-insensitive)
    unique_labels = [
        label for label in unique_labels 
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    print(f"INFO: Processing full dataset with {len(selected_df)} examples")
    print(f"INFO: Unique labels: {unique_labels}\n")
    
    # Output structure
    col_names = ["id", "ori_text", "ori_label", "pattern", "highlight"]
    data_collector = []
    
    # Process each label
    for label in unique_labels:
        print(f"Processing label: {label}")
        
        # Filter examples for this label
        label_df = selected_df[selected_df[col_label] == label]
        
        # Enhanced: Take up to max_examples_per_label (default 50) examples per label
        num_samples = min(max_examples_per_label, len(label_df))
        sample_examples = label_df.head(num_samples)
        
        if len(sample_examples) == 0:
            print(f"  ⚠ WARNING: No examples found for label '{label}', skipping")
            continue
        
        print(f"  INFO: Processing {len(sample_examples)} examples for '{label}'")
        
        # Enhanced: Process in batches for better API efficiency
        batch_size = 20  # Process 20 examples per API call
        
        for batch_start in range(0, len(sample_examples), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_examples))
            batch_examples = sample_examples.iloc[batch_start:batch_end]
            
            print(f"    Processing batch {batch_start//batch_size + 1}: examples {batch_start+1}-{batch_end}")
            
            # Create examples text for prompt
            examples_text = "\n".join([
                f"- {row[col_text]}" 
                for _, row in batch_examples.iterrows()
            ])
        
            # Construct prompt messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at identifying key phrases in text that express specific concepts or categories."
                },
                {
                    "role": "user",
                    "content": f"""Analyze these sentences labeled as '{label}':

{examples_text}

For each sentence, identify 1-3 key phrases that most strongly express or indicate the '{label}' category.

Return a JSON array with this format:
[
  {{
    "sentence": "the original sentence",
    "key_phrases": ["phrase1", "phrase2"]
  }}
]"""
                }
            ]
            
            # Call LLM
            try:
                result = llm_provider.chat_completion(
                    messages=messages,
                    temperature=llm_config['temperature'],
                    max_tokens=llm_config['max_tokens']
                )
                
                # Parse JSON response
                # Try to extract JSON from response (handle markdown code blocks and extra text)
                result_clean = result.strip()
                if result_clean.startswith('```'):
                    # Remove markdown code blocks
                    lines = result_clean.split('\n')
                    result_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_clean
                    result_clean = result_clean.replace('```json', '').replace('```', '').strip()
                
                # Try to extract just the JSON array if there's extra text
                try:
                    analyses = json.loads(result_clean)
                except json.JSONDecodeError:
                    # Try to find the JSON array in the response
                    import re
                    # More aggressive regex to find JSON-like structures
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', result_clean, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        print(f"    INFO: Found JSON-like structure with regex, attempting to parse.")
                        analyses = json.loads(json_str)
                    else:
                        print(f"    ERROR: Could not find a valid JSON array in the response.")
                        raise
                
                # Process each analysis
                for analysis in analyses:
                    sentence = analysis.get('sentence', '')
                    key_phrases = analysis.get('key_phrases', [])
                    
                    # Find matching row in dataframe
                    matching_rows = batch_examples[batch_examples[col_text] == sentence]
                    
                    if len(matching_rows) > 0:
                        row = matching_rows.iloc[0]
                        
                        # Format pattern and highlight
                        pattern = ", ".join(key_phrases)
                        highlight = str([[[phrase]] for phrase in key_phrases])
                        
                        data_collector.append([
                            row[col_id],
                            row[col_text],
                            row[col_label],
                            pattern,
                            highlight
                        ])
                
                print(f"    ✓ Processed {len(analyses)} examples in batch")
                
            except Exception as e:
                print(f"    ERROR: Failed to process batch for '{label}': {e}")
                continue
    
    # Create DataFrame and save
    df_patterns = pd.DataFrame(data_collector, columns=col_names)
    
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    output_file = f"{dirs['output_data']}/annotated_data_with_pattern_{dataset_name}.csv"
    df_patterns.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS: Saved {len(df_patterns)} annotated examples to:")
    print(f"  {output_file}\n")


def get_candidate_phrases(config: dict, llm_provider):
    """
    Generate alternative phrases for label transformation.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Candidate Phrase Generation ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['candidate_generation']
    
    seed = processing['seed']
    token_limit = processing['token_limit']
    
    # Load original dataset for unique labels
    file_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    df_original = load_dataset(file_path, config)
    
    col_label = dataset_config['columns']['label']
    unique_labels = get_unique_labels(
        df_original,
        col_label,
        dataset_config.get('exclude_labels', [])
    )
    
    # Filter out null/NaN values and 'none' (case-insensitive)
    unique_labels = [
        label for label in unique_labels 
        if pd.notna(label) and str(label).lower() not in ['none', 'null', '', 'nan']
    ]
    
    # Load annotated patterns from previous step
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    pattern_file = f"{dirs['output_data']}/annotated_data_with_pattern_{dataset_name}.csv"
    
    try:
        df = pd.read_csv(pattern_file)
        print(f"INFO: Loaded {len(df)} annotated patterns")
    except FileNotFoundError:
        print(f"ERROR: Pattern file not found: {pattern_file}")
        print("Please run get_llm_patterns() first")
        sys.exit(1)
    
    # Initialize token counter
    num_tokens = 0
    processed_examples = 0
    
    # Initialize output structure
    col_names_2 = [
        "id", "ori_text", "ori_label", "pattern", 
        "highlight", "target_label", "candidate_phrases"
    ]
    data_collector_2 = []
    
    print(f"INFO: Generating candidates for {len(unique_labels)} target labels")
    print(f"INFO: Processing {len(df)} annotated examples\n")
    
    # Iterate through annotated examples
    for i, row in df.iterrows():
        # Check token budget
        if num_tokens > token_limit:
            print(f"WARNING: Token limit ({token_limit:,}) reached. Stopping.")
            print(f"Processed {processed_examples} examples out of {len(df)}")
            break
        
        # Progress logging for large datasets
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(df)} examples processed")
            print(f"  Tokens used so far: {num_tokens:,}")
        
        print(f"Processing example {i+1}/{len(df)}: {row['id']}")
        
        sentence = row['ori_text']
        label = row['ori_label']
        
        # Parse highlight to get individual phrases
        try:
            highlight = ast.literal_eval(row['highlight'])
            marked_phrases = [phrase[0][0] for phrase in highlight]
        except:
            print(f"  ⚠ Warning: Could not parse highlight for {row['id']}")
            continue
        
        # For each matched phrase
        for matched_phrase in marked_phrases:
            # For each target label (excluding original)
            for target_label in unique_labels:
                if target_label == label:
                    continue
                
                # Construct prompt
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert at generating alternative phrases for text transformation."
                    },
                    {
                        "role": "user",
                        "content": f"""Sentence: "{sentence}"

Key phrase: "{matched_phrase}"

Generate 5-7 alternative phrases that could replace "{matched_phrase}" to express '{target_label}' instead of '{label}'.

Requirements:
- Maintain sentence structure
- Keep the context
- Only change the content to reflect '{target_label}'
- Be natural and diverse

Provide comma-separated alternatives."""
                    }
                ]
                
                # Call LLM
                try:
                    result = llm_provider.chat_completion(
                        messages=messages,
                        temperature=llm_config['temperature'],
                        max_tokens=llm_config['max_tokens']
                    )
                    
                    # Count tokens (approximate)
                    prompt_text = " ".join([m['content'] for m in messages])
                    num_tokens += llm_provider.count_tokens(prompt_text + result)
                    
                    # Parse comma-separated phrases
                    candidates = [p.strip() for p in result.split(',')]
                    
                    # Store result
                    data_collector_2.append([
                        row['id'],
                        row['ori_text'],
                        row['ori_label'],
                        row['pattern'],
                        matched_phrase,
                        target_label,
                        str(candidates)
                    ])
                    
                    processed_examples += 1
                    
                except Exception as e:
                    print(f"  ⚠ Error generating candidates: {e}")
                    continue
        
        if (i + 1) % 10 == 0:
            print(f"  Tokens used so far: {num_tokens:,}")
    
    # Create DataFrame and save
    df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
    output_file = f"{dirs['output_data']}/[{seed}]{dataset_name}_candidate_phrases_annotated_data.csv"
    df2.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS: Saved {len(df2)} candidate phrase sets to:")
    print(f"  {output_file}")
    print(f"  Total tokens used: {num_tokens:,}")
    print(f"  Processed examples: {processed_examples}")
    print(f"  Expected improvement: {len(df2)} vs ~800 candidate sets in original approach\n")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 01: Data Formatting")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Run both steps
    get_llm_patterns(config, llm_provider)
    get_candidate_phrases(config, llm_provider)
    
    print("="*60)
    print("Script 01 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
