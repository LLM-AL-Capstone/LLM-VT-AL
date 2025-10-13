#!/usr/bin/env python3
"""
Script 02: Counterfactual Over-Generation

This script generates complete counterfactual sentences using LLM.
It reads candidate phrases from Script 01 and creates full sentence transformations.

Input: output_data/[{SEED}]{dataset}_candidate_phrases_annotated_data.csv
Output: output_data/[{seed}]counterfactuals_{dataset}.csv
"""

import sys
import pandas as pd
import ast
from utils import (
    load_config,
    ensure_directories,
    get_llm_provider
)


def generate_counterfactuals(config: dict, llm_provider):
    """
    Generate complete counterfactual sentences.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Counterfactual Over-Generation ===\n")
    
    # Load configuration
    dirs = config['directories']
    dataset_config = config['dataset']
    processing = config['processing']
    llm_config = config['llm']['models']['counterfactual_generation']
    
    seed = processing['seed']
    dataset_name = dataset_config['train_file'].replace('.csv', '')
    
    # Construct input filename
    candidate_file = f"[{seed}]{dataset_name}_candidate_phrases_annotated_data.csv"
    input_path = f"{dirs['output_data']}/{candidate_file}"
    
    # Load candidate phrases from Script 01
    try:
        df = pd.read_csv(input_path)
        print(f"INFO: Loaded {len(df)} candidate phrase sets")
    except FileNotFoundError:
        print(f"ERROR: Candidate file not found: {input_path}")
        print("Please run Script 01 (01_data_formatting.py) first")
        sys.exit(1)
    
    # Define output schema
    col_names = [
        "id",
        "ori_text",
        "ori_label",
        "pattern",
        "highlight",
        "candidate_phrases",
        "target_label",
        "counterfactual"
    ]
    
    # Initialize data collector
    data_collector = []
    
    print(f"INFO: Generating counterfactuals...\n")
    
    # Iterate over each row and generate counterfactuals
    for index, row in df.iterrows():
        if (index + 1) % 50 == 0:
            print(f"Processing {index+1}/{len(df)}...")
        
        text = row["ori_text"]
        label = row["ori_label"]
        target_label = row["target_label"]
        highlight = row["highlight"]
        pattern = row["pattern"]
        
        # Show progress for each item
        print(f"  [{index+1}/{len(df)}] Processing {row['id']}: {label} → {target_label}...", end=' ', flush=True)
        
        # Parse candidate phrases
        try:
            generated_phrases = ast.literal_eval(row["candidate_phrases"])
        except:
            generated_phrases = row["candidate_phrases"]
        
        # Construct prompt messages
        messages = [
            {
                "role": "system",
                "content": "The assistant will create a counterfactual example close to the original sentence that contains one of the given phrases."
            },
            {
                "role": "user",
                "content": f"""Task: Transform the sentence to express a different category.

Instructions:
1. Use ONE phrase from the 'generated phrases' list (exactly as written, no rewording)
2. Change the sentence from '{label}' to '{target_label}' category
3. The modified sentence should NOT also express '{label}'
4. Do NOT use the word '{target_label}' in the sentence (avoid label leakage)
5. Keep the sentence grammatically correct and natural

Data:
- Original text: {text}
- Original label: {label}
- Target label: {target_label}
- Generated phrases: {generated_phrases}

Modified text:"""
            }
        ]
        
        # Call LLM
        try:
            response = llm_provider.chat_completion(
                messages=messages,
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                stop=llm_config.get('stop', ["\n"])
            )
            
            # Clean response (remove quotes if present)
            counterfactual = response.strip().strip('"\'')
            
            print(f"✓")  # Success indicator
            
            # Store result
            data_collector.append([
                row["id"],
                row["ori_text"],
                row["ori_label"],
                row["pattern"],
                row["highlight"],
                row["candidate_phrases"],
                row["target_label"],
                counterfactual
            ])
            
        except Exception as e:
            print(f"✗ (Error: {str(e)[:50]})")  # Show error
            # Store with empty counterfactual on error
            data_collector.append([
                row["id"],
                row["ori_text"],
                row["ori_label"],
                row["pattern"],
                row["highlight"],
                row["candidate_phrases"],
                row["target_label"],
                ""
            ])
    
    # Save output
    df2 = pd.DataFrame(data_collector, columns=col_names)
    output_file = f"{dirs['output_data']}/[{seed}]counterfactuals_{dataset_config['train_file']}"
    df2.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS: Generated {len(df2)} counterfactuals")
    print(f"  Output saved to: {output_file}\n")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 02: Counterfactual Over-Generation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Generate counterfactuals
    generate_counterfactuals(config, llm_provider)
    
    print("="*60)
    print("Script 02 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
