#!/usr/bin/env python3
"""
Script 03: Counterfactual Filtering (LLM-Based)

Filter counterfactual sentences generated in Script 02 to ensure high quality.
Uses three-stage filtering pipeline:
1. Heuristic filtering (remove meta-responses)
2. LLM semantic filtering (validate phrase usage & context)
3. Discriminator filtering (validate label transformation)

Input: output_data/[{SEED}]counterfactuals_{dataset}.csv
Output: 
- output_data/[{seed}]filtered_{dataset}.csv (complete with filter results)
- output_data/[{seed}]fine_tuneset_{dataset}.csv (high-quality subset)
"""

import sys
import pandas as pd
import ast
from utils import (
    load_config,
    ensure_directories,
    get_llm_provider
)


def heuristic_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter 1: Remove GPT meta-responses using heuristic rules.
    
    Marks as INVALID if counterfactual contains both:
    - "given the constraints" (case-insensitive)
    - "the assistant" (case-insensitive)
    
    Args:
        df: DataFrame with counterfactuals
        
    Returns:
        DataFrame with 'heuristic_filtered' column added
    """
    print("\n--- Filter 1: Heuristic Filtering ---")
    
    def is_valid(text):
        """Check if counterfactual is valid (not a meta-response)"""
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return not ("given the constraints" in text_lower and "the assistant" in text_lower)
    
    df['heuristic_filtered'] = df['counterfactual'].apply(is_valid)
    
    passed = df['heuristic_filtered'].sum()
    total = len(df)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"  Passed: {passed}/{total} ({pass_rate:.2%})")
    
    return df


def llm_semantic_filtering(df: pd.DataFrame, config: dict, llm_provider) -> pd.DataFrame:
    """
    Filter 2: LLM-based semantic validation.
    
    Verifies that counterfactual:
    1. Uses one of the candidate phrases (not reworded)
    2. Maintains semantic context from original
    
    Args:
        df: DataFrame with counterfactuals
        config: Configuration dictionary
        llm_provider: LLM provider instance
        
    Returns:
        DataFrame with 'matched_pattern' column added
    """
    print("\n--- Filter 2: LLM Semantic Filtering ---")
    
    llm_config = config['llm']['models']['semantic_filtering']
    
    # Initialize matched_pattern column
    df['matched_pattern'] = None
    
    # Only process rows that passed heuristic filter
    valid_indices = df[df['heuristic_filtered'] == True].index
    
    print(f"  Processing {len(valid_indices)} counterfactuals...")
    
    for idx, row_idx in enumerate(valid_indices):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(valid_indices)}...")
        
        row = df.loc[row_idx]
        
        # Show progress for each item
        print(f"    [{idx+1}/{len(valid_indices)}] Validating {row['id']}: {row['ori_label']} → {row['target_label']}...", end=' ', flush=True)
        
        ori_text = row['ori_text']
        highlight = row['highlight']
        counterfactual = row['counterfactual']
        
        # Parse candidate phrases
        try:
            candidate_phrases = ast.literal_eval(row['candidate_phrases'])
        except:
            candidate_phrases = row['candidate_phrases']
        
        # Construct validation prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert at validating text transformations."
            },
            {
                "role": "user",
                "content": f"""Validate the following transformation:

Original sentence: "{ori_text}"
Original phrase: "{highlight}"
Candidate replacement phrases: {candidate_phrases}
Modified sentence: "{counterfactual}"

Check TWO criteria:
1. Does the modified sentence use ONE of the candidate phrases (not rephrased)?
2. Does the modified sentence maintain similar topic as the original?

Answer with only 'YES' if BOTH criteria are met, otherwise 'NO'."""
            }
        ]
        
        # Call LLM
        try:
            response = llm_provider.chat_completion(
                messages=messages,
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            
            # Parse response
            response_clean = response.strip().upper()
            is_valid = 'YES' in response_clean
            
            df.at[row_idx, 'matched_pattern'] = is_valid
            
            print('✓' if is_valid else '✗')
            
        except Exception as e:
            print(f"Error: {e}")
            df.at[row_idx, 'matched_pattern'] = False
    
    # Calculate statistics
    passed = df['matched_pattern'].sum()
    total = len(valid_indices)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"\n  Passed: {passed}/{total} ({pass_rate:.2%})")
    print(f"  The pattern keeping rate is: {pass_rate:.2f}")
    
    return df


def gpt_discriminator_filtering(df: pd.DataFrame, config: dict, llm_provider) -> pd.DataFrame:
    """
    Filter 3: GPT discriminator validation.
    
    Validates that counterfactual:
    1. NOT about original label anymore
    2. IS about target label
    
    Args:
        df: DataFrame with counterfactuals
        config: Configuration dictionary
        llm_provider: LLM provider instance
        
    Returns:
        DataFrame with 'is_ori' and 'is_target' columns added
    """
    print("\n--- Filter 3: Discriminator Filtering ---")
    
    llm_config = config['llm']['models']['discriminator_filtering']
    
    # Initialize columns
    df['is_ori'] = None
    df['is_target'] = None
    
    # Only process rows that passed previous filters
    valid_indices = df[
        (df['heuristic_filtered'] == True) & 
        (df['matched_pattern'] == True)
    ].index
    
    print(f"  Processing {len(valid_indices)} counterfactuals...")
    
    for idx, row_idx in enumerate(valid_indices):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(valid_indices)}...")
        
        row = df.loc[row_idx]
        
        # Show progress for each item
        print(f"    [{idx+1}/{len(valid_indices)}] Classifying {row['id']}: {row['ori_label']} → {row['target_label']}...", end=' ', flush=True)
        
        counterfactual = row['counterfactual']
        ori_label = row['ori_label']
        target_label = row['target_label']
        
        # Construct classification prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert at text classification."
            },
            {
                "role": "user",
                "content": f"""Classify this sentence for two categories:

Sentence: "{counterfactual}"

Question 1: Does this sentence belong to category '{ori_label}'?
Question 2: Does this sentence belong to category '{target_label}'?

Answer in format: ANSWER1, ANSWER2
Where each answer is either YES or NO.

Example: NO, YES"""
            }
        ]
        
        # Call LLM
        try:
            response = llm_provider.chat_completion(
                messages=messages,
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            
            # Parse response (format: "NO, YES" or "YES, NO")
            parts = response.strip().upper().replace(',', ' ').split()
            
            if len(parts) >= 2:
                is_original = 'YES' in parts[0]
                is_target = 'YES' in parts[-1]
            else:
                is_original = True  # Conservative: mark as failed
                is_target = False
            
            df.at[row_idx, 'is_ori'] = is_original
            df.at[row_idx, 'is_target'] = is_target
            
            # Show result
            result = "✓" if (not is_original and is_target) else ("~" if (not is_original or is_target) else "✗")
            print(result)
            
        except Exception as e:
            print(f"Error: {e}")
            df.at[row_idx, 'is_ori'] = True
            df.at[row_idx, 'is_target'] = False
    
    # Calculate statistics
    not_ori = (df['is_ori'] == False).sum()
    is_target = (df['is_target'] == True).sum()
    both_pass = ((df['is_ori'] == False) & (df['is_target'] == True)).sum()
    total = len(valid_indices)
    
    not_ori_rate = not_ori / total if total > 0 else 0
    is_target_rate = is_target / total if total > 0 else 0
    
    print(f"\n  Not original label: {not_ori}/{total} ({not_ori_rate:.2%})")
    print(f"  Is target label: {is_target}/{total} ({is_target_rate:.2%})")
    print(f"  Passed both: {both_pass}/{total}")
    print(f"\n  The percentage of is not original label: {not_ori_rate:.2f}")
    print(f"  The percentage of is the target counterfactual label: {is_target_rate:.2f}")
    
    return df


def create_fine_tune_dataset(df: pd.DataFrame, config: dict):
    """
    Create fine-tuning dataset from filtered counterfactuals.
    
    Only includes counterfactuals that passed all three filters.
    
    Args:
        df: DataFrame with filter results
        config: Configuration dictionary
    """
    print("\n--- Creating Fine-Tune Dataset ---")
    
    # Filter for high-quality counterfactuals
    df_finetune = df[
        (df['heuristic_filtered'] == True) &
        (df['matched_pattern'] == True) &
        (df['is_ori'] == False) &
        (df['is_target'] == True)
    ].copy()
    
    # Select and rename columns for fine-tuning
    df_finetune = df_finetune[[
        'id', 'ori_text', 'ori_label', 'pattern', 
        'counterfactual', 'target_label'
    ]].rename(columns={
        'counterfactual': 'counter_text',
        'target_label': 'counter_label'
    })
    
    print(f"  High-quality counterfactuals: {len(df_finetune)}")
    
    # Save fine-tuning dataset
    dirs = config['directories']
    seed = config['processing']['seed']
    dataset_file = config['dataset']['train_file']
    
    output_file = f"{dirs['output_data']}/[{seed}]fine_tuneset_{dataset_file}"
    df_finetune.to_csv(output_file, index=False)
    
    print(f"  Saved to: {output_file}")
    print(f"\n  The number of finetune dataset is: {len(df_finetune)}")


def filter_counterfactuals(config: dict, llm_provider):
    """
    Run three-stage filtering pipeline.
    
    Args:
        config: Configuration dictionary
        llm_provider: LLM provider instance
    """
    print("\n=== Starting Counterfactual Filtering ===")
    
    # Load configuration
    dirs = config['directories']
    seed = config['processing']['seed']
    dataset_file = config['dataset']['train_file']
    
    # Load counterfactuals from Script 02
    input_file = f"{dirs['output_data']}/[{seed}]counterfactuals_{dataset_file}"
    
    try:
        df = pd.read_csv(input_file)
        print(f"\nINFO: Loaded {len(df)} counterfactuals from {input_file}")
    except FileNotFoundError:
        print(f"ERROR: Counterfactual file not found: {input_file}")
        print("Please run Script 02 (02_counterfactual_over_generation.py) first")
        sys.exit(1)
    
    # Run three-stage filtering
    df = heuristic_filtering(df)
    
    # Save interim output
    interim_file = f"{dirs['interim_output']}/[{seed}]heuristic_filtered_{dataset_file}"
    df.to_csv(interim_file, index=False)
    print(f"  Interim saved to: {interim_file}")
    
    df = llm_semantic_filtering(df, config, llm_provider)
    
    # Save interim output
    interim_file = f"{dirs['interim_output']}/[{seed}]semantic_filtered_{dataset_file}"
    df.to_csv(interim_file, index=False)
    print(f"  Interim saved to: {interim_file}")
    
    df = gpt_discriminator_filtering(df, config, llm_provider)
    
    # Save complete filtered dataset
    output_file = f"{dirs['output_data']}/[{seed}]filtered_{dataset_file}"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Complete filtered dataset saved to: {output_file}")
    
    # Create fine-tuning dataset
    create_fine_tune_dataset(df, config)


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("Script 03: Counterfactual Filtering")
    print("="*60)
    
    # Load configuration
    config = load_config()
    ensure_directories(config)
    
    # Initialize LLM provider
    print(f"\nINFO: Using LLM provider: {config['llm']['provider']}")
    llm_provider = get_llm_provider(config)
    
    # Run filtering pipeline
    filter_counterfactuals(config, llm_provider)
    
    print("\n" + "="*60)
    print("Script 03 Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
