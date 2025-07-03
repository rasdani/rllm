#!/usr/bin/env python3
"""
Prepare minimal SWE dataset for Qwen3-8B training.
This script creates a subset of the full datasets for resource-constrained reproduction.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm

# Add rllm to path
sys.path.append(str(Path(__file__).parent))

from rllm.data.dataset import DatasetRegistry

# Configuration
SUBSET_SIZE = {
    "train": 100,  # Reduced from full dataset
    "val": 20,     # Reduced from full dataset
}

OUTPUT_DIR = Path("./data/swe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_row(row):
    """Process a single row from the dataset."""
    row_dict = dict(row)
    
    # Ensure required fields exist
    required_fields = ["instance_id", "problem_statement", "base_commit"]
    for field in required_fields:
        if field not in row_dict:
            print(f"Warning: Missing field {field} in row")
    
    return row_dict

def prepare_minimal_datasets():
    """Prepare minimal train and validation datasets."""
    
    print("Loading R2E-Gym-Subset for training...")
    try:
        # Load minimal training subset
        train_dataset = load_dataset(
            "R2E-Gym/R2E-Gym-Subset", 
            split=f"train[:{SUBSET_SIZE['train']}]"
        )
        print(f"Loaded {len(train_dataset)} training examples")
        
        # Process training data
        train_data = []
        for row in tqdm(train_dataset, desc="Processing training data"):
            processed = process_row(row)
            train_data.append(processed)
        
        # Save as parquet
        train_df = pd.DataFrame(train_data)
        train_path = OUTPUT_DIR / "R2E_Gym_Subset_minimal.parquet"
        train_df.to_parquet(train_path)
        print(f"Saved training data to {train_path}")
        
        # Register with DatasetRegistry
        DatasetRegistry.register_dataset("R2E_Gym_Subset_minimal", train_data, "train")
        
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return None, None
    
    print("\nLoading SWE-Bench-Verified for validation...")
    try:
        # Load minimal validation subset
        val_dataset = load_dataset(
            "R2E-Gym/SWE-Bench-Verified", 
            split=f"test[:{SUBSET_SIZE['val']}]"
        )
        print(f"Loaded {len(val_dataset)} validation examples")
        
        # Process validation data
        val_data = []
        for row in tqdm(val_dataset, desc="Processing validation data"):
            processed = process_row(row)
            val_data.append(processed)
        
        # Save as parquet
        val_df = pd.DataFrame(val_data)
        val_path = OUTPUT_DIR / "SWE_Bench_Verified_minimal.parquet"
        val_df.to_parquet(val_path)
        print(f"Saved validation data to {val_path}")
        
        # Register with DatasetRegistry
        DatasetRegistry.register_dataset("SWE_Bench_Verified_minimal", val_data, "test")
        
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        return train_data, None
    
    return train_data, val_data

def create_dataset_info():
    """Create a JSON file with dataset information."""
    info = {
        "train": {
            "dataset": "R2E-Gym/R2E-Gym-Subset",
            "size": SUBSET_SIZE["train"],
            "path": str(OUTPUT_DIR / "R2E_Gym_Subset_minimal.parquet")
        },
        "val": {
            "dataset": "R2E-Gym/SWE-Bench-Verified",
            "size": SUBSET_SIZE["val"],
            "path": str(OUTPUT_DIR / "SWE_Bench_Verified_minimal.parquet")
        }
    }
    
    info_path = OUTPUT_DIR / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDataset info saved to {info_path}")

if __name__ == "__main__":
    print("Preparing minimal SWE datasets for Qwen3-8B training...")
    print(f"Train size: {SUBSET_SIZE['train']}, Val size: {SUBSET_SIZE['val']}")
    
    train_data, val_data = prepare_minimal_datasets()
    
    if train_data and val_data:
        create_dataset_info()
        print("\nDataset preparation complete!")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        
        # Print sample
        print("\nSample training example:")
        print(f"Instance ID: {train_data[0].get('instance_id', 'N/A')}")
        print(f"Problem statement preview: {train_data[0].get('problem_statement', '')[:200]}...")
    else:
        print("\nError: Dataset preparation failed!")