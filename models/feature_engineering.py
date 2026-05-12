"""
Feature engineering for ionic conductivity prediction
- Remove redundant/weak features
- Add interaction features
- Handle upper-bound samples
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import re
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
DEFAULT_TRAIN = FEATURE_DIR / "ionic_26_features_train.csv"
DEFAULT_TEST = FEATURE_DIR / "ionic_26_features_test.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "data"
TARGET_COLUMN = "Ionic conductivity (S cm-1)"
METADATA_COLUMNS = [
    "ID",
    "True Composition",
    "Z_by_element",
    TARGET_COLUMN,
    "conductivity_value",
    "conductivity_used",
    "conductivity_qualifier",
    "log10_conductivity",
    "sample_weight",
]


def parse_conductivity(value):
    """Parse conductivity value and qualifier"""
    if pd.isna(value):
        return math.nan, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), "exact"
    
    text = str(value).strip().replace("−", "-")
    qualifier = "exact"
    if text.startswith(("<", "≤")):
        qualifier = "upper_bound"
    elif text.startswith((">", "≥")):
        qualifier = "lower_bound"
    
    pattern = re.compile(r"^[<>=~≤≥\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    match = pattern.match(text.replace(",", ""))
    if not match:
        return math.nan, "unparsed"
    return float(match.group(1)), qualifier


def feature_set_name(train_path: str | Path) -> str:
    name = Path(train_path).stem
    if name.endswith("_train"):
        name = name[:-len("_train")]
    return name


def engineer_features(
    df,
    remove_upper_bound=False,
    add_interactions=True,
    upper_bound_threshold=1e-10,
    upper_bound_replacement=1e-11,
    upper_bound_weight=0.3,
):
    """
    Engineer features for ionic conductivity prediction
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw feature dataframe
    remove_upper_bound : bool
        If True, remove samples with upper-bound conductivity (<1E-10)
    add_interactions : bool
        If True, add interaction features
    
    Returns:
    --------
    df_processed : pd.DataFrame
        Processed dataframe with engineered features
    feature_cols : list
        List of feature column names
    """
    df = df.copy()
    
    # Parse conductivity
    values = df[TARGET_COLUMN].apply(parse_conductivity)
    df['conductivity_value'] = [item[0] for item in values]
    df['conductivity_qualifier'] = [item[1] for item in values]
    upper_mask = (
        (df['conductivity_qualifier'] == 'upper_bound')
        & (df['conductivity_value'] <= upper_bound_threshold)
    )
    df['conductivity_used'] = df['conductivity_value'].mask(upper_mask, upper_bound_replacement)
    df['sample_weight'] = np.where(upper_mask, upper_bound_weight, 1.0)
    
    # Handle upper-bound samples
    if remove_upper_bound:
        print(f"Removing {(df['conductivity_qualifier'] == 'upper_bound').sum()} upper-bound samples")
        df = df[df['conductivity_qualifier'] == 'exact'].copy()
    
    # Create log10 target
    df['log10_conductivity'] = np.log10(df['conductivity_used'].where(df['conductivity_used'] > 0))
    
    # Get all feature columns
    all_feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    
    # Remove redundant/weak features based on analysis
    redundant_features = [
        'ρ⁺(incl Li⁺) - ρ⁻',      # r=1.000 with ρ⁺(incl Li⁺)
        'r⁺(excl Li⁺) / r⁻',      # r=0.988 with r⁺(excl Li⁺)
        'ρₐₗₗ (C m⁻³)',           # r=0.974 with ρ⁺(incl Li⁺)
        'χ⁺(incl Li⁺)',           # r=-0.006 with target (extremely weak)
        'nₐₙᵢₒₙ',                 # r=0.020 with target
        'nₕₐₗᵢdₑ',                # r=-0.035 with target
    ]
    
    feature_cols = [c for c in all_feature_cols if c not in redundant_features]
    
    # Add interaction features
    if add_interactions:
        # Interaction 1: Ionic radius ratio × Electronegativity difference
        # Physical meaning: Coulombic interaction strength proxy
        if 'r⁺(incl Li⁺) / r⁻' in df.columns and 'χ⁺(incl Li⁺) - χ⁻' in df.columns:
            df['r_ratio_x_chi_diff'] = (
                pd.to_numeric(df['r⁺(incl Li⁺) / r⁻'], errors='coerce') * 
                pd.to_numeric(df['χ⁺(incl Li⁺) - χ⁻'], errors='coerce')
            )
            feature_cols.append('r_ratio_x_chi_diff')
        
        # Interaction 2: Log of charge density ratio (compress extreme values)
        if 'ρ⁺(incl Li⁺) / ρ⁻' in df.columns:
            rho_ratio = pd.to_numeric(df['ρ⁺(incl Li⁺) / ρ⁻'], errors='coerce')
            df['log_rho_ratio'] = np.log10(rho_ratio.clip(lower=1e-3))
            feature_cols.append('log_rho_ratio')
        
        # Interaction 3: Li content × anion radius
        if 'n_Li' in df.columns and 'r⁻ (pm)' in df.columns:
            df['n_Li × r⁻ (pm)'] = (
                pd.to_numeric(df['n_Li'], errors='coerce') * 
                pd.to_numeric(df['r⁻ (pm)'], errors='coerce')
            )
            feature_cols.append('n_Li × r⁻ (pm)')
        
        # Interaction 4: Electronegativity range × average radius
        if 'χₘₐₓ - χₘᵢₙ' in df.columns and 'rₐₗₗ (pm)' in df.columns:
            df['chi_range_x_r_avg'] = (
                pd.to_numeric(df['χₘₐₓ - χₘᵢₙ'], errors='coerce') * 
                pd.to_numeric(df['rₐₗₗ (pm)'], errors='coerce')
            )
            feature_cols.append('chi_range_x_r_avg')
        
        # Interaction 5: Field strength × radius difference
        if 'Φ⁺(incl Li⁺) (|Z| pm⁻¹)' in df.columns and 'r⁺(incl Li⁺) - r⁻' in df.columns:
            df['field_x_r_diff'] = (
                pd.to_numeric(df['Φ⁺(incl Li⁺) (|Z| pm⁻¹)'], errors='coerce') * 
                pd.to_numeric(df['r⁺(incl Li⁺) - r⁻'], errors='coerce')
            )
            feature_cols.append('field_x_r_diff')
        
        print(f"Added {len([c for c in feature_cols if c not in all_feature_cols])} interaction features")
    
    print(f"Total features: {len(feature_cols)} (removed {len(redundant_features)} redundant)")
    
    return df, feature_cols


def prepare_data(
    train_path=DEFAULT_TRAIN,
    test_path=DEFAULT_TEST,
    output_dir=DEFAULT_OUTPUT_DIR,
    remove_upper_bound=False,
    add_interactions=True,
    upper_bound_threshold=1e-10,
    upper_bound_replacement=1e-11,
    upper_bound_weight=0.3,
    save_processed=True,
):
    """
    Prepare train and test data with feature engineering
    
    Returns:
    --------
    dict with keys: X_train, y_train, X_test, y_test, feature_cols, train_ids, test_ids
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Raw train samples: {len(train_raw)}")
    print(f"Raw test samples: {len(test_raw)}")
    
    # Engineer features
    print("\n--- Processing training set ---")
    train_processed, feature_cols = engineer_features(
        train_raw, 
        remove_upper_bound=remove_upper_bound,
        add_interactions=add_interactions,
        upper_bound_threshold=upper_bound_threshold,
        upper_bound_replacement=upper_bound_replacement,
        upper_bound_weight=upper_bound_weight,
    )
    
    print("\n--- Processing test set ---")
    test_processed, _ = engineer_features(
        test_raw,
        remove_upper_bound=remove_upper_bound,
        add_interactions=add_interactions,
        upper_bound_threshold=upper_bound_threshold,
        upper_bound_replacement=upper_bound_replacement,
        upper_bound_weight=upper_bound_weight,
    )
    
    print(f"\nProcessed train samples: {len(train_processed)}")
    print(f"Processed test samples: {len(test_processed)}")
    
    # Extract features and target
    X_train = train_processed[feature_cols].apply(pd.to_numeric, errors='coerce')
    y_train = train_processed['log10_conductivity']
    w_train = train_processed['sample_weight']
    train_ids = train_processed['ID']
    
    X_test = test_processed[feature_cols].apply(pd.to_numeric, errors='coerce')
    y_test = test_processed['log10_conductivity']
    w_test = test_processed['sample_weight']
    test_ids = test_processed['ID']
    
    # Handle inf values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Check for missing values
    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"\nWarning: Missing values - train: {train_missing}, test: {test_missing}")
        print("Filling with column medians...")
        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    if save_processed:
        train_processed.to_csv(output_dir / 'train_processed.csv', index=False)
        test_processed.to_csv(output_dir / 'test_processed.csv', index=False)
        X_train.to_csv(output_dir / 'X_train.csv', index=False)
        X_test.to_csv(output_dir / 'X_test.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False, header=['log10_conductivity'])
        y_test.to_csv(output_dir / 'y_test.csv', index=False, header=['log10_conductivity'])
        w_train.to_csv(output_dir / 'w_train.csv', index=False, header=['sample_weight'])
        w_test.to_csv(output_dir / 'w_test.csv', index=False, header=['sample_weight'])
        with open(output_dir / 'feature_list.txt', 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        print(f"\nProcessed data saved to {output_dir}")
    print(f"Feature columns: {len(feature_cols)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'w_train': w_train,
        'X_test': X_test,
        'y_test': y_test,
        'w_test': w_test,
        'feature_cols': feature_cols,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'train_processed': train_processed,
        'test_processed': test_processed,
        'feature_set': feature_set_name(train_path),
    }


if __name__ == '__main__':
    # Test feature engineering
    data = prepare_data(
        train_path=DEFAULT_TRAIN,
        test_path=DEFAULT_TEST,
        output_dir=DEFAULT_OUTPUT_DIR,
        remove_upper_bound=False,
        add_interactions=True
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_train range: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
    print(f"y_test range: [{data['y_test'].min():.2f}, {data['y_test'].max():.2f}]")
