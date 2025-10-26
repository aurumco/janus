"""Dataset inspection and health check tool."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def inspect_dataset(
    dataset_path: str,
    show_head: int = 5,
    show_tail: int = 0,
    show_stats: bool = True,
    show_nulls: bool = True,
    show_dtypes: bool = True,
    check_health: bool = True
) -> None:
    """Inspect dataset and perform health checks."""
    
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"error: file not found: {dataset_path}")
        sys.exit(1)
    
    file_size = path.stat().st_size
    print(f"\nDataset: {path.name}")
    print(f"- path: {path.absolute()}")
    print(f"- size: {format_size(file_size)}")
    
    try:
        if path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        elif path.suffix == '.csv':
            df = pd.read_csv(dataset_path, index_col=0)
        else:
            print(f"error: unsupported format: {path.suffix}")
            sys.exit(1)
    except Exception as e:
        print(f"error: failed to load dataset: {e}")
        sys.exit(1)
    
    print(f"- shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if df.index.name:
        print(f"- index: {df.index.name} ({df.index.dtype})")
    
    if show_dtypes:
        print("\nColumn types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
    
    if show_nulls:
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print("\nNull values:")
            for col, count in null_counts[null_counts > 0].items():
                pct = 100 * count / len(df)
                print(f"  - {col}: {count:,} ({pct:.2f}%)")
        else:
            print("\n✓ No null values")
    
    if show_stats:
        print("\nColumn statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  - {col}:")
                print(f"    - mean: {vals.mean():.6f}")
                print(f"    - std: {vals.std():.6f}")
                print(f"    - min: {vals.min():.6f}")
                print(f"    - max: {vals.max():.6f}")
                print(f"    - median: {vals.median():.6f}")
    
    if 'target' in df.columns:
        print("\nTarget distribution:")
        target = df['target']
        pos = (target > 0).sum()
        neg = (target < 0).sum()
        neu = (target == 0).sum()
        total = len(target)
        print(f"  - positive: {pos:,} ({100*pos/total:.2f}%)")
        print(f"  - negative: {neg:,} ({100*neg/total:.2f}%)")
        print(f"  - neutral: {neu:,} ({100*neu/total:.2f}%)")
    
    if 'asset_id' in df.columns:
        print("\nAsset distribution:")
        asset_counts = df['asset_id'].value_counts().sort_index()
        for asset_id, count in asset_counts.items():
            pct = 100 * count / len(df)
            print(f"  - asset {asset_id}: {count:,} ({pct:.2f}%)")
    
    if show_head > 0:
        print(f"\nFirst {show_head} rows:")
        print(df.head(show_head).to_string())
    
    if show_tail > 0:
        print(f"\nLast {show_tail} rows:")
        print(df.tail(show_tail).to_string())
    
    if check_health:
        print("\nHealth checks:")
        issues = []
        
        if df.isnull().any().any():
            issues.append("contains null values")
        
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            issues.append(f"{dup_count:,} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                issues.append(f"{col}: {inf_count:,} inf values")
        
        if len(df) == 0:
            issues.append("empty dataset")
        
        if len(df.columns) == 0:
            issues.append("no columns")
        
        if issues:
            print("  ⚠ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ Dataset is healthy")
    
    print("\ndone: inspection complete")


def main():
    parser = argparse.ArgumentParser(
        description='Janus Dataset Inspector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_dataset.py --path outputs/datasets/pre-train/parquet/janus_pretrain_5min_dataset.parquet
  python inspect_dataset.py --path dataset.parquet --head 10 --tail 5
  python inspect_dataset.py --path dataset.csv --no-stats --no-nulls
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to dataset file (parquet or csv)'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=5,
        help='Number of head rows to show (default: 5, 0 to disable)'
    )
    parser.add_argument(
        '--tail',
        type=int,
        default=0,
        help='Number of tail rows to show (default: 0)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable column statistics'
    )
    parser.add_argument(
        '--no-nulls',
        action='store_true',
        help='Disable null value check'
    )
    parser.add_argument(
        '--no-dtypes',
        action='store_true',
        help='Disable column types display'
    )
    parser.add_argument(
        '--no-health',
        action='store_true',
        help='Disable health checks'
    )
    
    args = parser.parse_args()
    
    inspect_dataset(
        dataset_path=args.path,
        show_head=args.head,
        show_tail=args.tail,
        show_stats=not args.no_stats,
        show_nulls=not args.no_nulls,
        show_dtypes=not args.no_dtypes,
        check_health=not args.no_health
    )


if __name__ == '__main__':
    main()
