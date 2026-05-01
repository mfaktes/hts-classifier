"""
Data preparation script
-----------------------
Run this once after downloading a fresh HTSUS export from
https://hts.usitc.gov/export

It transforms the raw flat CSV into one with a `full_description` column that
contains the full hierarchical context of each HTS code.

Why this matters: in the raw HTSUS, the row for `6105.10.00.10` just says
"Men's (338)". To know that means *men's cotton knitted shirts*, you need to
walk up the indent levels to find the parent rows "Of cotton", "Men's or
boys' shirts, knitted or crocheted". This script does that walk for every row.

Usage:
    python prepare_data.py path/to/htsdata_raw.csv

Output:
    hts_processed.csv  (used by classifier.py and app.py)
"""

import sys
import pandas as pd
from pathlib import Path


def build_hierarchical_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Walk the indent column to construct full_description for each row."""
    parent_stack = {}  # indent_level -> description at that level
    full_descriptions = []

    for _, row in df.iterrows():
        indent = int(row['Indent']) if pd.notna(row['Indent']) else 0
        desc = str(row['Description']) if pd.notna(row['Description']) else ""

        # Set this level
        parent_stack[indent] = desc.strip().rstrip(':')
        # Forget anything deeper (it belongs to a different branch now)
        for deeper in list(parent_stack.keys()):
            if deeper > indent:
                del parent_stack[deeper]

        # Concatenate root -> current
        full = " > ".join(parent_stack[i] for i in sorted(parent_stack.keys()))
        full_descriptions.append(full)

    df = df.copy()
    df['full_description'] = full_descriptions
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py path/to/htsdata_raw.csv")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    print(f"Loading {src}...")
    df = pd.read_csv(src)
    print(f"  {len(df):,} raw rows loaded")

    print("Building hierarchical descriptions...")
    df = build_hierarchical_descriptions(df)

    # Drop placeholder rows that have no HTS code (they're category headers
    # whose context is now baked into their children's full_description)
    df['HTS Number'] = df['HTS Number'].astype(str).replace('nan', '')
    df = df[df['HTS Number'].str.len() > 0].copy()
    print(f"  {len(df):,} classifiable rows after filtering")

    out = Path(__file__).parent / "hts_processed.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
