#!/usr/bin/env python3
"""
Generic CSV preprocessor for decision tree training data.

Converts any raw CSV file into sketch-based format for decision tree construction.
Automatically detects categorical columns and generates BitVector sketches for each
"column_name=value" combination.

Usage:
    python generate_processed_csv.py input.csv output.csv --target col1,col2
    python generate_processed_csv.py data.csv processed.csv --target outcome --threshold 0.3
    python generate_processed_csv.py file.csv out.csv --target target_col --expected cat1,cat2,cat3
"""

import argparse
import sys
import os
from typing import List, Optional

# Import our classes (assuming they're in the same directory)
from sketches import DataSketch, SketchFactory #BitVector
from raw_data_to_csv_processor import CSVPreprocessor


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert raw CSV to decision tree format with BitVector sketches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - specify input, output, and target column
  python generate_processed_csv.py data.csv processed.csv --target outcome
  
  # Multiple target columns
  python generate_processed_csv.py data.csv out.csv --target "col1,col2"
  
  # Custom categorical threshold
  python generate_processed_csv.py data.csv out.csv --target outcome --threshold 0.3
  
  # Specify expected categorical columns for validation
  python generate_processed_csv.py data.csv out.csv --target outcome --expected "cat1,cat2,cat3"
  
  # Use set-based mode instead of bitarray
  python generate_processed_csv.py data.csv out.csv --target outcome --no-bitarray
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_csv',
        help='Path to input raw CSV file'
    )
    
    parser.add_argument(
        'output_csv', 
        help='Path for output processed CSV file'
    )
    
    parser.add_argument(
        '--target', '-t',
        required=True,
        help='Target column name(s) for prediction (comma-separated if multiple)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Categorical threshold: if distinct_values/total_rows < threshold, treat as categorical (default: 0.5)'
    )
    
    parser.add_argument(
        '--expected',
        help='Expected categorical columns for validation (comma-separated)'
    )
    
    parser.add_argument(
        '--no-bitarray',
        action='store_true',
        help='Use set-based mode instead of bitarray mode'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def parse_comma_separated(value: str) -> List[str]:
    """
    Parse comma-separated string into list of strings.
    
    Args:
        value (str): Comma-separated string
        
    Returns:
        List[str]: List of trimmed strings
    """
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def validate_inputs(args) -> bool:
    """
    Validate command line arguments and input files.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if inputs are valid
    """
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"❌ ERROR: Input file '{args.input_csv}' not found")
        return False
    
    # Check if input file is readable
    try:
        with open(args.input_csv, 'r') as f:
            f.readline()  # Try to read first line
    except Exception as e:
        print(f"❌ ERROR: Cannot read input file '{args.input_csv}': {e}")
        return False
    
    # Check threshold range
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ ERROR: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        return False
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        print(f"❌ ERROR: Output directory '{output_dir}' does not exist")
        return False
    
    return True


def generate_decision_tree_csv(input_file: str, output_file: str, 
                              target_columns: List[str],
                              categorical_threshold: float = 0.5,
                              expected_categorical: Optional[List[str]] = None,
                              use_bitarray: bool = True,
                              verbose: bool = False):
    """
    Generate processed CSV for decision tree from any raw CSV data.
    
    Args:
        input_file (str): Path to raw CSV file
        output_file (str): Path for processed CSV output
        target_columns (List[str]): Column names to treat as targets
        categorical_threshold (float): Threshold for categorical detection
        expected_categorical (List[str], optional): Expected categorical columns
        use_bitarray (bool): Whether to use bitarray mode
        verbose (bool): Enable verbose output
    """
    
    if verbose:
        print("="*80)
        print("GENERIC CSV PREPROCESSOR FOR DECISION TREE")
        print("="*80)
        print(f"Input file:              {input_file}")
        print(f"Output file:             {output_file}")
        print(f"Target columns:          {target_columns}")
        print(f"Categorical threshold:   {categorical_threshold}")
        print(f"Use bitarray:            {use_bitarray}")
        print(f"Expected categorical:    {expected_categorical or 'Auto-detect'}")
        print("="*80)
    
    # Initialize preprocessor
    preprocessor = CSVPreprocessor(
        categorical_threshold=categorical_threshold,
        use_bitarray=use_bitarray
    )
    
    try:
        # Process and save
        stats = preprocessor.process_and_save(
            input_csv_path=input_file,
            output_csv_path=output_file,
            target_columns=target_columns,
            expected_categorical=expected_categorical
        )
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"✅ Input file:           {input_file}")
        print(f"✅ Output file:          {output_file}")
        print(f"✅ Total rows processed: {stats['total_rows']:,}")
        print(f"✅ Universe size:        {stats['universe_size']:,}")
        print(f"✅ Categorical columns:  {stats['categorical_columns']}")
        print(f"✅ Total sketches:       {stats['total_sketches']:,}")
        print(f"✅ Target sketches:      {stats['target_sketches']}")
        print(f"✅ Feature sketches:     {stats['feature_sketches']}")
        
        # Verify output file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ Output file size:     {file_size:,} bytes")
            
            if verbose:
                print(f"\n📋 NEXT STEPS:")
                print(f"   1. Use '{output_file}' as input for decision tree training")
                print(f"   2. Each row contains: dim, type, base64_sketch")
                print(f"   3. BitVectors are ready for intersection operations")
        else:
            print(f"❌ ERROR: Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False
        
    return True


def main():
    """
    Main function to process command line arguments and run preprocessing.
    """
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Parse target columns
    target_columns = parse_comma_separated(args.target)
    if not target_columns:
        print("❌ ERROR: At least one target column must be specified")
        sys.exit(1)
    
    # Parse expected categorical columns (optional)
    expected_categorical = None
    if args.expected:
        expected_categorical = parse_comma_separated(args.expected)
    
    # Determine bitarray usage
    use_bitarray = not args.no_bitarray
    
    # Run the preprocessing
    success = generate_decision_tree_csv(
        input_file=args.input_csv,
        output_file=args.output_csv,
        target_columns=target_columns,
        categorical_threshold=args.threshold,
        expected_categorical=expected_categorical,
        use_bitarray=use_bitarray,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)
    
    print(f"\n🎉 SUCCESS: '{args.output_csv}' is ready for decision tree training!")


if __name__ == "__main__":
    main()
