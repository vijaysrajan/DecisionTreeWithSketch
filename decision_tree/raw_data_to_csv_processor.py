import pandas as pd
import csv
from typing import Set, Dict, List, Optional, Tuple
from collections import defaultdict
import os

# Import our BitVector from sketches.py
from sketches import DataSketch, SketchFactory #BitVector

class CSVPreprocessor:
    """
    Preprocesses raw CSV files into sketch-based format for decision tree training.
    
    Converts raw data into BitVector sketches for each categorical column value,
    preparing data for efficient decision tree construction.
    """
    
    def __init__(self, categorical_threshold: float = 0.5, 
                 sketch_type: str = 'bitvector', 
                 use_bitarray: bool = True):
        """
        Initialize the CSV preprocessor.
        
        Args:
            categorical_threshold (float): Threshold for identifying categorical columns.
                                         If distinct_values/total_rows < threshold, treat as categorical.
            sketch_type (str): Type of sketch to create ('bitvector' or 'thetasketch')
            use_bitarray (bool): Whether to use bitarray mode for BitVectors (requires universe_size)
        """
        self.categorical_threshold = categorical_threshold
        self.sketch_type = sketch_type
        self.use_bitarray = use_bitarray
        self.universe_size: Optional[int] = None
        self.categorical_columns: List[str] = []
        self.column_value_counts: Dict[str, Dict[str, int]] = {}
        
    def identify_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify which columns should be treated as categorical.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            List[str]: List of column names that are categorical
        """
        categorical_cols = []
        total_rows = len(df)
        
        print(f"Analyzing {len(df.columns)} columns in {total_rows} rows...")
        print(f"Categorical threshold: {self.categorical_threshold} (distinct_values/total_rows)")
        
        for column in df.columns:
            # Skip if column is likely a unique identifier
            if self._is_likely_id_column(df[column]):
                print(f"  {column}: SKIPPED (likely unique ID column)")
                continue
                
            distinct_count = df[column].nunique()
            distinct_ratio = distinct_count / total_rows
            
            # Check if column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            data_type = "numeric" if is_numeric else "string/object"
            
            print(f"  {column}: {distinct_count} distinct values ({distinct_ratio:.3f} ratio) [{data_type}]")
            
            # Apply categorical rules
            if distinct_ratio < self.categorical_threshold:
                categorical_cols.append(column)
                print(f"    -> CATEGORICAL (ratio {distinct_ratio:.3f} < {self.categorical_threshold})")
            else:
                if is_numeric:
                    print(f"    -> CONTINUOUS NUMERIC (ratio {distinct_ratio:.3f} >= {self.categorical_threshold})")
                else:
                    print(f"    -> HIGH-CARDINALITY STRING (ratio {distinct_ratio:.3f} >= {self.categorical_threshold})")
                print(f"    -> SKIPPED")
                
        return categorical_cols
    
    def validate_expected_categorical_columns(self, df: pd.DataFrame, 
                                            expected_categorical: List[str]) -> Dict[str, str]:
        """
        Validate which expected categorical columns will actually be processed.
        
        Args:
            df (pd.DataFrame): Input dataframe
            expected_categorical (List[str]): List of expected categorical columns
            
        Returns:
            Dict[str, str]: Status for each expected column
        """
        validation_results = {}
        total_rows = len(df)
        
        print(f"Validating expected categorical columns...")
        
        for col_name in expected_categorical:
            if col_name not in df.columns:
                validation_results[col_name] = "MISSING - Column not found in data"
                print(f"  {col_name}: MISSING - Column not found in data")
                continue
            
            if self._is_likely_id_column(df[col_name]):
                validation_results[col_name] = "SKIPPED - Likely ID column"
                print(f"  {col_name}: SKIPPED - Likely ID column") 
                continue
                
            distinct_count = df[col_name].nunique()
            distinct_ratio = distinct_count / total_rows
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
            
            if distinct_ratio < self.categorical_threshold:
                validation_results[col_name] = f"WILL_PROCESS - {distinct_count} values ({distinct_ratio:.3f})"
                print(f"  {col_name}: WILL_PROCESS - {distinct_count} distinct values ({distinct_ratio:.3f} ratio)")
            else:
                data_type = "numeric" if is_numeric else "string"
                validation_results[col_name] = f"WILL_SKIP - Too many distinct values ({distinct_count}, {distinct_ratio:.3f}) [{data_type}]"
                print(f"  {col_name}: WILL_SKIP - {distinct_count} distinct values ({distinct_ratio:.3f} ratio) [{data_type}]")
        
        return validation_results
    
    def _is_likely_id_column(self, column_data: pd.Series) -> bool:
        """
        Check if a column is likely a unique identifier.
        
        Args:
            column_data (pd.Series): Column to analyze
            
        Returns:
            bool: True if likely an ID column
        """
        # High uniqueness ratio suggests ID column
        uniqueness_ratio = column_data.nunique() / len(column_data)
        
        # If >90% unique, likely an ID column
        if uniqueness_ratio > 0.9:
            return True
            
        # Check for ID-like patterns in column name
        column_name_lower = column_data.name.lower()
        id_keywords = ['id', 'key', 'uuid', 'guid', 'index']
        
        if any(keyword in column_name_lower for keyword in id_keywords):
            return True
            
        return False
    
    def create_sketch_features(self, df: pd.DataFrame) -> Dict[str, DataSketch]:
        """
        Create DataSketch features for each categorical column value.
        
        Args:
            df (pd.DataFrame): Input dataframe with row_id column added
            
        Returns:
            Dict[str, DataSketch]: Mapping from "column_name=value" to DataSketch
        """
        sketches = {}
        
        print(f"Creating {self.sketch_type} sketches (universe_size={self.universe_size})...")
        
        for column in self.categorical_columns:
            unique_values = df[column].unique()
            print(f"  Processing column '{column}' with {len(unique_values)} unique values...")
            
            for value in unique_values:
                # Skip null/NaN values
                if pd.isna(value):
                    continue
                    
                # Create dimension name
                dim_name = f"{column}={value}"
                
                # Find rows where this column has this value
                matching_rows = df[df[column] == value]['row_id'].tolist()
                
                # Create DataSketch with these row IDs using factory
                sketch_kwargs = {}
                if self.use_bitarray and self.universe_size is not None:
                    sketch_kwargs['universe_size'] = self.universe_size
                
                data_sketch = SketchFactory.create_sketch(
                    self.sketch_type, 
                    set(matching_rows), 
                    **sketch_kwargs
                )
                
                sketches[dim_name] = data_sketch
                
                print(f"    {dim_name}: {len(matching_rows)} rows -> {data_sketch.get_count()} count")
        
        return sketches
    
    def process_raw_csv(self, input_csv_path: str, 
                       target_columns: Optional[List[str]] = None,
                       expected_categorical: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, DataSketch]]:
        """
        Process raw CSV file and return dataframe with sketches.
        
        Args:
            input_csv_path (str): Path to raw CSV file
            target_columns (List[str], optional): Columns to treat as targets (rest are features)
            expected_categorical (List[str], optional): Expected categorical columns for validation
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, BitVector]]: Processed dataframe and sketch dictionary
        """
        print(f"Processing raw CSV: {input_csv_path}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(input_csv_path)
            print(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Add internal row_id column (0-indexed)
        df['row_id'] = range(len(df))
        self.universe_size = len(df)
        
        print(f"Added row_id column. Universe size: {self.universe_size}")
        
        # Validate expected categorical columns if provided
        if expected_categorical:
            validation_results = self.validate_expected_categorical_columns(df, expected_categorical)
            
            # Count how many will actually be processed
            will_process = sum(1 for status in validation_results.values() if status.startswith("WILL_PROCESS"))
            print(f"Expected {len(expected_categorical)} categorical columns, will process {will_process}")
        
        # Identify categorical columns
        self.categorical_columns = self.identify_categorical_columns(df)
        
        if not self.categorical_columns:
            raise ValueError("No categorical columns found. Check your categorical_threshold setting.")
        
        print(f"Found {len(self.categorical_columns)} categorical columns: {self.categorical_columns}")
        
        # Create DataSketch features
        sketches = self.create_sketch_features(df)
        
        print(f"Created {len(sketches)} {self.sketch_type} sketches")
        
        return df, sketches
    
    def save_processed_csv(self, sketches: Dict[str, DataSketch], 
                          output_csv_path: str,
                          target_columns: Optional[List[str]] = None):
        """
        Save processed sketches to CSV file in decision tree format.
        
        Args:
            sketches (Dict[str, BitVector]): Dictionary of dimension sketches
            output_csv_path (str): Path for output CSV file
            target_columns (List[str], optional): Columns to mark as targets
        """
        target_columns = target_columns or []
        
        print(f"Saving processed CSV to: {output_csv_path}")
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['dim', 'type', 'base64_sketch'])
            
            # Write each sketch
            for dim_name, data_sketch in sketches.items():
                # Determine if this is a target or feature
                column_name = dim_name.split('=')[0]  # Extract column name before '='
                
                if column_name in target_columns:
                    sketch_type = 'target'
                else:
                    sketch_type = 'feature'
                
                # Serialize DataSketch to base64
                base64_sketch = data_sketch.to_base64()
                
                # Write row
                writer.writerow([dim_name, sketch_type, base64_sketch])
        
        print(f"Successfully saved {len(sketches)} sketches to {output_csv_path}")
    
    def process_and_save(self, input_csv_path: str, output_csv_path: str,
                        target_columns: Optional[List[str]] = None,
                        expected_categorical: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Complete preprocessing pipeline: load, process, and save.
        
        Args:
            input_csv_path (str): Input raw CSV file
            output_csv_path (str): Output processed CSV file
            target_columns (List[str], optional): Target column names
            expected_categorical (List[str], optional): Expected categorical columns for validation
            
        Returns:
            Dict[str, int]: Summary statistics
        """
        # Process the raw CSV
        df, sketches = self.process_raw_csv(input_csv_path, target_columns, expected_categorical)
        
        # Save processed data
        self.save_processed_csv(sketches, output_csv_path, target_columns)
        
        # Return summary statistics
        target_count = 0
        feature_count = 0
        target_columns = target_columns or []
        
        for dim_name in sketches.keys():
            column_name = dim_name.split('=')[0]
            if column_name in target_columns:
                target_count += 1
            else:
                feature_count += 1
        
        return {
            'total_rows': len(df) - 1,  # Subtract 1 for added row_id
            'total_sketches': len(sketches),
            'categorical_columns': len(self.categorical_columns),
            'target_sketches': target_count,
            'feature_sketches': feature_count,
            'universe_size': self.universe_size
        }
