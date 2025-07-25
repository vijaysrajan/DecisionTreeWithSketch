import csv
from typing import Dict, Tuple, List, Optional
import os

# Import our abstract base class and factory
from sketches import DataSketch, SketchFactory


def parse_csv_input(csv_file_path: str, sketch_type: str = 'bitvector', **sketch_kwargs) -> Tuple[Dict[str, DataSketch], Dict[str, DataSketch]]:
    """
    Parse processed CSV file into feature and target sketches.
    
    Expected CSV format:
    dim,type,base64_sketch
    source=Uber,feature,<base64_encoded_bitvector>
    city=Mumbai,feature,<base64_encoded_bitvector>
    tripOutcome=success,target,<base64_encoded_bitvector>
    tripOutcome=cancelled,target,<base64_encoded_bitvector>
    
    Args:
        csv_file_path (str): Path to processed CSV file
        sketch_type (str): Type of sketch to create ('bitvector' or 'thetasketch')
        **sketch_kwargs: Additional parameters for sketch creation (e.g., universe_size)
        
    Returns:
        Tuple[Dict[str, DataSketch], Dict[str, DataSketch]]: 
            - features dict: {"dim_name": DataSketch, ...}
            - targets dict: {"y=0": DataSketch, "y=1": DataSketch}
            
    Raises:
        ValueError: If target doesn't have exactly 2 values or other validation errors
        FileNotFoundError: If CSV file doesn't exist
    """
    
    # Validate file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    features = {}
    target_sketches = {}
    target_dims = []
    
    print(f"Parsing CSV input: {csv_file_path}")
    print(f"Using sketch type: {sketch_type}")
    
    # Read CSV file
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Validate CSV header
        expected_columns = {'dim', 'type', 'base64_sketch'}
        if not expected_columns.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must have columns: {expected_columns}. Found: {reader.fieldnames}")
        
        row_count = 0
        for row in reader:
            row_count += 1
            dim_name = row['dim'].strip()
            sketch_type_col = row['type'].strip().lower()
            base64_sketch = row['base64_sketch'].strip()
            
            # Validate row data
            if not dim_name or not sketch_type_col or not base64_sketch:
                raise ValueError(f"Row {row_count}: Missing data in dim/type/base64_sketch")
            
            # Deserialize DataSketch from base64 using factory
            try:
                data_sketch = SketchFactory.from_base64(base64_sketch, sketch_type, **sketch_kwargs)
            except Exception as e:
                raise ValueError(f"Row {row_count}: Failed to decode DataSketch for '{dim_name}': {e}")
            
            # Categorize as feature or target
            if sketch_type_col == 'feature':
                features[dim_name] = data_sketch
                print(f"  Feature: {dim_name} -> {data_sketch.get_count()} rows")
                
            elif sketch_type_col == 'target':
                target_sketches[dim_name] = data_sketch
                target_dims.append(dim_name)
                print(f"  Target: {dim_name} -> {data_sketch.get_count()} rows")
                
            else:
                raise ValueError(f"Row {row_count}: Invalid type '{sketch_type_col}'. Must be 'feature' or 'target'")
    
    print(f"Loaded {len(features)} features and {len(target_sketches)} target values")
    
    # Validate binary target (exactly 2 target values)
    if len(target_sketches) != 2:
        raise ValueError(f"Target must have exactly 2 values for binary classification. Found {len(target_sketches)}: {list(target_sketches.keys())}")
    
    # Convert target sketches to y=0, y=1 format
    target_values = list(target_sketches.keys())
    binary_targets = {
        "y=0": target_sketches[target_values[0]],  # First target value -> y=0
        "y=1": target_sketches[target_values[1]]   # Second target value -> y=1
    }
    
    print(f"Binary target mapping:")
    print(f"  y=0 -> {target_values[0]} ({binary_targets['y=0'].get_count()} rows)")
    print(f"  y=1 -> {target_values[1]} ({binary_targets['y=1'].get_count()} rows)")
    
    return features, binary_targets


def validate_sketches(features: Dict[str, DataSketch], targets: Dict[str, DataSketch]) -> Dict[str, int]:
    """
    Validate and analyze parsed sketches.
    
    Args:
        features (Dict[str, DataSketch]): Feature sketches
        targets (Dict[str, DataSketch]): Target sketches (y=0, y=1)
        
    Returns:
        Dict[str, int]: Validation statistics
    """
    
    stats = {
        'total_features': len(features),
        'total_targets': len(targets),
        'y0_count': targets['y=0'].get_count(),
        'y1_count': targets['y=1'].get_count(),
    }
    
    # Check for empty sketches
    empty_features = [name for name, sketch in features.items() if sketch.get_count() == 0]
    empty_targets = [name for name, sketch in targets.items() if sketch.get_count() == 0]
    
    if empty_features:
        print(f"Warning: {len(empty_features)} features have no rows: {empty_features}")
    
    if empty_targets:
        print(f"Warning: {len(empty_targets)} targets have no rows: {empty_targets}")
    
    # Check for overlap between y=0 and y=1 (should be disjoint)
    overlap = targets['y=0'].intersect(targets['y=1'])
    if overlap.get_count() > 0:
        print(f"Warning: Target classes overlap in {overlap.get_count()} rows (should be disjoint)")
        stats['target_overlap'] = overlap.get_count()
    else:
        print("✅ Target classes are disjoint (good)")
        stats['target_overlap'] = 0
    
    # Total dataset size
    total_rows = stats['y0_count'] + stats['y1_count']
    stats['total_rows'] = total_rows
    stats['class_balance'] = stats['y1_count'] / total_rows if total_rows > 0 else 0
    
    print(f"Dataset statistics:")
    print(f"  Features: {stats['total_features']}")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Class 0: {stats['y0_count']:,} ({(1-stats['class_balance']):.1%})")  
    print(f"  Class 1: {stats['y1_count']:,} ({stats['class_balance']:.1%})")
    
    return stats
