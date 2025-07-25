#!/usr/bin/env python3
"""
Complete example of using the Decision Tree Classifier.

This shows the end-to-end workflow:
1. Raw CSV data → Processed sketches
2. Training the decision tree
3. Making predictions
4. Analyzing results
"""

# Import all necessary components
from sketches import SketchFactory
from raw_data_to_csv_processor  import CSVPreprocessor
from data_utils import parse_csv_input, validate_sketches
from tree import DecisionTreeClassifier
from impurity import ImpurityMethod

import pandas as pd
import os


def example_1_simple_manual_data():
    """Example 1: Create simple data manually and train tree."""
    
    print("="*70)
    print("EXAMPLE 1: Simple Manual Data")
    print("="*70)
    
    # Step 1: Create sample data manually using sketches
    universe_size = 100
    
    # Create target classes
    # Class 0 (approved): rows 0-59 (60 samples)
    # Class 1 (denied): rows 60-99 (40 samples)
    y0_sketch = SketchFactory.create_sketch('bitvector', set(range(0, 60)), universe_size=universe_size)
    y1_sketch = SketchFactory.create_sketch('bitvector', set(range(60, 100)), universe_size=universe_size)
    targets = {"y=0": y0_sketch, "y=1": y1_sketch}
    
    # Create features
    # High income: strong predictor (80% of high income get approved)
    high_income_approved = set(range(0, 48))  # 48 approved
    high_income_denied = set(range(60, 72))   # 12 denied
    high_income = SketchFactory.create_sketch('bitvector', high_income_approved | high_income_denied, universe_size=universe_size)
    
    # Good credit: moderate predictor  
    good_credit_approved = set(range(10, 50))  # 40 approved
    good_credit_denied = set(range(70, 90))    # 20 denied
    good_credit = SketchFactory.create_sketch('bitvector', good_credit_approved | good_credit_denied, universe_size=universe_size)
    
    features = {
        "income=high": high_income,
        "credit=good": good_credit
    }
    
    print(f"Dataset: {y0_sketch.get_count() + y1_sketch.get_count()} samples")
    print(f"Features: {len(features)}")
    print(f"Class distribution: {y0_sketch.get_count()} approved, {y1_sketch.get_count()} denied")
    
    # Step 2: Train the decision tree
    tree = DecisionTreeClassifier(
        criterion='entropy',           # or 'gini'
        max_depth=5,                  # Maximum tree depth
        min_samples_leaf=5,           # Minimum samples in leaf
        store_node_sketches=True      # Store sketches for analysis
    )
    
    print(f"\nTraining decision tree...")
    tree.fit(features, targets)
    
    # Step 3: Print tree structure
    tree.print_tree()
    
    # Step 4: Make predictions
    test_samples = [
        {"income": "high", "credit": "good"},    # Should likely approve
        {"income": "high", "credit": "poor"},    # Mixed case
        {"income": "low", "credit": "good"},     # Mixed case  
        {"income": "low", "credit": "poor"}      # Should likely deny
    ]
    
    print(f"\nMaking predictions:")
    predictions = tree.predict(test_samples)
    probabilities = tree.predict_proba(test_samples)
    
    for i, sample in enumerate(test_samples):
        pred_class = "APPROVED" if predictions[i] == 0 else "DENIED"
        prob_approve = probabilities[i][0] * 100
        prob_deny = probabilities[i][1] * 100
        
        print(f"  {sample} → {pred_class} (Approve: {prob_approve:.1f}%, Deny: {prob_deny:.1f}%)")
    
    # Step 5: Feature importance
    importance = tree.get_feature_importance()
    print(f"\nFeature importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.3f}")
    
    return tree


def example_2_csv_preprocessing_workflow():
    """Example 2: Complete workflow from raw CSV to predictions."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Complete CSV Preprocessing Workflow")
    print("="*70)
    
    # Step 1: Create sample raw CSV data
    raw_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'income_level': ['high'] * 40 + ['medium'] * 35 + ['low'] * 25,
        'credit_score': ['excellent'] * 30 + ['good'] * 40 + ['fair'] * 20 + ['poor'] * 10,
        'employment': ['stable'] * 70 + ['unstable'] * 30,
        'loan_approved': ['yes'] * 65 + ['no'] * 35
    })
    
    raw_csv_path = 'sample_loan_data.csv'
    raw_data.to_csv(raw_csv_path, index=False)
    print(f"Created sample raw data: {raw_csv_path}")
    print(f"Raw data shape: {raw_data.shape}")
    print(raw_data.head())
    
    # Step 2: Preprocess raw CSV into sketch format
    preprocessor = CSVPreprocessor(
        categorical_threshold=0.5,    # Columns with <50% unique values are categorical
        sketch_type='bitvector',      # Use BitVector sketches
        use_bitarray=True            # Use efficient bitarray mode
    )
    
    processed_csv_path = 'processed_loan_data.csv'
    
    print(f"\nPreprocessing CSV...")
    stats = preprocessor.process_and_save(
        input_csv_path=raw_csv_path,
        output_csv_path=processed_csv_path,
        target_columns=['loan_approved']  # Specify target column
    )
    
    print(f"Preprocessing complete:")
    print(f"  Total sketches: {stats['total_sketches']}")
    print(f"  Feature sketches: {stats['feature_sketches']}")
    print(f"  Target sketches: {stats['target_sketches']}")
    print(f"  Universe size: {stats['universe_size']}")
    
    # Step 3: Parse processed CSV for decision tree
    print(f"\nParsing processed CSV...")
    features, targets = parse_csv_input(
        processed_csv_path, 
        sketch_type='bitvector',
        universe_size=stats['universe_size']
    )
    
    # Step 4: Validate data
    validation_stats = validate_sketches(features, targets)
    
    # Step 5: Train decision tree
    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=4,
        min_samples_leaf=3
    )
    
    print(f"\nTraining decision tree on real data...")
    tree.fit(features, targets)
    
    # Step 6: Print results
    tree.print_tree()
    
    # Step 7: Make predictions on new samples
    new_customers = [
        {"income_level": "high", "credit_score": "excellent", "employment": "stable"},
        {"income_level": "low", "credit_score": "poor", "employment": "unstable"},
        {"income_level": "medium", "credit_score": "good", "employment": "stable"}
    ]
    
    print(f"\nPredictions for new customers:")
    predictions = tree.predict(new_customers)
    probabilities = tree.predict_proba(new_customers)
    
    for i, customer in enumerate(new_customers):
        decision = "APPROVE" if predictions[i] == 0 else "DENY"
        confidence = max(probabilities[i]) * 100
        print(f"  Customer {i+1}: {customer}")
        print(f"    Decision: {decision} (Confidence: {confidence:.1f}%)")
    
    # Cleanup
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
    if os.path.exists(processed_csv_path):
        os.remove(processed_csv_path)
    
    return tree


def example_3_your_du_raw_data():
    """Example 3: How to use with your actual DU_raw.csv data."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Using Your DU_raw.csv Data")
    print("="*70)
    
    # This shows how you would process your actual data
    print("To use with your DU_raw.csv data:")
    print()
    
    code_example = '''
# Step 1: Preprocess your raw CSV
preprocessor = CSVPreprocessor(
    categorical_threshold=0.5,
    sketch_type='bitvector', 
    use_bitarray=True
)

# Convert DU_raw.csv to sketch format
stats = preprocessor.process_and_save(
    input_csv_path='DU_raw.csv',
    output_csv_path='DU_processed.csv',
    target_columns=['tripOutcome']  # Your target column
)

# Step 2: Parse for decision tree
features, targets = parse_csv_input(
    'DU_processed.csv', 
    sketch_type='bitvector',
    universe_size=stats['universe_size']
)

# Step 3: Train decision tree
tree = DecisionTreeClassifier(
    criterion='entropy',      # or 'gini'
    max_depth=10,            # Adjust based on your data
    min_samples_leaf=20      # Adjust based on dataset size
)

tree.fit(features, targets)

# Step 4: Make predictions
new_trips = [
    {"source": "Uber", "city": "Mumbai", "dayType": "weekday"},
    {"source": "Lyft", "city": "Delhi", "dayType": "weekend"}
]

predictions = tree.predict(new_trips)
probabilities = tree.predict_proba(new_trips)

# Step 5: Analyze results
tree.print_tree()
feature_importance = tree.get_feature_importance()
    '''
    
    print(code_example)


def example_4_advanced_usage():
    """Example 4: Advanced usage patterns."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Advanced Usage Patterns")
    print("="*70)
    
    print("Advanced features and configurations:")
    print()
    
    advanced_examples = '''
# 1. Using different sketch types (when ThetaSketch is implemented)
tree_theta = DecisionTreeClassifier(
    sketch_type='thetasketch',
    criterion='gini'
)

# 2. Storing node sketches for analysis
tree_with_analysis = DecisionTreeClassifier(
    store_node_sketches=True,  # Enable detailed analysis
    criterion='entropy'
)

# After training, analyze data flow through tree
def analyze_tree_nodes(node, depth=0):
    indent = "  " * depth
    y0_sketch, y1_sketch = node.get_data_sketches()
    
    if y0_sketch and y1_sketch:
        total = y0_sketch.get_count() + y1_sketch.get_count()
        print(f"{indent}Node {node.node_id}: {total} samples")
        print(f"{indent}  Class 0: {y0_sketch.get_count()}")
        print(f"{indent}  Class 1: {y1_sketch.get_count()}")
    
    if not node.is_leaf:
        analyze_tree_nodes(node.left_child, depth + 1)
        analyze_tree_nodes(node.right_child, depth + 1)

# 3. Comparing different criteria
for criterion in ['entropy', 'gini']:
    tree = DecisionTreeClassifier(criterion=criterion)
    tree.fit(features, targets)
    importance = tree.get_feature_importance()
    print(f"{criterion} - Best feature: {max(importance.items(), key=lambda x: x[1])}")

# 4. Grid search for hyperparameters
best_score = 0
best_params = {}

for max_depth in [3, 5, 7, 10]:
    for min_samples_leaf in [1, 5, 10]:
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        tree.fit(features, targets)
        
        # Evaluate model (you'd use cross-validation in practice)
        score = tree.leaf_count  # Simple metric
        if score > best_score:
            best_score = score
            best_params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

print(f"Best parameters: {best_params}")
    '''
    
    print(advanced_examples)


def main():
    """Run all examples."""
    
    print("DECISION TREE CLASSIFIER - COMPLETE USAGE GUIDE")
    print("="*70)
    
    try:
        # Run examples
        tree1 = example_1_simple_manual_data()
        tree2 = example_2_csv_preprocessing_workflow()
        example_3_your_du_raw_data()
        example_4_advanced_usage()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Key takeaways:")
        print("1. Use CSVPreprocessor to convert raw data to sketch format")
        print("2. Use parse_csv_input() to load sketches for training")
        print("3. Create DecisionTreeClassifier with your preferred settings")
        print("4. Call tree.fit(features, targets) to train")
        print("5. Use tree.predict() and tree.predict_proba() for predictions")
        print("6. Analyze with tree.print_tree() and tree.get_feature_importance()")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
