#!/usr/bin/env python3
"""
QUICK START: Minimum code to use Decision Tree Classifier with Threshold and ROC-AUC

For your DU_raw.csv data - just run this!
"""

from raw_data_to_csv_processor import CSVPreprocessor
from data_utils import parse_csv_input
from tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Convert your raw CSV to sketch format (ONE TIME SETUP)
# ============================================================================

def preprocess_your_data():
    """Convert DU_raw.csv to decision tree format."""
    
    # Create preprocessor
    preprocessor = CSVPreprocessor(sketch_type='bitvector', use_bitarray=True)
    
    # Process your data
    stats = preprocessor.process_and_save(
        input_csv_path='DU_raw.csv',        # Your raw data
        output_csv_path='DU_processed.csv', # Processed output
        target_columns=['tripOutcome']      # What you want to predict
    )
    
    print(f"✅ Processed {stats['total_rows']} rows into {stats['total_sketches']} sketches")
    return stats['universe_size']

# ============================================================================
# STEP 2: Train and use the decision tree with threshold
# ============================================================================

def train_and_predict():
    """Train decision tree and make predictions with different thresholds."""
    
    # Parse processed data
    features, targets = parse_csv_input('DU_processed.csv', sketch_type='bitvector')
    
    # Create and train tree with a specific prediction threshold
    tree = DecisionTreeClassifier(
        criterion='entropy',        # or 'gini'
        max_depth=16,               # Adjust for your data
        min_samples_leaf=18,       # Adjust for your data size
        prediction_threshold=0.5   # Default threshold
    )
    
    tree.fit(features, targets)
    
    # Print tree structure with default threshold
    print("\n🌲 Tree structure with default threshold (0.5):")
    tree.print_tree()
    
    # Make predictions on new trips with different thresholds
    new_trips = [
        {"source": "ios", "city": "Bangalore", "zone_popularity": "POPULARITY_INDEX_1", 
         "dayType": "WEEKEND", "sla": "Scheduled", "pickUpHourOfDay": "VERYLATE"},
        {"source": "android", "city": "Hyderabad", "zone_popularity": "POPULARITY_INDEX_1", 
         "dayType": "NORMAL_WEEKDAY", "sla": "Scheduled", "pickUpHourOfDay": "EVENING"},
        {"source": "android", "city": "Pune", "zone_popularity": "POPULARITY_INDEX_5", 
         "dayType": "NORMAL_WEEKDAY", "sla": "Scheduled", "pickUpHourOfDay": "EVENING"},
    ]
    
    # Get probabilities
    probabilities = tree.predict_proba(new_trips)
    
    print("\n📊 Predictions with different thresholds:")
    print("Sample probabilities [P(y=0), P(y=1)]:")
    for i, trip in enumerate(new_trips):
        print(f"Trip {i+1}: P(success)={probabilities[i][0]:.3f}, P(failure)={probabilities[i][1]:.3f}")
    
    print("\nPredictions at different thresholds:")
    for threshold in [0.3, 0.5, 0.7]:
        predictions = tree.predict(new_trips, threshold=threshold)
        print(f"\nThreshold={threshold}:")
        for i, trip in enumerate(new_trips):
            outcome = "SUCCESS" if predictions[i] == 0 else "FAILURE"
            print(f"  Trip {i+1}: {outcome}")
    
    # Feature importance
    importance = tree.get_feature_importance()
    print(f"\n📈 Most important features:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {score:.3f}")
    
    return tree

# ============================================================================
# STEP 3: Evaluate model with ROC-AUC
# ============================================================================

def evaluate_model_with_roc(tree, test_file='DU_test_processed.csv'):
    """Evaluate model using ROC-AUC on test data."""
    
    print("\n" + "="*50)
    print("🎯 Model Evaluation with ROC-AUC")
    print("="*50)
    
    # For demo purposes, we'll use the training data as test data
    # In practice, you should have a separate test set
    print("\n⚠️  Note: Using training data for demo. Use separate test set in practice!")
    
    # Parse test data (using training data for demo)
    features, targets = parse_csv_input('DU_processed.csv', sketch_type='bitvector')
    
    # Create test samples and true labels
    # We need to reconstruct individual samples from the sketches
    # This is a simplified approach - in practice, you'd keep track of the original data
    test_samples = []
    true_labels = []
    
    # Generate some synthetic test samples based on feature combinations
    # This is for demonstration - you should use actual test data
    feature_names = list(features.keys())
    base_features = list(set([f.split('=')[0] for f in feature_names]))
    
    print(f"\nGenerating test samples from {len(base_features)} base features...")
    
    # Create a few test samples
    sample_configs = [
        {"source": "android", "city": "Mumbai", "dayType": "WEEKEND"},
        {"source": "ios", "city": "Bangalore", "dayType": "NORMAL_WEEKDAY"},
        {"source": "android", "city": "Delhi", "dayType": "NORMAL_WEEKDAY"},
        {"source": "ios", "city": "Mumbai", "dayType": "WEEKEND"},
        {"source": "android", "city": "Bangalore", "dayType": "WEEKEND"},
    ]
    
    # For each sample config, predict using the tree
    for config in sample_configs:
        test_samples.append(config)
        # Get prediction probability to simulate a label
        # In practice, you'd have the actual label
        proba = tree.predict_proba([config])[0]
        # Simulate true label based on probability (for demo)
        true_label = 1 if proba[1] > 0.6 else 0
        true_labels.append(true_label)
    
    # Add more diverse samples
    import random
    random.seed(42)  # For reproducibility
    
    cities = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune"]
    sources = ["android", "ios", "web"]
    day_types = ["WEEKEND", "NORMAL_WEEKDAY", "HOLIDAY"]
    
    for _ in range(50):  # Add 50 more random samples
        sample = {
            "source": random.choice(sources),
            "city": random.choice(cities),
            "dayType": random.choice(day_types)
        }
        test_samples.append(sample)
        
        # Predict and simulate true label
        proba = tree.predict_proba([sample])[0]
        # Add some noise to make it more realistic
        noise = random.uniform(-0.1, 0.1)
        true_label = 1 if (proba[1] + noise) > 0.5 else 0
        true_labels.append(true_label)
    
    print(f"Created {len(test_samples)} test samples")
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds, auc = tree.compute_roc_auc(test_samples, true_labels)
    
    print(f"\n📊 ROC-AUC Results:")
    print(f"AUC Score: {auc:.3f}")
    
    # Print some threshold analysis
    print(f"\nThreshold Analysis (showing every 5th point):")
    print(f"{'Threshold':>10} {'FPR':>10} {'TPR':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 50)
    
    for i in range(0, len(thresholds), max(1, len(thresholds)//10)):
        threshold = thresholds[i]
        if threshold == 0.0:
            continue  # Skip the 0 threshold
            
        # Calculate precision for this threshold
        predictions = tree.predict(test_samples, threshold=threshold)
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"{threshold:10.3f} {fpr[i]:10.3f} {tpr[i]:10.3f} {precision:10.3f} {recall:10.3f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree Classifier')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('roc_curve.png', dpi=150)
    print(f"\n📈 ROC curve saved as 'roc_curve.png'")
    
    # Find optimal threshold based on Youden's J statistic
    j_scores = [tpr[i] - fpr[i] for i in range(len(tpr))]
    optimal_idx = j_scores.index(max(j_scores))
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🎯 Optimal threshold (Youden's J): {optimal_threshold:.3f}")
    print(f"   At this threshold: FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f}")
    
    return fpr, tpr, thresholds, auc

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("🌳 Decision Tree Classifier with Threshold & ROC-AUC")
    print("="*50)
    
    # Step 1: Preprocess data (run once)
    print("Step 1: Preprocessing data...")
    universe_size = preprocess_your_data()
    
    # Step 2: Train and predict with thresholds
    print("\nStep 2: Training tree and making predictions...")
    tree = train_and_predict()
    
    # Step 3: Evaluate with ROC-AUC
    print("\nStep 3: Evaluating model with ROC-AUC...")
    fpr, tpr, thresholds, auc = evaluate_model_with_roc(tree)
    
    print("\n🎉 Done! Your decision tree is ready to use.")
    
    # Show how to use different thresholds
    print("\n" + "="*50)
    print("📝 Usage Examples:")
    print("\n1. To predict with a custom threshold:")
    print("   predictions = tree.predict([samples], threshold=0.7)")
    
    print("\n2. To get probabilities:")
    print("   probabilities = tree.predict_proba([samples])")
    
    print("\n3. To compute ROC-AUC on your test data:")
    print("   fpr, tpr, thresholds, auc = tree.compute_roc_auc(test_samples, true_labels)")
    
    print("\n4. To find optimal threshold for your use case:")
    print("   - For balanced accuracy: Use Youden's J statistic")
    print("   - For high precision: Choose threshold with desired FPR")
    print("   - For high recall: Choose threshold with desired TPR")
