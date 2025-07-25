#!/usr/bin/env python3
"""
QUICK START: Minimum code to use Decision Tree Classifier

For your DU_raw.csv data - just run this!
"""

from raw_data_to_csv_processor import CSVPreprocessor
from data_utils import parse_csv_input
from tree import DecisionTreeClassifier

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
# STEP 2: Train and use the decision tree
# ============================================================================

def train_and_predict():
    """Train decision tree and make predictions."""
    
    # Parse processed data
    features, targets = parse_csv_input('DU_processed.csv', sketch_type='bitvector')
    
    # Create and train tree
    tree = DecisionTreeClassifier(
        criterion='entropy',        #'entropy',    # or 'gini'
        max_depth=7,           # Adjust for your data
        min_samples_leaf=50    # Adjust for your data size
    )
    
    tree.fit(features, targets)
    
    # Print tree structure
    tree.print_tree()
    
    # Make predictions on new trips
    new_trips = [
        {"source": "ios", "city": "Bangalore", "zone_popularity" : "POPULARITY_INDEX_1", "dayType": "WEEKEND", "sla": "Scheduled", "pickUpHourOfDay" : "VERYLATE" },
        {"source": "android", "city": "Hyderabad", "zone_popularity" : "POPULARITY_INDEX_1", "dayType": "NORMAL_WEEKDAY", "sla": "Scheduled", "pickUpHourOfDay" : "EVENING" },
        {"source": "android", "city": "Pune", "zone_popularity" : "POPULARITY_INDEX_5", "dayType": "NORMAL_WEEKDAY", "sla": "Scheduled", "pickUpHourOfDay" : "EVENING" },
    ]
    #source,estimated_usage_bins,city,zone,zone_demand_popularity,dayType,pickUpHourOfDay,sla,booking_type,tripOutcome
    #android,LTE_1_HOUR,Hyderabad,141,POPULARITY_INDEX_1,WEEKEND,LATENIGHT,Scheduled,one_way_trip,good
 
    predictions = tree.predict(new_trips)
    probabilities = tree.predict_proba(new_trips)
    
    print("\nPredictions:")
    for i, trip in enumerate(new_trips):
        outcome = "SUCCESS" if predictions[i] == 0 else "FAILURE"  
        confidence = max(probabilities[i]) * 100
        print(f"Trip {i+1}: {outcome} (Confidence: {confidence:.1f}%)")
        print(f"  Details: {trip}")
    
    # Feature importance
    importance = tree.get_feature_importance()
    print(f"\nMost important features:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {score:.3f}")
    
    return tree

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("🌳 Decision Tree Classifier - Quick Start")
    print("="*50)
    
    # Step 1: Preprocess data (run once)
    print("Step 1: Preprocessing data...")
    universe_size = preprocess_your_data()
    
    # Step 2: Train and predict
    print("\nStep 2: Training tree and making predictions...")
    tree = train_and_predict()
    
    print("\n🎉 Done! Your decision tree is ready to use.")
    
    # Show how to make more predictions
    print("\n" + "="*50)
    print("To make more predictions, use:")
    print("predictions = tree.predict([your_new_samples])")
    print("probabilities = tree.predict_proba([your_new_samples])")
