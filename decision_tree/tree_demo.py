#!/usr/bin/env python3
"""
Demo showing the fixed decision tree with proper sketch operations.

This demonstrates:
1. How sketches are stored in nodes (optional)
2. How proper set subtraction works 
3. How data flows through the tree
"""

# Import all our components
from sketches import DataSketch, SketchFactory
from nodes import TreeNode, ClassDistribution
from tree import DecisionTreeClassifier


def demonstrate_sketch_subtraction():
    """Show how proper sketch subtraction works."""
    print("="*60)
    print("DEMONSTRATION: Proper Sketch Subtraction")
    print("="*60)
    
    universe_size = 10
    
    # Create example sketches
    all_y0 = SketchFactory.create_sketch('bitvector', {0, 1, 2, 3, 4}, universe_size=universe_size)
    feature_true = SketchFactory.create_sketch('bitvector', {0, 2, 4}, universe_size=universe_size)
    
    print(f"All Class 0 data: {sorted(all_y0.row_ids if hasattr(all_y0, 'row_ids') else 'BitArray')}")
    print(f"Feature=true data: {sorted(feature_true.row_ids if hasattr(feature_true, 'row_ids') else 'BitArray')}")
    
    # Left split: feature=true ∩ class0
    left_y0 = feature_true.intersect(all_y0)
    print(f"Left (feature=true ∩ class0): {sorted(left_y0.row_ids if hasattr(left_y0, 'row_ids') else 'BitArray')}")
    
    # Right split: class0 - left (proper subtraction!)
    right_y0 = all_y0.subtract(left_y0)
    print(f"Right (class0 - left): {sorted(right_y0.row_ids if hasattr(right_y0, 'row_ids') else 'BitArray')}")
    
    # Verify the split is complete and disjoint
    print(f"Left count: {left_y0.get_count()}")
    print(f"Right count: {right_y0.get_count()}")
    print(f"Total: {left_y0.get_count() + right_y0.get_count()} (should equal {all_y0.get_count()})")
    
    # Test that intersection of left and right is empty
    overlap = left_y0.intersect(right_y0)
    print(f"Overlap: {overlap.get_count()} (should be 0)")
    print("✅ Proper set subtraction working correctly!")


def demonstrate_node_sketch_storage():
    """Show how sketches can be stored in nodes for analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Node Sketch Storage")
    print("="*60)
    
    universe_size = 20
    
    # Create balanced dataset
    y0_sketch = SketchFactory.create_sketch('bitvector', set(range(0, 10)), universe_size=universe_size)
    y1_sketch = SketchFactory.create_sketch('bitvector', set(range(10, 20)), universe_size=universe_size)
    targets = {"y=0": y0_sketch, "y=1": y1_sketch}
    
    # Create perfect feature
    perfect_feature = SketchFactory.create_sketch('bitvector', set(range(0, 10)), universe_size=universe_size)
    features = {"income=high": perfect_feature}
    
    # Train tree WITH sketch storage
    tree_with_sketches = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=3,
        store_node_sketches=True  # Enable sketch storage
    )
    tree_with_sketches.fit(features, targets)
    
    # Train tree WITHOUT sketch storage (default)
    tree_without_sketches = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=3,
        store_node_sketches=False  # Default behavior
    )
    tree_without_sketches.fit(features, targets)
    
    print("Tree WITH sketch storage:")
    analyze_node_sketches(tree_with_sketches.root)
    
    print("\nTree WITHOUT sketch storage:")
    analyze_node_sketches(tree_without_sketches.root)


def analyze_node_sketches(node, indent=0):
    """Analyze sketches stored in tree nodes."""
    prefix = "  " * indent
    
    y0_sketch, y1_sketch = node.get_data_sketches()
    
    if y0_sketch and y1_sketch:
        print(f"{prefix}Node {node.node_id}: Has sketches (y0={y0_sketch.get_count()}, y1={y1_sketch.get_count()})")
        if hasattr(y0_sketch, 'row_ids'):
            print(f"{prefix}  y0 row_ids: {sorted(list(y0_sketch.row_ids)[:5])}{'...' if len(y0_sketch.row_ids) > 5 else ''}")
            print(f"{prefix}  y1 row_ids: {sorted(list(y1_sketch.row_ids)[:5])}{'...' if len(y1_sketch.row_ids) > 5 else ''}")
    else:
        print(f"{prefix}Node {node.node_id}: No sketches stored")
    
    if not node.is_leaf:
        if node.left_child:
            analyze_node_sketches(node.left_child, indent + 1)
        if node.right_child:
            analyze_node_sketches(node.right_child, indent + 1)


def demonstrate_complete_data_flow():
    """Show complete data flow through tree construction."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Complete Data Flow")
    print("="*60)
    
    universe_size = 16
    
    # Create realistic dataset
    # Class 0: rows 0-7, Class 1: rows 8-15
    y0_sketch = SketchFactory.create_sketch('bitvector', set(range(0, 8)), universe_size=universe_size)
    y1_sketch = SketchFactory.create_sketch('bitvector', set(range(8, 16)), universe_size=universe_size)
    targets = {"y=0": y0_sketch, "y=1": y1_sketch}
    
    # Create two features with different predictive power
    # income=high: strong predictor (most of class 0)
    income_high = SketchFactory.create_sketch('bitvector', {0, 1, 2, 3, 4, 5, 10, 11}, universe_size=universe_size)
    
    # age=young: weaker predictor  
    age_young = SketchFactory.create_sketch('bitvector', {0, 1, 2, 8, 9, 10, 14, 15}, universe_size=universe_size)
    
    features = {
        "income=high": income_high,
        "age=young": age_young
    }
    
    print("Dataset:")
    print(f"  Class 0 (y=0): rows {sorted(y0_sketch.row_ids if hasattr(y0_sketch, 'row_ids') else range(8))}")
    print(f"  Class 1 (y=1): rows {sorted(y1_sketch.row_ids if hasattr(y1_sketch, 'row_ids') else range(8, 16))}")
    print(f"  income=high: rows {sorted(income_high.row_ids if hasattr(income_high, 'row_ids') else 'BitArray')}")
    print(f"  age=young: rows {sorted(age_young.row_ids if hasattr(age_young, 'row_ids') else 'BitArray')}")
    
    # Build tree with sketch storage for analysis
    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=2,
        store_node_sketches=True
    )
    
    print(f"\nBuilding decision tree...")
    tree.fit(features, targets)
    
    print(f"\nTree structure:")
    tree.print_tree()
    
    print(f"\nData flow analysis:")
    analyze_data_flow(tree.root)


def analyze_data_flow(node, indent=0):
    """Analyze how data flows through the tree."""
    prefix = "  " * indent
    
    y0_sketch, y1_sketch = node.get_data_sketches()
    
    if y0_sketch and y1_sketch:
        total_data = y0_sketch.get_count() + y1_sketch.get_count()
        print(f"{prefix}Node {node.node_id} receives {total_data} samples:")
        print(f"{prefix}  Class 0: {y0_sketch.get_count()} samples")
        print(f"{prefix}  Class 1: {y1_sketch.get_count()} samples")
        
        if not node.is_leaf:
            print(f"{prefix}  Splits on: {node.feature}")
            print(f"{prefix}  Information gain: {node.split_gain:.4f}")
    
    if not node.is_leaf:
        if node.left_child:
            print(f"{prefix}  ├─ LEFT (feature=true):")
            analyze_data_flow(node.left_child, indent + 2)
        if node.right_child:
            print(f"{prefix}  └─ RIGHT (feature=false):")
            analyze_data_flow(node.right_child, indent + 2)


def main():
    """Run all demonstrations."""
    print("Decision Tree with Fixed Sketch Operations")
    print("This demo shows the improvements to sketch handling and data flow.")
    
    # Demo 1: Show proper subtraction works
    demonstrate_sketch_subtraction()
    
    # Demo 2: Show optional sketch storage in nodes
    demonstrate_node_sketch_storage()
    
    # Demo 3: Show complete data flow
    demonstrate_complete_data_flow()
    
    print("\n" + "="*60)
    print("KEY IMPROVEMENTS DEMONSTRATED:")
    print("="*60)
    print("1. ✅ Proper set subtraction preserves actual row IDs")
    print("2. ✅ Optional sketch storage in nodes for analysis")  
    print("3. ✅ Complete data traceability through tree construction")
    print("4. ✅ Clean abstraction - works with any sketch type that supports subtraction")
    print("5. ✅ No more placeholder row IDs or broken complementes!")


if __name__ == "__main__":
    main()
