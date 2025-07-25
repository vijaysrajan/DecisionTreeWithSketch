#!/usr/bin/env python3
"""
Test both fixes: print_tree format and min_samples_leaf stopping condition.
"""

def test_print_tree_format():
    """Test the fixed print_tree format."""
    
    print("Testing print_tree format fix...")
    
    try:
        from nodes import TreeNode, ClassDistribution
        
        # Create a test node with split_gain = None
        class_dist = ClassDistribution(y0_count=10, y1_count=5)
        node = TreeNode(
            feature="test_feature=value",
            is_leaf=False,
            class_distribution=class_dist,
            depth=1,
            node_id=1
        )
        
        # Test with None split_gain (should not crash)
        print("Testing with None split_gain:")
        node.print_tree()
        
        # Test with actual split_gain
        node.split_gain = 0.1234
        print("\nTesting with numeric split_gain:")
        node.print_tree()
        
        print("✅ print_tree format fix works!")
        return True
        
    except Exception as e:
        print(f"❌ print_tree test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_min_samples_leaf():
    """Test the fixed min_samples_leaf stopping condition."""
    
    print("\nTesting min_samples_leaf stopping condition...")
    
    try:
        from sketches import SketchFactory
        from tree import DecisionTreeClassifier
        
        universe_size = 20
        
        # Create small dataset that should trigger min_samples_leaf
        # y=0: 10 samples (rows 0-9), y=1: 10 samples (rows 10-19)
        y0_small = SketchFactory.create_sketch('bitvector', set(range(0, 10)), universe_size=universe_size)
        y1_small = SketchFactory.create_sketch('bitvector', set(range(10, 20)), universe_size=universe_size)
        targets_small = {"y=0": y0_small, "y=1": y1_small}
        
        # Feature that would create very small leaves
        # Only covers 5 samples (rows 0-4), leaving 15 for the other side
        feature = SketchFactory.create_sketch('bitvector', set(range(0, 5)), universe_size=universe_size)
        features = {"feature=true": feature}
        
        print(f"Dataset: 20 total samples")
        print(f"Feature covers: 5 samples")
        print(f"Split would create: 5 samples (left) + 15 samples (right)")
        print(f"min_samples_leaf: 10")
        print(f"Expected: Should stop (left child < 10)")
        
        # Build tree with min_samples_leaf=10
        tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
        tree.fit(features, targets_small)
        
        # Check if root is a leaf (should be due to min_samples_leaf constraint)
        if tree.root.is_leaf:
            print("✅ min_samples_leaf stopping condition works!")
            print(f"   Root is leaf with prediction: {tree.root.class_distribution.predicted_class}")
            return True
        else:
            print("❌ min_samples_leaf stopping condition failed!")
            print(f"   Root split on: {tree.root.feature}")
            print(f"   This should not have happened with min_samples_leaf=10")
            return False
            
    except Exception as e:
        print(f"❌ min_samples_leaf test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run both tests."""
    print("="*60)
    print("TESTING BOTH FIXES")
    print("="*60)
    
    success1 = test_print_tree_format()
    success2 = test_min_samples_leaf()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 ALL FIXES WORK!")
        print("You should now be able to run the full test suite without errors.")
    else:
        print("❌ Some fixes still need work:")
        if not success1:
            print("   - print_tree format issue")
        if not success2:
            print("   - min_samples_leaf stopping condition issue")
    print("="*60)


if __name__ == "__main__":
    main()
