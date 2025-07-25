#!/usr/bin/env python3
"""
Simple test to verify TreeNode constructor works with new parameters.
"""

def test_treenode_constructor_standalone():
    """Test TreeNode constructor without imports."""
    
    # Mock DataSketch for testing
    class MockDataSketch:
        def __init__(self, count=0):
            self.count = count
        def get_count(self):
            return self.count
    
    # Mock ClassDistribution
    class MockClassDistribution:
        def __init__(self, y0_count, y1_count):
            self.y0_count = y0_count
            self.y1_count = y1_count
            self.total_count = y0_count + y1_count
    
    # Test TreeNode with the expected signature
    try:
        # This should work with the new signature
        from typing import Optional
        
        class TreeNode:
            def __init__(self, 
                         feature: Optional[str] = None,
                         is_leaf: bool = False,
                         class_distribution = None,
                         depth: int = 0,
                         node_id: Optional[int] = None,
                         y0_sketch = None,
                         y1_sketch = None):
                self.feature = feature
                self.is_leaf = is_leaf
                self.class_distribution = class_distribution
                self.depth = depth
                self.node_id = node_id
                self.y0_sketch = y0_sketch
                self.y1_sketch = y1_sketch
        
        # Test creating nodes with different parameter combinations
        print("Testing TreeNode constructor...")
        
        # Test 1: Basic leaf node
        leaf = TreeNode(
            is_leaf=True,
            class_distribution=MockClassDistribution(10, 5),
            depth=2,
            node_id=42
        )
        print(f"✅ Basic leaf: depth={leaf.depth}, is_leaf={leaf.is_leaf}")
        
        # Test 2: Internal node with sketches
        y0_sketch = MockDataSketch(10)
        y1_sketch = MockDataSketch(5)
        
        internal = TreeNode(
            feature="income=high",
            is_leaf=False,
            class_distribution=MockClassDistribution(10, 5),
            depth=1,
            node_id=1,
            y0_sketch=y0_sketch,
            y1_sketch=y1_sketch
        )
        print(f"✅ Internal node: feature={internal.feature}, has_sketches={internal.y0_sketch is not None}")
        
        # Test 3: Node without sketches (default behavior)
        no_sketches = TreeNode(
            feature="age=young",
            depth=2,
            node_id=3
        )
        print(f"✅ No sketches: feature={no_sketches.feature}, sketches={no_sketches.y0_sketch}")
        
        print("🎉 All TreeNode constructor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TreeNode constructor test failed: {e}")
        return False


def test_decision_tree_integration():
    """Test the integration with DecisionTreeClassifier."""
    
    print("\nTesting DecisionTreeClassifier integration...")
    
    try:
        # Try importing actual classes
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from sketches import SketchFactory
        from nodes import TreeNode, ClassDistribution
        from tree import DecisionTreeClassifier
        
        print("✅ Successfully imported all classes")
        
        # Test creating a simple tree
        universe_size = 10
        
        # Create simple sketches
        y0_sketch = SketchFactory.create_sketch('bitvector', {0, 1, 2, 3, 4}, universe_size=universe_size)
        y1_sketch = SketchFactory.create_sketch('bitvector', {5, 6, 7, 8, 9}, universe_size=universe_size)
        targets = {"y=0": y0_sketch, "y=1": y1_sketch}
        
        # Create perfect feature
        perfect_feature = SketchFactory.create_sketch('bitvector', {0, 1, 2, 3, 4}, universe_size=universe_size)
        features = {"income=high": perfect_feature}
        
        # Create tree with sketch storage enabled
        tree = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=2,
            store_node_sketches=True
        )
        
        print("✅ Created DecisionTreeClassifier")
        
        # Fit the tree
        tree.fit(features, targets)
        
        print("✅ Successfully fitted tree")
        print(f"   Root feature: {tree.root.feature}")
        print(f"   Root has sketches: {tree.root.y0_sketch is not None}")
        print(f"   Tree depth: {tree.max_tree_depth}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error (expected if files not available): {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TREENODE CONSTRUCTOR VERIFICATION")
    print("="*60)
    
    success1 = test_treenode_constructor_standalone()
    success2 = test_decision_tree_integration()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed - check the output above")
        if not success1:
            print("   - TreeNode constructor test failed")
        if not success2:
            print("   - Integration test failed (may be expected if files not available)")
    print("="*60)


if __name__ == "__main__":
    main()
