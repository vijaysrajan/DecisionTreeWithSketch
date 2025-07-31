#!/usr/bin/env python3
"""
Test runner specifically for JSON functionality tests.
Run this to test only the JSON-related methods.
"""

import unittest
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from test_node import TestClassDistribution, TestTreeNode, TestTreeNodeIntegration
    from test_tree import TestDecisionTreeBasic, TestDecisionTreeAdvanced, TestDecisionTreeIntegration
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import test modules: {e}")

def run_json_tests():
    """Run only JSON-related tests."""
    if not IMPORTS_AVAILABLE:
        print("Tests not available - check imports")
        return
    
    # Create test suite for JSON tests only
    json_suite = unittest.TestSuite()
    
    # JSON tests from test_node.py
    json_suite.addTest(TestClassDistribution('test_to_dict_json_serialization'))
    json_suite.addTest(TestTreeNode('test_leaf_node_to_json'))
    json_suite.addTest(TestTreeNode('test_internal_node_to_json'))
    json_suite.addTest(TestTreeNode('test_json_threshold_propagation'))
    json_suite.addTest(TestTreeNodeIntegration('test_complex_tree_json_structure'))
    
    # JSON tests from test_tree.py
    json_suite.addTest(TestDecisionTreeBasic('test_json_before_training'))
    json_suite.addTest(TestDecisionTreeBasic('test_json_after_training'))
    json_suite.addTest(TestDecisionTreeBasic('test_json_with_different_thresholds'))
    json_suite.addTest(TestDecisionTreeBasic('test_json_pretty_printing'))
    json_suite.addTest(TestDecisionTreeAdvanced('test_json_complex_tree_structure'))
    json_suite.addTest(TestDecisionTreeAdvanced('test_json_with_different_criteria'))
    json_suite.addTest(TestDecisionTreeAdvanced('test_json_serialization_edge_cases'))
    json_suite.addTest(TestDecisionTreeIntegration('test_integration_json_loan_scenario'))
    
    # Run the JSON tests
    print("🧪 Running JSON-specific tests...")
    print("=" * 60)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(json_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 JSON Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("   ✅ All JSON tests passed!")
        elif success_rate >= 80:
            print("   ⚠️  Most JSON tests passed")
        else:
            print("   ❌ Many JSON tests failed")
    
    # Print details about failures if any
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    return result.wasSuccessful()

def run_quick_json_demo():
    """Run a quick demonstration of JSON functionality."""
    print("\n🚀 Quick JSON Demo:")
    print("=" * 60)
    
    try:
        from sketches import SketchFactory
        from nodes import TreeNode, ClassDistribution
        from tree import DecisionTreeClassifier
        import json
        
        print("1. Testing ClassDistribution.to_dict()...")
        cd = ClassDistribution(y0_count=75, y1_count=25)
        cd_dict = cd.to_dict()
        print(f"   Result: {cd_dict}")
        print(f"   JSON serializable: {json.dumps(cd_dict) is not None}")
        
        print("\n2. Testing TreeNode.to_json()...")
        leaf = TreeNode(
            is_leaf=True,
            class_distribution=cd,
            depth=1,
            node_id=5
        )
        leaf_json = leaf.to_json(threshold=0.5)
        print(f"   Leaf JSON keys: {list(leaf_json.keys())}")
        print(f"   Predicted class: {leaf_json['predicted_class']}")
        
        print("\n3. Testing DecisionTreeClassifier.to_json()...")
        
        # Create simple dataset
        universe_size = 100
        y0 = SketchFactory.create_sketch('bitvector', set(range(0, 50)), universe_size=universe_size)
        y1 = SketchFactory.create_sketch('bitvector', set(range(50, 100)), universe_size=universe_size)
        targets = {"y=0": y0, "y=1": y1}
        
        feature = SketchFactory.create_sketch('bitvector', set(range(0, 50)), universe_size=universe_size)
        features = {"income=high": feature}
        
        # Train tree
        tree = DecisionTreeClassifier(criterion='entropy', prediction_threshold=0.6)
        tree.fit(features, targets)
        
        # Get JSON
        tree_json_str = tree.to_json(threshold=0.4, indent=2)
        tree_json = json.loads(tree_json_str)
        
        print(f"   Tree JSON top-level keys: {list(tree_json.keys())}")
        print(f"   Tree info keys: {list(tree_json['tree_info'].keys())}")
        print(f"   Threshold used: {tree_json['tree_info']['prediction_threshold']}")
        print(f"   JSON length: {len(tree_json_str)} characters")
        
        # Test different thresholds
        print("\n4. Testing threshold variations...")
        for threshold in [0.2, 0.5, 0.8]:
            json_dict = json.loads(tree.to_json(threshold=threshold))
            root_pred = json_dict['tree_structure']['left_child']['predicted_class']
            print(f"   Threshold {threshold}: left child predicts class {root_pred}")
        
        print("\n✅ JSON demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ JSON demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🌳 Decision Tree JSON Testing")
    print("=" * 60)
    
    # Run quick demo first
    demo_success = run_quick_json_demo()
    
    if demo_success:
        print("\n" + "=" * 60)
        # Run comprehensive tests
        test_success = run_json_tests()
        
        if test_success:
            print("\n🎉 All JSON functionality is working correctly!")
        else:
            print("\n⚠️  Some JSON tests failed - check the output above")
    else:
        print("\n❌ JSON demo failed - check your implementation")
        sys.exit(1)
