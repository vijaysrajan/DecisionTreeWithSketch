#!/usr/bin/env python3
"""
Test file for nodes.py - tests TreeNode and ClassDistribution with threshold support and JSON functionality.
"""

import unittest
from typing import Dict, Any, Optional
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nodes import TreeNode, ClassDistribution
    from sketches import DataSketch, SketchFactory
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import required modules.")


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestClassDistribution(unittest.TestCase):
    """Test ClassDistribution functionality including threshold support."""
    
    def test_basic_properties(self):
        """Test basic properties of ClassDistribution."""
        # Balanced distribution
        cd_balanced = ClassDistribution(y0_count=50, y1_count=50)
        self.assertEqual(cd_balanced.total_count, 100)
        self.assertAlmostEqual(cd_balanced.y0_probability, 0.5)
        self.assertAlmostEqual(cd_balanced.y1_probability, 0.5)
        self.assertAlmostEqual(cd_balanced.confidence, 0.5)
        
        # Imbalanced distribution
        cd_imbalanced = ClassDistribution(y0_count=70, y1_count=30)
        self.assertEqual(cd_imbalanced.total_count, 100)
        self.assertAlmostEqual(cd_imbalanced.y0_probability, 0.7)
        self.assertAlmostEqual(cd_imbalanced.y1_probability, 0.3)
        self.assertAlmostEqual(cd_imbalanced.confidence, 0.7)
        
        # Empty distribution
        cd_empty = ClassDistribution(y0_count=0, y1_count=0)
        self.assertEqual(cd_empty.total_count, 0)
        self.assertEqual(cd_empty.y0_probability, 0.0)
        self.assertEqual(cd_empty.y1_probability, 0.0)
        self.assertEqual(cd_empty.confidence, 0.0)
    
    def test_predicted_class_with_threshold(self):
        """Test predicted_class method with different thresholds."""
        # Create distribution with P(y=1) = 0.3
        cd = ClassDistribution(y0_count=70, y1_count=30)
        
        # Test default threshold (0.5)
        self.assertEqual(cd.predicted_class(), 0)  # 0.3 < 0.5
        self.assertEqual(cd.predicted_class(0.5), 0)  # 0.3 < 0.5
        
        # Test lower threshold
        self.assertEqual(cd.predicted_class(0.2), 1)  # 0.3 >= 0.2
        self.assertEqual(cd.predicted_class(0.3), 1)  # 0.3 >= 0.3 (edge case)
        
        # Test higher threshold
        self.assertEqual(cd.predicted_class(0.7), 0)  # 0.3 < 0.7
        self.assertEqual(cd.predicted_class(0.9), 0)  # 0.3 < 0.9
        
        # Test extreme thresholds
        self.assertEqual(cd.predicted_class(0.0), 1)  # Always predict 1
        self.assertEqual(cd.predicted_class(1.0), 0)  # Only predict 1 if P(y=1) = 1.0
    
    def test_predicted_class_edge_cases(self):
        """Test predicted_class with edge cases."""
        # Pure class 0
        cd_pure0 = ClassDistribution(y0_count=100, y1_count=0)
        self.assertEqual(cd_pure0.predicted_class(0.0), 1)  # P(y=1) = 0.0 >= 0.0
        self.assertEqual(cd_pure0.predicted_class(0.5), 0)
        self.assertEqual(cd_pure0.predicted_class(1.0), 0)
        
        # Pure class 1
        cd_pure1 = ClassDistribution(y0_count=0, y1_count=100)
        self.assertEqual(cd_pure1.predicted_class(0.0), 1)
        self.assertEqual(cd_pure1.predicted_class(0.5), 1)
        self.assertEqual(cd_pure1.predicted_class(1.0), 1)  # P(y=1) = 1.0 >= 1.0
        
        # Empty distribution
        cd_empty = ClassDistribution(y0_count=0, y1_count=0)
        self.assertEqual(cd_empty.predicted_class(0.5), 0)  # Default to 0 when empty
    
    def test_to_dict_json_serialization(self):
        """Test to_dict method for JSON serialization."""
        # Test normal distribution
        cd = ClassDistribution(y0_count=75, y1_count=25)
        result = cd.to_dict()
        
        expected = {
            'y0_count': 75,
            'y1_count': 25,
            'total_count': 100,
            'y0_probability': 0.75,
            'y1_probability': 0.25,
            'confidence': 0.75
        }
        
        self.assertEqual(result, expected)
        
        # Test edge case - empty distribution
        cd_empty = ClassDistribution(y0_count=0, y1_count=0)
        result_empty = cd_empty.to_dict()
        
        expected_empty = {
            'y0_count': 0,
            'y1_count': 0,
            'total_count': 0,
            'y0_probability': 0.0,
            'y1_probability': 0.0,
            'confidence': 0.0
        }
        
        self.assertEqual(result_empty, expected_empty)
        
        # Test that result is JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)
        
        # Test that we can deserialize it back
        parsed = json.loads(json_str)
        self.assertEqual(parsed, expected)
    
    def test_string_representation(self):
        """Test string representation includes probability."""
        cd = ClassDistribution(y0_count=75, y1_count=25)
        str_repr = str(cd)
        self.assertIn("y0=75", str_repr)
        self.assertIn("y1=25", str_repr)
        self.assertIn("P(y=1)=0.250", str_repr)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestTreeNode(unittest.TestCase):
    """Test TreeNode functionality including threshold support."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.universe_size = 100
        
    def test_constructor_parameters(self):
        """Test TreeNode constructor with all parameters."""
        # Create mock sketches
        y0_sketch = SketchFactory.create_sketch('bitvector', {0, 1, 2}, universe_size=10)
        y1_sketch = SketchFactory.create_sketch('bitvector', {3, 4}, universe_size=10)
        class_dist = ClassDistribution(y0_count=3, y1_count=2)
        
        # Test all constructor parameters
        node = TreeNode(
            feature="income=high",
            is_leaf=False,
            class_distribution=class_dist,
            depth=2,
            node_id=5,
            y0_sketch=y0_sketch,
            y1_sketch=y1_sketch
        )
        
        self.assertEqual(node.feature, "income=high")
        self.assertEqual(node.split_feature_name, "income")
        self.assertEqual(node.split_feature_value, "high")
        self.assertFalse(node.is_leaf)
        self.assertEqual(node.depth, 2)
        self.assertEqual(node.node_id, 5)
        self.assertIsNotNone(node.y0_sketch)
        self.assertIsNotNone(node.y1_sketch)
        self.assertEqual(node.get_data_count(), 5)
    
    def test_leaf_node_prediction_with_threshold(self):
        """Test leaf node predictions with different thresholds."""
        # Create leaf with P(y=1) = 0.4
        class_dist = ClassDistribution(y0_count=60, y1_count=40)
        leaf = TreeNode(
            is_leaf=True,
            class_distribution=class_dist,
            depth=3,
            node_id=7
        )
        
        sample = {"feature": "value"}  # Dummy sample
        
        # Test predictions with different thresholds
        self.assertEqual(leaf.predict_single(sample, threshold=0.5), 0)  # 0.4 < 0.5
        self.assertEqual(leaf.predict_single(sample, threshold=0.3), 1)  # 0.4 >= 0.3
        self.assertEqual(leaf.predict_single(sample, threshold=0.4), 1)  # 0.4 >= 0.4
        
        # Test probability prediction (threshold doesn't affect this)
        proba = leaf.predict_proba_single(sample)
        self.assertAlmostEqual(proba[0], 0.6)
        self.assertAlmostEqual(proba[1], 0.4)
    
    def test_leaf_node_to_json(self):
        """Test leaf node JSON serialization."""
        class_dist = ClassDistribution(y0_count=60, y1_count=40)
        leaf = TreeNode(
            is_leaf=True,
            class_distribution=class_dist,
            depth=2,
            node_id=5
        )
        leaf.impurity = 0.6931  # entropy value
        
        # Test with default threshold
        json_dict = leaf.to_json(threshold=0.5)
        
        expected_keys = {'node_id', 'depth', 'is_leaf', 'samples', 'class_distribution', 
                        'predicted_class', 'impurity', 'type'}
        self.assertEqual(set(json_dict.keys()), expected_keys)
        
        # Check specific values
        self.assertEqual(json_dict['node_id'], 5)
        self.assertEqual(json_dict['depth'], 2)
        self.assertTrue(json_dict['is_leaf'])
        self.assertEqual(json_dict['samples'], 100)
        self.assertEqual(json_dict['type'], 'leaf')
        self.assertEqual(json_dict['predicted_class'], 0)  # 0.4 < 0.5
        self.assertAlmostEqual(json_dict['impurity'], 0.6931, places=4)
        
        # Check class distribution structure
        class_dist_dict = json_dict['class_distribution']
        self.assertEqual(class_dist_dict['y0_count'], 60)
        self.assertEqual(class_dist_dict['y1_count'], 40)
        self.assertAlmostEqual(class_dist_dict['y1_probability'], 0.4)
        
        # Test with different threshold
        json_dict_03 = leaf.to_json(threshold=0.3)
        self.assertEqual(json_dict_03['predicted_class'], 1)  # 0.4 >= 0.3
        
        # Test JSON serializability
        json_str = json.dumps(json_dict)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['node_id'], 5)
    
    def test_internal_node_to_json(self):
        """Test internal node JSON serialization."""
        # Create children
        left_dist = ClassDistribution(y0_count=30, y1_count=70)
        left_leaf = TreeNode(
            is_leaf=True, 
            class_distribution=left_dist, 
            depth=2, 
            node_id=3
        )
        
        right_dist = ClassDistribution(y0_count=70, y1_count=30)
        right_leaf = TreeNode(
            is_leaf=True, 
            class_distribution=right_dist, 
            depth=2, 
            node_id=4
        )
        
        # Create internal node
        root_dist = ClassDistribution(y0_count=100, y1_count=100)
        root = TreeNode(
            feature="income=high",
            is_leaf=False,
            class_distribution=root_dist,
            depth=1,
            node_id=1
        )
        root.split_gain = 0.3219
        root.impurity = 1.0
        root.set_children(left_leaf, right_leaf)
        
        # Get JSON representation
        json_dict = root.to_json(threshold=0.5)
        
        # Check root node structure
        expected_keys = {'node_id', 'depth', 'is_leaf', 'samples', 'class_distribution',
                        'predicted_class', 'impurity', 'type', 'feature', 'split_feature_name',
                        'split_feature_value', 'split_gain', 'left_child', 'right_child'}
        self.assertEqual(set(json_dict.keys()), expected_keys)
        
        # Check root values
        self.assertEqual(json_dict['type'], 'internal')
        self.assertEqual(json_dict['feature'], 'income=high')
        self.assertEqual(json_dict['split_feature_name'], 'income')
        self.assertEqual(json_dict['split_feature_value'], 'high')
        self.assertAlmostEqual(json_dict['split_gain'], 0.3219, places=4)
        
        # Check children exist
        self.assertIn('left_child', json_dict)
        self.assertIn('right_child', json_dict)
        
        # Check left child
        left_child = json_dict['left_child']
        self.assertEqual(left_child['node_id'], 3)
        self.assertEqual(left_child['type'], 'leaf')
        self.assertEqual(left_child['predicted_class'], 1)  # P(y=1) = 0.7 >= 0.5
        
        # Check right child
        right_child = json_dict['right_child']
        self.assertEqual(right_child['node_id'], 4)
        self.assertEqual(right_child['type'], 'leaf')
        self.assertEqual(right_child['predicted_class'], 0)  # P(y=1) = 0.3 < 0.5
        
        # Test JSON serializability
        json_str = json.dumps(json_dict)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['feature'], 'income=high')
        self.assertEqual(parsed['left_child']['node_id'], 3)
    
    def test_json_threshold_propagation(self):
        """Test that threshold is properly propagated to children in JSON."""
        # Create tree structure
        left_dist = ClassDistribution(y0_count=40, y1_count=60)  # P(y=1) = 0.6
        left_leaf = TreeNode(is_leaf=True, class_distribution=left_dist, node_id=2)
        
        right_dist = ClassDistribution(y0_count=80, y1_count=20)  # P(y=1) = 0.2
        right_leaf = TreeNode(is_leaf=True, class_distribution=right_dist, node_id=3)
        
        root = TreeNode(feature="test=A", node_id=1)
        root.set_children(left_leaf, right_leaf)
        
        # Test with threshold 0.5
        json_05 = root.to_json(threshold=0.5)
        self.assertEqual(json_05['left_child']['predicted_class'], 1)   # 0.6 >= 0.5
        self.assertEqual(json_05['right_child']['predicted_class'], 0)  # 0.2 < 0.5
        
        # Test with threshold 0.1
        json_01 = root.to_json(threshold=0.1)
        self.assertEqual(json_01['left_child']['predicted_class'], 1)   # 0.6 >= 0.1
        self.assertEqual(json_01['right_child']['predicted_class'], 1)  # 0.2 >= 0.1
        
        # Test with threshold 0.8
        json_08 = root.to_json(threshold=0.8)
        self.assertEqual(json_08['left_child']['predicted_class'], 0)   # 0.6 < 0.8
        self.assertEqual(json_08['right_child']['predicted_class'], 0)  # 0.2 < 0.8
    
    def test_internal_node_traversal_with_threshold(self):
        """Test tree traversal with threshold propagation."""
        # Create a small tree
        #       Root (income=high)
        #      /                  \
        #   Left (P=0.7)      Right (P=0.3)
        
        left_dist = ClassDistribution(y0_count=30, y1_count=70)  # P(y=1) = 0.7
        left_leaf = TreeNode(is_leaf=True, class_distribution=left_dist)
        
        right_dist = ClassDistribution(y0_count=70, y1_count=30)  # P(y=1) = 0.3
        right_leaf = TreeNode(is_leaf=True, class_distribution=right_dist)
        
        root = TreeNode(feature="income=high", is_leaf=False)
        root.set_children(left_leaf, right_leaf)
        
        # Test predictions with different samples and thresholds
        high_income = {"income": "high"}
        low_income = {"income": "low"}
        
        # High income goes left (P=0.7)
        self.assertEqual(root.predict_single(high_income, threshold=0.5), 1)  # 0.7 >= 0.5
        self.assertEqual(root.predict_single(high_income, threshold=0.8), 0)  # 0.7 < 0.8
        
        # Low income goes right (P=0.3)
        self.assertEqual(root.predict_single(low_income, threshold=0.5), 0)  # 0.3 < 0.5
        self.assertEqual(root.predict_single(low_income, threshold=0.2), 1)  # 0.3 >= 0.2
        
        # Test probability predictions
        high_proba = root.predict_proba_single(high_income)
        self.assertAlmostEqual(high_proba[0], 0.3)
        self.assertAlmostEqual(high_proba[1], 0.7)
        
        low_proba = root.predict_proba_single(low_income)
        self.assertAlmostEqual(low_proba[0], 0.7)
        self.assertAlmostEqual(low_proba[1], 0.3)
    
    def test_feature_condition_checking(self):
        """Test _check_feature_condition method."""
        node = TreeNode(feature="city=Mumbai")
        
        # Exact match
        self.assertTrue(node._check_feature_condition({"city": "Mumbai"}))
        
        # No match
        self.assertFalse(node._check_feature_condition({"city": "Delhi"}))
        self.assertFalse(node._check_feature_condition({"city": "Bangalore"}))
        
        # Missing feature
        self.assertFalse(node._check_feature_condition({"state": "Maharashtra"}))
        self.assertFalse(node._check_feature_condition({}))
        
        # Node without feature
        empty_node = TreeNode()
        self.assertFalse(empty_node._check_feature_condition({"city": "Mumbai"}))
    
    def test_tree_structure_methods(self):
        """Test tree structure methods."""
        # Build a tree
        #         Root
        #        /    \
        #     Node1   Leaf1
        #     /   \
        #  Leaf2  Leaf3
        
        # Create leaves with proper depths
        leaf1 = TreeNode(is_leaf=True, class_distribution=ClassDistribution(10, 5), depth=1)
        leaf2 = TreeNode(is_leaf=True, class_distribution=ClassDistribution(20, 10), depth=2)
        leaf3 = TreeNode(is_leaf=True, class_distribution=ClassDistribution(15, 25), depth=2)
        
        node1 = TreeNode(feature="feature1=A", depth=1)
        node1.set_children(leaf2, leaf3)
        
        root = TreeNode(feature="feature2=B", depth=0)
        root.set_children(node1, leaf1)
        
        # Test leaf count
        self.assertEqual(root.get_leaf_count(), 3)
        self.assertEqual(node1.get_leaf_count(), 2)
        self.assertEqual(leaf1.get_leaf_count(), 1)
        
        # Test max depth
        self.assertEqual(root.get_max_depth(), 2)  # Depth of leaf2/leaf3
        self.assertEqual(node1.get_max_depth(), 2)
        self.assertEqual(leaf1.get_max_depth(), 1)
    
    def test_print_tree_with_threshold(self):
        """Test print_tree with threshold parameter."""
        # Create a simple tree
        left_dist = ClassDistribution(y0_count=40, y1_count=60)  # P(y=1) = 0.6
        left_leaf = TreeNode(is_leaf=True, class_distribution=left_dist, depth=1)
        
        right_dist = ClassDistribution(y0_count=80, y1_count=20)  # P(y=1) = 0.2
        right_leaf = TreeNode(is_leaf=True, class_distribution=right_dist, depth=1)
        
        root = TreeNode(feature="test=yes", is_leaf=False, depth=0)
        root.split_gain = 0.5
        root.set_children(left_leaf, right_leaf)
        
        # Capture print output
        import io
        from contextlib import redirect_stdout
        
        # Test with threshold 0.5
        output = io.StringIO()
        with redirect_stdout(output):
            root.print_tree(threshold=0.5)
        
        output_str = output.getvalue()
        self.assertIn("test=yes", output_str)
        self.assertIn("pred=1", output_str)  # Left leaf: 0.6 >= 0.5
        self.assertIn("pred=0", output_str)  # Right leaf: 0.2 < 0.5
        
        # Test with threshold 0.7
        output = io.StringIO()
        with redirect_stdout(output):
            root.print_tree(threshold=0.7)
        
        output_str = output.getvalue()
        # Both should predict 0 with threshold 0.7
        self.assertEqual(output_str.count("pred=0"), 2)  # Both leaves predict 0
    
    def test_node_metadata(self):
        """Test node metadata attributes."""
        node = TreeNode(feature="age=young")
        node.split_gain = 0.75
        node.impurity = 0.45
        
        self.assertEqual(node.split_gain, 0.75)
        self.assertEqual(node.impurity, 0.45)
        
        # Test get_data_sketches
        y0_sketch = SketchFactory.create_sketch('bitvector', {0, 1}, universe_size=10)
        y1_sketch = SketchFactory.create_sketch('bitvector', {2, 3}, universe_size=10)
        
        node_with_sketches = TreeNode(y0_sketch=y0_sketch, y1_sketch=y1_sketch)
        sketches = node_with_sketches.get_data_sketches()
        self.assertEqual(sketches[0], y0_sketch)
        self.assertEqual(sketches[1], y1_sketch)
        
        node_without_sketches = TreeNode()
        empty_sketches = node_without_sketches.get_data_sketches()
        self.assertIsNone(empty_sketches[0])
        self.assertIsNone(empty_sketches[1])
    
    def test_make_leaf(self):
        """Test converting node to leaf."""
        node = TreeNode(feature="test=A", is_leaf=False)
        left = TreeNode(is_leaf=True, class_distribution=ClassDistribution(10, 5))
        right = TreeNode(is_leaf=True, class_distribution=ClassDistribution(5, 10))
        node.set_children(left, right)
        
        # Convert to leaf
        new_dist = ClassDistribution(y0_count=100, y1_count=50)
        node.make_leaf(new_dist)
        
        self.assertTrue(node.is_leaf)
        self.assertEqual(node.class_distribution, new_dist)
        self.assertIsNone(node.left_child)
        self.assertIsNone(node.right_child)
        self.assertIsNone(node.feature)
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        # Leaf node
        leaf = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(50, 30),
            depth=2
        )
        str_repr = str(leaf)
        self.assertIn("LeafNode", str_repr)
        self.assertIn("depth=2", str_repr)
        self.assertEqual(repr(leaf), str_repr)
        
        # Internal node
        internal = TreeNode(
            feature="income=high",
            is_leaf=False,
            depth=1
        )
        internal.set_children(leaf, leaf)
        
        str_repr = str(internal)
        self.assertIn("InternalNode", str_repr)
        self.assertIn("depth=1", str_repr)
        self.assertIn("feature=income=high", str_repr)
        self.assertIn("children=True", str_repr)
        self.assertEqual(repr(internal), str_repr)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")  
class TestTreeNodeIntegration(unittest.TestCase):
    """Integration tests for TreeNode with realistic scenarios."""
    
    def test_complex_tree_traversal(self):
        """Test traversal of a more complex tree with thresholds."""
        # Build a decision tree:
        #                Root (income=high)
        #               /                   \
        #        Node1 (age=young)      Leaf1 (P=0.8)
        #        /            \
        #   Leaf2 (P=0.3)   Leaf3 (P=0.6)
        
        # Create leaves with different probabilities
        leaf1 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=20, y1_count=80),  # P(y=1) = 0.8
            depth=1
        )
        
        leaf2 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=70, y1_count=30),  # P(y=1) = 0.3
            depth=2
        )
        
        leaf3 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=40, y1_count=60),  # P(y=1) = 0.6
            depth=2
        )
        
        # Create internal nodes
        node1 = TreeNode(feature="age=young", is_leaf=False, depth=1)
        node1.set_children(leaf2, leaf3)
        
        root = TreeNode(feature="income=high", is_leaf=False, depth=0)
        root.set_children(node1, leaf1)
        
        # Test different prediction paths with various thresholds
        test_cases = [
            # (sample, threshold, expected_pred)
            ({"income": "high", "age": "young"}, 0.5, 0),  # Goes to leaf2, P=0.3 < 0.5
            ({"income": "high", "age": "young"}, 0.2, 1),  # Goes to leaf2, P=0.3 >= 0.2
            ({"income": "high", "age": "old"}, 0.5, 1),    # Goes to leaf3, P=0.6 >= 0.5
            ({"income": "high", "age": "old"}, 0.7, 0),    # Goes to leaf3, P=0.6 < 0.7
            ({"income": "low"}, 0.5, 1),                    # Goes to leaf1, P=0.8 >= 0.5
            ({"income": "low"}, 0.9, 0),                    # Goes to leaf1, P=0.8 < 0.9
        ]
        
        for sample, threshold, expected in test_cases:
            pred = root.predict_single(sample, threshold=threshold)
            self.assertEqual(pred, expected, 
                           f"Failed for sample={sample}, threshold={threshold}")
    
    def test_complex_tree_json_structure(self):
        """Test JSON representation of complex tree structure."""
        # Build the same complex tree as above
        leaf1 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=20, y1_count=80),
            depth=1,
            node_id=3
        )
        
        leaf2 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=70, y1_count=30),
            depth=2,
            node_id=4
        )
        
        leaf3 = TreeNode(
            is_leaf=True,
            class_distribution=ClassDistribution(y0_count=40, y1_count=60),
            depth=2,
            node_id=5
        )
        
        node1 = TreeNode(
            feature="age=young", 
            is_leaf=False, 
            depth=1,
            node_id=2,
            class_distribution=ClassDistribution(y0_count=110, y1_count=90)
        )
        node1.set_children(leaf2, leaf3)
        
        root = TreeNode(
            feature="income=high", 
            is_leaf=False, 
            depth=0,
            node_id=1,
            class_distribution=ClassDistribution(y0_count=130, y1_count=170)
        )
        root.set_children(node1, leaf1)
        
        # Get JSON representation
        json_dict = root.to_json(threshold=0.5)
        
        # Test overall structure
        self.assertEqual(json_dict['type'], 'internal')
        self.assertEqual(json_dict['feature'], 'income=high')
        self.assertIn('left_child', json_dict)
        self.assertIn('right_child', json_dict)
        
        # Test left child (node1)
        left_child = json_dict['left_child']
        self.assertEqual(left_child['type'], 'internal')
        self.assertEqual(left_child['feature'], 'age=young')
        self.assertEqual(left_child['node_id'], 2)
        
        # Test left child's children
        self.assertIn('left_child', left_child)  # leaf2
        self.assertIn('right_child', left_child)  # leaf3
        
        leaf2_json = left_child['left_child']
        self.assertEqual(leaf2_json['type'], 'leaf')
        self.assertEqual(leaf2_json['node_id'], 4)
        self.assertEqual(leaf2_json['predicted_class'], 0)  # P(y=1) = 0.3 < 0.5
        
        leaf3_json = left_child['right_child']
        self.assertEqual(leaf3_json['type'], 'leaf')
        self.assertEqual(leaf3_json['node_id'], 5)
        self.assertEqual(leaf3_json['predicted_class'], 1)  # P(y=1) = 0.6 >= 0.5
        
        # Test right child (leaf1)
        right_child = json_dict['right_child']
        self.assertEqual(right_child['type'], 'leaf')
        self.assertEqual(right_child['node_id'], 3)
        self.assertEqual(right_child['predicted_class'], 1)  # P(y=1) = 0.8 >= 0.5
        
        # Test JSON serializability of complex structure
        json_str = json.dumps(json_dict, indent=2)
        self.assertIsInstance(json_str, str)
        
        # Test that parsed JSON maintains structure
        parsed = json.loads(json_str)
        self.assertEqual(parsed['left_child']['left_child']['node_id'], 4)
        self.assertEqual(parsed['right_child']['predicted_class'], 1)
    
    def test_default_threshold_behavior(self):
        """Test that default threshold is 0.5."""
        class_dist = ClassDistribution(y0_count=45, y1_count=55)  # P(y=1) = 0.55
        leaf = TreeNode(is_leaf=True, class_distribution=class_dist)
        
        sample = {"dummy": "value"}
        
        # Default should be 0.5
        self.assertEqual(leaf.predict_single(sample), 1)  # 0.55 >= 0.5
        self.assertEqual(leaf.predict_single(sample, 0.5), 1)  # Explicit 0.5


if __name__ == '__main__':
    unittest.main(verbosity=2)
