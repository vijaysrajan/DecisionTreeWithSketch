import unittest
from typing import Dict, List
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to import our classes
    from sketches import DataSketch, SketchFactory
    from nodes import TreeNode, ClassDistribution
    from tree import DecisionTreeClassifier
    IMPORTS_AVAILABLE = True
except ImportError:
    # Create mock classes for testing structure
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import classes. Tests will be skipped.")


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDecisionTreeBasic(unittest.TestCase):
    """Test basic decision tree functionality with simple scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.universe_size = 100
        self.sketch_type = 'bitvector'
        
        # Create balanced target classes
        # Class 0: rows 0-49, Class 1: rows 50-99
        self.y0_balanced = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        self.y1_balanced = SketchFactory.create_sketch(
            self.sketch_type, set(range(50, 100)), universe_size=self.universe_size
        )
        
        self.targets_balanced = {"y=0": self.y0_balanced, "y=1": self.y1_balanced}
    
    def test_perfect_single_feature_separation(self):
        """Test tree building with a single perfect feature."""
        # Perfect feature: separates classes completely
        # feature=true for all class 0, feature=false for all class 1
        perfect_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size  # Only class 0
        )
        
        features = {"income=high": perfect_feature}
        
        # Build tree
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
        tree.fit(features, self.targets_balanced)
        
        # Verify tree structure
        self.assertIsNotNone(tree.root)
        self.assertFalse(tree.root.is_leaf, "Root should not be leaf with perfect feature")
        self.assertEqual(tree.root.feature, "income=high")
        
        # Verify children are leaves (perfect separation)
        self.assertTrue(tree.root.left_child.is_leaf, "Left child should be leaf (pure)")
        self.assertTrue(tree.root.right_child.is_leaf, "Right child should be leaf (pure)")
        
        # Verify class predictions
        self.assertEqual(tree.root.left_child.class_distribution.predicted_class, 0, "Left child should predict class 0")
        self.assertEqual(tree.root.right_child.class_distribution.predicted_class, 1, "Right child should predict class 1")
        
        # Verify information gain
        self.assertAlmostEqual(tree.root.split_gain, 1.0, places=6, msg="Perfect split should have gain = 1.0")
    
    def test_single_sample_prediction(self):
        """Test prediction with single perfect feature."""
        # Create perfect feature
        perfect_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        features = {"income=high": perfect_feature}
        
        # Train tree
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(features, self.targets_balanced)
        
        # Test predictions
        test_samples = [
            {"income": "high"},    # Should predict class 0
            {"income": "low"},     # Should predict class 1  
            {"income": "medium"},  # Should predict class 1 (not high)
        ]
        
        predictions = tree.predict(test_samples)
        
        self.assertEqual(predictions[0], 0, "High income should predict class 0")
        self.assertEqual(predictions[1], 1, "Low income should predict class 1")
        self.assertEqual(predictions[2], 1, "Medium income should predict class 1")
    
    def test_probability_prediction(self):
        """Test probability prediction."""
        # Perfect feature
        perfect_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        features = {"income=high": perfect_feature}
        
        # Train tree
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(features, self.targets_balanced)
        
        # Test probability predictions
        test_samples = [{"income": "high"}, {"income": "low"}]
        probabilities = tree.predict_proba(test_samples)
        
        # High income should have P(class=0) = 1.0, P(class=1) = 0.0
        self.assertAlmostEqual(probabilities[0][0], 1.0, places=6)
        self.assertAlmostEqual(probabilities[0][1], 0.0, places=6)
        
        # Low income should have P(class=0) = 0.0, P(class=1) = 1.0
        self.assertAlmostEqual(probabilities[1][0], 0.0, places=6)
        self.assertAlmostEqual(probabilities[1][1], 1.0, places=6)
    
    def test_multiple_features_best_split(self):
        """Test that tree selects the best feature when multiple are available."""
        # Perfect feature (gain = 1.0)
        perfect_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        
        # Poor feature (balanced split, gain ≈ 0)
        poor_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 25)) | set(range(50, 75)), universe_size=self.universe_size
        )
        
        features = {
            "income=high": perfect_feature,
            "age=young": poor_feature
        }
        
        # Build tree
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        tree.fit(features, self.targets_balanced)
        
        # Tree should select the perfect feature for root split
        self.assertEqual(tree.root.feature, "income=high", "Should select perfect feature")
        self.assertAlmostEqual(tree.root.split_gain, 1.0, places=6)
    
    def test_max_depth_stopping_condition(self):
        """Test max_depth stopping condition."""
        # Create features that could split further
        feature1 = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        feature2 = SketchFactory.create_sketch(
            self.sketch_type, set(range(25, 75)), universe_size=self.universe_size
        )
        
        features = {"feature1=true": feature1, "feature2=true": feature2}
        
        # Build tree with max_depth=1
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        tree.fit(features, self.targets_balanced)
        
        # Tree should have depth 1
        self.assertEqual(tree.max_tree_depth, 1)
        
        # Root should split, children should be leaves
        self.assertFalse(tree.root.is_leaf)
        self.assertTrue(tree.root.left_child.is_leaf)
        self.assertTrue(tree.root.right_child.is_leaf)
    
    def test_min_samples_leaf_stopping_condition(self):
        """Test min_samples_leaf stopping condition."""
        # Create imbalanced classes for testing
        y0_small = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 10)), universe_size=self.universe_size  # 10 samples
        )
        y1_small = SketchFactory.create_sketch(
            self.sketch_type, set(range(10, 20)), universe_size=self.universe_size  # 10 samples
        )
        targets_small = {"y=0": y0_small, "y=1": y1_small}
        
        # Feature that would create very small leaves
        feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 5)), universe_size=self.universe_size  # Only 5 samples
        )
        features = {"feature=true": feature}
        
        # Build tree with min_samples_leaf=10
        tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
        tree.fit(features, targets_small)
        
        # Should not split (would create leaves < min_samples_leaf)
        self.assertTrue(tree.root.is_leaf, "Should create leaf due to min_samples_leaf constraint")
    
    def test_pure_node_stopping_condition(self):
        """Test stopping when node is pure."""
        # Create pure class distribution
        y0_pure = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        y1_empty = SketchFactory.create_sketch(
            self.sketch_type, set(), universe_size=self.universe_size  # Empty
        )
        targets_pure = {"y=0": y0_pure, "y=1": y1_empty}
        
        # Some feature (shouldn't matter)
        feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 25)), universe_size=self.universe_size
        )
        features = {"feature=true": feature}
        
        # Build tree
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(features, targets_pure)
        
        # Should create leaf immediately (pure node)
        self.assertTrue(tree.root.is_leaf, "Pure node should become leaf")
        self.assertEqual(tree.root.class_distribution.predicted_class, 0)
        self.assertAlmostEqual(tree.root.impurity, 0.0, places=6, msg="Pure node should have 0 impurity")


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDecisionTreeAdvanced(unittest.TestCase):
    """Test advanced decision tree functionality."""
    
    def setUp(self):
        """Set up more complex test scenarios."""
        self.universe_size = 200
        self.sketch_type = 'bitvector'
        
        # Create realistic imbalanced dataset
        # Class 0: 60 samples (rows 0-59)
        # Class 1: 40 samples (rows 60-99)
        self.y0_imbalanced = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 60)), universe_size=self.universe_size
        )
        self.y1_imbalanced = SketchFactory.create_sketch(
            self.sketch_type, set(range(60, 100)), universe_size=self.universe_size
        )
        self.targets_imbalanced = {"y=0": self.y0_imbalanced, "y=1": self.y1_imbalanced}
    
    def test_gini_vs_entropy_criterion(self):
        """Test that Gini and Entropy produce reasonable trees."""
        # Create feature with moderate predictive power
        feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 40)) | set(range(60, 80)), universe_size=self.universe_size
        )
        features = {"feature=true": feature}
        
        # Build trees with both criteria
        tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        tree_entropy.fit(features, self.targets_imbalanced)
        
        tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
        tree_gini.fit(features, self.targets_imbalanced)
        
        # Both should select the same feature (only one available)
        self.assertEqual(tree_entropy.root.feature, "feature=true")
        self.assertEqual(tree_gini.root.feature, "feature=true")
        
        # Gains should be different but both positive
        self.assertGreater(tree_entropy.root.split_gain, 0)
        self.assertGreater(tree_gini.root.split_gain, 0)
        self.assertNotAlmostEqual(tree_entropy.root.split_gain, tree_gini.root.split_gain, places=3)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Create features with different predictive power
        strong_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size  # Strong predictor
        )
        weak_feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 30)) | set(range(60, 75)), universe_size=self.universe_size  # Weak predictor
        )
        
        features = {
            "income=high": strong_feature,
            "age=young": weak_feature
        }
        
        # Build tree
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        tree.fit(features, self.targets_imbalanced)
        
        # Get feature importance
        importance = tree.get_feature_importance()
        
        # Strong feature should have higher importance
        self.assertIn("income", importance)
        self.assertIn("age", importance)
        self.assertGreater(importance["income"], importance["age"])
        
        # Importance should sum to approximately 1.0
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=6)
    
    def test_tree_structure_properties(self):
        """Test tree structure properties and metadata."""
        # Create multiple features for a more complex tree
        feature1 = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 30)) | set(range(60, 70)), universe_size=self.universe_size
        )
        feature2 = SketchFactory.create_sketch(
            self.sketch_type, set(range(10, 50)), universe_size=self.universe_size  
        )
        feature3 = SketchFactory.create_sketch(
            self.sketch_type, set(range(20, 40)) | set(range(80, 90)), universe_size=self.universe_size
        )
        
        features = {
            "feature1=true": feature1,
            "feature2=true": feature2,
            "feature3=true": feature3
        }
        
        # Build tree
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
        tree.fit(features, self.targets_imbalanced)
        
        # Verify metadata
        self.assertEqual(tree.n_samples, 100)
        self.assertEqual(tree.n_features, 3)
        self.assertGreater(tree.node_count, 1)  # Should have multiple nodes
        self.assertGreater(tree.leaf_count, 1)   # Should have multiple leaves
        self.assertLessEqual(tree.max_tree_depth, 3)  # Should respect max_depth
        self.assertGreater(tree.training_time, 0)  # Should record training time
    
    def test_no_features_edge_case(self):
        """Test behavior with no features."""
        features = {}
        
        tree = DecisionTreeClassifier(criterion='entropy')
        
        with self.assertRaises(ValueError):
            tree.fit(features, self.targets_imbalanced)
    
    def test_invalid_targets_edge_case(self):
        """Test behavior with invalid targets."""
        feature = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 50)), universe_size=self.universe_size
        )
        features = {"feature=true": feature}
        
        # Invalid targets (missing y=1)
        invalid_targets = {"y=0": self.y0_imbalanced}
        
        tree = DecisionTreeClassifier(criterion='entropy')
        
        with self.assertRaises(ValueError):
            tree.fit(features, invalid_targets)
    
    def test_prediction_before_training(self):
        """Test prediction before training raises error."""
        tree = DecisionTreeClassifier(criterion='entropy')
        
        with self.assertRaises(ValueError):
            tree.predict([{"feature": "value"}])
        
        with self.assertRaises(ValueError):
            tree.predict_proba([{"feature": "value"}])


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDecisionTreeIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""
    
    def test_loan_approval_scenario(self):
        """Test complete loan approval decision tree scenario."""
        universe_size = 1000
        sketch_type = 'bitvector'
        
        # Create realistic loan dataset
        # Approved: 600 loans (60%)
        # Denied: 400 loans (40%)
        approved = SketchFactory.create_sketch(sketch_type, set(range(0, 600)), universe_size=universe_size)
        denied = SketchFactory.create_sketch(sketch_type, set(range(600, 1000)), universe_size=universe_size)
        targets = {"y=0": approved, "y=1": denied}
        
        # Create predictive features
        # High income: 80% approval rate
        high_income_approved = set(range(0, 400))  # 400 approved
        high_income_denied = set(range(600, 700))   # 100 denied
        high_income = SketchFactory.create_sketch(sketch_type, high_income_approved | high_income_denied, universe_size=universe_size)
        
        # Good credit: 75% approval rate  
        good_credit_approved = set(range(100, 550))  # 450 approved
        good_credit_denied = set(range(700, 850))    # 150 denied
        good_credit = SketchFactory.create_sketch(sketch_type, good_credit_approved | good_credit_denied, universe_size=universe_size)
        
        # Low debt: 65% approval rate
        low_debt_approved = set(range(200, 590))   # 390 approved
        low_debt_denied = set(range(750, 950))     # 200 denied
        low_debt = SketchFactory.create_sketch(sketch_type, low_debt_approved | low_debt_denied, universe_size=universe_size)
        
        features = {
            "income=high": high_income,
            "credit=good": good_credit,
            "debt=low": low_debt
        }
        
        # Train decision tree
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=20)
        tree.fit(features, targets)
        
        # Verify tree was built successfully
        self.assertIsNotNone(tree.root)
        self.assertGreater(tree.leaf_count, 1)
        self.assertLessEqual(tree.max_tree_depth, 4)
        
        # Test predictions on realistic samples
        test_samples = [
            {"income": "high", "credit": "good", "debt": "low"},     # Should likely approve
            {"income": "low", "credit": "poor", "debt": "high"},    # Should likely deny
            {"income": "medium", "credit": "good", "debt": "medium"} # Mixed case
        ]
        
        predictions = tree.predict(test_samples)
        probabilities = tree.predict_proba(test_samples)
        
        # Verify predictions are reasonable
        self.assertIn(predictions[0], [0, 1])  # Valid prediction
        self.assertIn(predictions[1], [0, 1])  # Valid prediction  
        self.assertIn(predictions[2], [0, 1])  # Valid prediction
        
        # Verify probabilities are valid
        for prob in probabilities:
            self.assertAlmostEqual(prob[0] + prob[1], 1.0, places=6)
            self.assertGreaterEqual(prob[0], 0.0)
            self.assertLessEqual(prob[0], 1.0)
            self.assertGreaterEqual(prob[1], 0.0)
            self.assertLessEqual(prob[1], 1.0)
        
        # Test feature importance
        importance = tree.get_feature_importance()
        self.assertTrue(all(score >= 0 for score in importance.values()))
        
        # Print tree for visual inspection
        if hasattr(tree, 'print_tree'):
            print(f"\nLoan Approval Decision Tree:")
            tree.print_tree()


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDecisionTreeBasic,
        TestDecisionTreeAdvanced, 
        TestDecisionTreeIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
