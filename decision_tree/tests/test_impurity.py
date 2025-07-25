import unittest
import math
from typing import Dict


# Import the classes we're testing
import sys
import os
# Add parent directory to path so we can import sketches module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_utils import parse_csv_input, validate_sketches
    from sketches import DataSketch, SketchFactory
    from impurity import ImpurityCalculator, ImpurityMethod
except ImportError:
    # If sketches.py is in same directory as test file
    from data_utils import parse_csv_input, validate_sketches
    from sketches import DataSketch, SketchFactory
    from impurity import ImpurityCalculator, ImpurityMethod

class TestImpurityCalculator(unittest.TestCase):
    """Test impurity calculations for decision tree."""
    
    def setUp(self):
        """Set up test fixtures with known data."""
        self.universe_size = 100
        self.sketch_type = 'bitvector'
        
        # Create test target sketches
        # y=0: rows 0-39 (40 instances)  
        # y=1: rows 40-79 (40 instances)
        # Total: 80 instances, balanced classes
        self.y0_sketch = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 40)), universe_size=self.universe_size
        )
        self.y1_sketch = SketchFactory.create_sketch(
            self.sketch_type, set(range(40, 80)), universe_size=self.universe_size
        )
        
        # Create test feature sketches
        # Perfect split: feature_perfect separates classes completely
        self.feature_perfect = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 40)), universe_size=self.universe_size  # Only y=0
        )
        
        # Balanced split: feature_balanced splits each class equally
        self.feature_balanced = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 20)) | set(range(40, 60)), universe_size=self.universe_size  # Half of each class
        )
        
        # Poor split: feature_poor provides no information
        self.feature_poor = SketchFactory.create_sketch(
            self.sketch_type, set(range(0, 10)) | set(range(40, 50)), universe_size=self.universe_size  # Same proportion from each
        )
        
        # Empty feature
        self.feature_empty = SketchFactory.create_sketch(
            self.sketch_type, set(), universe_size=self.universe_size
        )
    
    def test_entropy_pure_classes(self):
        """Test entropy calculation for pure classes."""
        # Pure class 0 (only y=0, no y=1)
        empty_y1 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        entropy = ImpurityCalculator.calculate_entropy(self.y0_sketch, empty_y1)
        self.assertAlmostEqual(entropy, 0.0, places=6, msg="Pure class should have entropy 0")
        
        # Pure class 1 (only y=1, no y=0)
        empty_y0 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        entropy = ImpurityCalculator.calculate_entropy(empty_y0, self.y1_sketch)
        self.assertAlmostEqual(entropy, 0.0, places=6, msg="Pure class should have entropy 0")
    
    def test_entropy_balanced_classes(self):
        """Test entropy calculation for balanced classes."""
        # Balanced classes (50-50 split) should have maximum entropy = 1.0
        entropy = ImpurityCalculator.calculate_entropy(self.y0_sketch, self.y1_sketch)
        expected_entropy = 1.0  # -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        self.assertAlmostEqual(entropy, expected_entropy, places=6)
    
    def test_entropy_imbalanced_classes(self):
        """Test entropy calculation for imbalanced classes."""
        # Create imbalanced classes: 75% class 0, 25% class 1
        y0_large = SketchFactory.create_sketch(self.sketch_type, set(range(0, 60)), universe_size=self.universe_size)  # 60 instances
        y1_small = SketchFactory.create_sketch(self.sketch_type, set(range(60, 80)), universe_size=self.universe_size)  # 20 instances
        
        entropy = ImpurityCalculator.calculate_entropy(y0_large, y1_small)
        
        # Manual calculation: p0=0.75, p1=0.25
        # entropy = -(0.75*log2(0.75) + 0.25*log2(0.25)) ≈ 0.811
        expected_entropy = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        self.assertAlmostEqual(entropy, expected_entropy, places=6)
    
    def test_gini_pure_classes(self):
        """Test Gini impurity for pure classes."""
        # Pure classes should have Gini = 0
        empty_y1 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        gini = ImpurityCalculator.calculate_gini(self.y0_sketch, empty_y1)
        self.assertAlmostEqual(gini, 0.0, places=6)
        
        empty_y0 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        gini = ImpurityCalculator.calculate_gini(empty_y0, self.y1_sketch)
        self.assertAlmostEqual(gini, 0.0, places=6)
    
    def test_gini_balanced_classes(self):
        """Test Gini impurity for balanced classes."""
        # Balanced classes (50-50) should have Gini = 0.5
        gini = ImpurityCalculator.calculate_gini(self.y0_sketch, self.y1_sketch)
        expected_gini = 1.0 - (0.5**2 + 0.5**2)  # 1 - (0.25 + 0.25) = 0.5
        self.assertAlmostEqual(gini, expected_gini, places=6)
    
    def test_gini_imbalanced_classes(self):
        """Test Gini impurity for imbalanced classes."""
        # Create imbalanced: 75% class 0, 25% class 1
        y0_large = SketchFactory.create_sketch(self.sketch_type, set(range(0, 60)), universe_size=self.universe_size)
        y1_small = SketchFactory.create_sketch(self.sketch_type, set(range(60, 80)), universe_size=self.universe_size)
        
        gini = ImpurityCalculator.calculate_gini(y0_large, y1_small)
        
        # Manual calculation: gini = 1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375
        expected_gini = 1.0 - (0.75**2 + 0.25**2)
        self.assertAlmostEqual(gini, expected_gini, places=6)
    
    def test_feature_split_counts_perfect(self):
        """Test split count calculation for perfect feature."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_perfect, self.y0_sketch, self.y1_sketch
        )
        
        # Perfect feature should put all y=0 on left, all y=1 on right
        self.assertEqual(left_y0, 40, "All class 0 should be on left")
        self.assertEqual(left_y1, 0, "No class 1 should be on left") 
        self.assertEqual(right_y0, 0, "No class 0 should be on right")
        self.assertEqual(right_y1, 40, "All class 1 should be on right")
    
    def test_feature_split_counts_balanced(self):
        """Test split count calculation for balanced feature."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_balanced, self.y0_sketch, self.y1_sketch
        )
        
        # Balanced feature splits each class equally
        self.assertEqual(left_y0, 20, "Half of class 0 on left")
        self.assertEqual(left_y1, 20, "Half of class 1 on left")
        self.assertEqual(right_y0, 20, "Half of class 0 on right") 
        self.assertEqual(right_y1, 20, "Half of class 1 on right")
    
    def test_feature_split_counts_empty(self):
        """Test split count calculation for empty feature."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_empty, self.y0_sketch, self.y1_sketch
        )
        
        # Empty feature puts nothing on left, everything on right
        self.assertEqual(left_y0, 0)
        self.assertEqual(left_y1, 0)
        self.assertEqual(right_y0, 40)
        self.assertEqual(right_y1, 40)
    
    def test_information_gain_perfect_split(self):
        """Test information gain for perfect split."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_perfect, self.y0_sketch, self.y1_sketch
        )
        
        gain = ImpurityCalculator.calculate_information_gain(
            self.y0_sketch, self.y1_sketch,
            left_y0, left_y1, right_y0, right_y1,
            ImpurityMethod.ENTROPY
        )
        
        # Perfect split should achieve maximum information gain
        # Parent entropy = 1.0 (balanced), child entropy = 0.0 (pure)
        # Information gain = 1.0 - 0.0 = 1.0
        self.assertAlmostEqual(gain, 1.0, places=6, msg="Perfect split should have maximum gain")
    
    def test_information_gain_no_split(self):
        """Test information gain when feature provides no information."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_poor, self.y0_sketch, self.y1_sketch
        )
        
        gain = ImpurityCalculator.calculate_information_gain(
            self.y0_sketch, self.y1_sketch,
            left_y0, left_y1, right_y0, right_y1,
            ImpurityMethod.ENTROPY
        )
        
        # Poor feature maintains same class distribution on both sides
        # Should provide no information gain (gain ≈ 0)
        self.assertAlmostEqual(gain, 0.0, places=6, msg="Poor split should provide no gain")
    
    def test_information_gain_gini(self):
        """Test information gain calculation using Gini impurity."""
        left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
            self.feature_perfect, self.y0_sketch, self.y1_sketch
        )
        
        gain = ImpurityCalculator.calculate_information_gain(
            self.y0_sketch, self.y1_sketch,
            left_y0, left_y1, right_y0, right_y1,
            ImpurityMethod.GINI
        )
        
        # Perfect split with Gini: parent=0.5, children=0.0
        # Information gain = 0.5 - 0.0 = 0.5
        self.assertAlmostEqual(gain, 0.5, places=6)
    
    def test_evaluate_all_features(self):
        """Test evaluation of multiple features."""
        features = {
            'perfect_feature=true': self.feature_perfect,
            'balanced_feature=true': self.feature_balanced,
            'poor_feature=true': self.feature_poor,
            'empty_feature=true': self.feature_empty
        }
        
        gains = ImpurityCalculator.evaluate_all_features(
            features, self.y0_sketch, self.y1_sketch, ImpurityMethod.ENTROPY
        )
        
        # Verify we got gains for all features
        self.assertEqual(len(gains), 4)
        
        # Perfect feature should have highest gain
        self.assertAlmostEqual(gains['perfect_feature=true'], 1.0, places=6)
        
        # Poor feature should have lowest gain (≈0)
        self.assertLess(gains['poor_feature=true'], 0.1)
        
        # Empty feature should have zero gain
        self.assertAlmostEqual(gains['empty_feature=true'], 0.0, places=6)
    
    def test_find_best_split(self):
        """Test finding the best feature split."""
        features = {
            'medium_feature=true': self.feature_balanced, 
            'perfect_feature=true': self.feature_perfect,
            'poor_feature=true': self.feature_poor
        }
        
        best_feature, best_gain = ImpurityCalculator.find_best_split(
            features, self.y0_sketch, self.y1_sketch, ImpurityMethod.ENTROPY
        )
        
        # Perfect feature should be selected
        self.assertEqual(best_feature, 'perfect_feature=true')
        self.assertAlmostEqual(best_gain, 1.0, places=6)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty dataset
        empty_y0 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        empty_y1 = SketchFactory.create_sketch(self.sketch_type, set(), universe_size=self.universe_size)
        
        entropy = ImpurityCalculator.calculate_entropy(empty_y0, empty_y1)
        self.assertEqual(entropy, 0.0)
        
        gini = ImpurityCalculator.calculate_gini(empty_y0, empty_y1)
        self.assertEqual(gini, 0.0)
        
        # No features for best split
        with self.assertRaises(ValueError):
            ImpurityCalculator.find_best_split({}, self.y0_sketch, self.y1_sketch)
    
    def test_calculate_impurity_method_dispatch(self):
        """Test the generic impurity calculation method."""
        # Test entropy method
        entropy = ImpurityCalculator.calculate_impurity(
            self.y0_sketch, self.y1_sketch, ImpurityMethod.ENTROPY
        )
        expected_entropy = ImpurityCalculator.calculate_entropy(self.y0_sketch, self.y1_sketch)
        self.assertAlmostEqual(entropy, expected_entropy, places=6)
        
        # Test gini method
        gini = ImpurityCalculator.calculate_impurity(
            self.y0_sketch, self.y1_sketch, ImpurityMethod.GINI
        )
        expected_gini = ImpurityCalculator.calculate_gini(self.y0_sketch, self.y1_sketch)
        self.assertAlmostEqual(gini, expected_gini, places=6)


class TestImpurityIntegration(unittest.TestCase):
    """Integration tests with realistic decision tree scenarios."""
    
    def test_loan_approval_scenario(self):
        """Test with realistic loan approval data."""
        universe_size = 1000
        sketch_type = 'bitvector'
        
        # Create realistic target distribution
        # Approved: 60% (rows 0-599)
        # Denied: 40% (rows 600-999)
        approved = SketchFactory.create_sketch(sketch_type, set(range(0, 600)), universe_size=universe_size)
        denied = SketchFactory.create_sketch(sketch_type, set(range(600, 1000)), universe_size=universe_size)
        
        # Create features with different predictive power
        # High income: strong predictor (80% of high income approved)
        high_income_approved = set(range(0, 400))  # 400 approved
        high_income_denied = set(range(600, 700))   # 100 denied  
        high_income = SketchFactory.create_sketch(sketch_type, high_income_approved | high_income_denied, universe_size=universe_size)
        
        # Young age: weak predictor (55% approved)
        young_approved = set(range(0, 275))    # 275 approved
        young_denied = set(range(600, 825))    # 225 denied
        young_age = SketchFactory.create_sketch(sketch_type, young_approved | young_denied, universe_size=universe_size)
        
        features = {
            'income=high': high_income,
            'age=young': young_age
        }
        
        # Evaluate features
        gains = ImpurityCalculator.evaluate_all_features(
            features, approved, denied, ImpurityMethod.ENTROPY
        )
        
        # High income should have higher gain than young age
        self.assertGreater(gains['income=high'], gains['age=young'], 
                          "Strong predictor should have higher information gain")
        
        # Find best split
        best_feature, best_gain = ImpurityCalculator.find_best_split(
            features, approved, denied, ImpurityMethod.ENTROPY
        )
        
        self.assertEqual(best_feature, 'income=high')
        self.assertGreater(best_gain, 0.1, "Best feature should provide meaningful gain")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestImpurityCalculator, TestImpurityIntegration]
    
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
