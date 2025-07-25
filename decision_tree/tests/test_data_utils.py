import unittest
import os
import tempfile
import csv

# Import the classes we're testing
import sys
import os
# Add parent directory to path so we can import sketches module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sketches import DataSketch, SketchFactory
    from data_utils import parse_csv_input, validate_sketches
except ImportError:
    # If sketches.py is in same directory as test file
    from sketches import DataSketch, SketchFactory
    from data_utils import parse_csv_input, validate_sketches


def create_sample_csv(output_path: str, universe_size: int = 10, sketch_type: str = 'bitvector'):
    """
    Create a sample CSV file for testing purposes.
    
    Args:
        output_path (str): Path to create sample CSV
        universe_size (int): Size of universe for test sketches
        sketch_type (str): Type of sketch to create
    """
    
    print(f"Creating sample CSV: {output_path}")
    
    # Create sample sketches using factory
    feature1 = SketchFactory.create_sketch(sketch_type, {0, 2, 4, 6}, universe_size=universe_size)  # Even rows
    feature2 = SketchFactory.create_sketch(sketch_type, {1, 3, 5, 7}, universe_size=universe_size)  # Odd rows
    feature3 = SketchFactory.create_sketch(sketch_type, {0, 1, 2, 3}, universe_size=universe_size)  # First half
    
    target_class0 = SketchFactory.create_sketch(sketch_type, {0, 1, 4, 5}, universe_size=universe_size)  # 4 rows for class 0
    target_class1 = SketchFactory.create_sketch(sketch_type, {2, 3, 6, 7}, universe_size=universe_size)  # 4 rows for class 1
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['dim', 'type', 'base64_sketch'])
        
        # Features
        writer.writerow(['feature_A=true', 'feature', feature1.to_base64()])
        writer.writerow(['feature_B=true', 'feature', feature2.to_base64()])
        writer.writerow(['feature_C=true', 'feature', feature3.to_base64()])
        
        # Targets
        writer.writerow(['outcome=success', 'target', target_class0.to_base64()])
        writer.writerow(['outcome=failure', 'target', target_class1.to_base64()])
    
    print(f"Sample CSV created with {universe_size} rows and 5 sketches")
    return output_path


class TestDataUtils(unittest.TestCase):
    """Test CSV parsing functionality for decision tree input."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_csv = os.path.join(self.temp_dir, "test_sample.csv")
        
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_sample_csv(self):
        """Test creation of sample CSV file."""
        universe_size = 10
        create_sample_csv(self.sample_csv, universe_size)
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.sample_csv))
        
        # Verify file has expected content
        with open(self.sample_csv, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 6)  # Header + 5 data rows
            self.assertIn('dim,type,base64_sketch', lines[0])
    
    def test_parse_csv_basic(self):
        """Test basic CSV parsing functionality."""
        universe_size = 10
        sketch_type = 'bitvector'
        create_sample_csv(self.sample_csv, universe_size, sketch_type)
        
        # Parse the CSV
        features, targets = parse_csv_input(self.sample_csv, sketch_type=sketch_type, universe_size=universe_size)
        
        # Verify feature parsing
        self.assertEqual(len(features), 3)  # 3 features in sample
        self.assertIn('feature_A=true', features)
        self.assertIn('feature_B=true', features)  
        self.assertIn('feature_C=true', features)
        
        # Verify target parsing
        self.assertEqual(len(targets), 2)  # Binary classification
        self.assertIn('y=0', targets)
        self.assertIn('y=1', targets)
        
        # Verify DataSketch counts
        self.assertEqual(features['feature_A=true'].get_count(), 4)  # Even rows: 0,2,4,6
        self.assertEqual(features['feature_B=true'].get_count(), 4)  # Odd rows: 1,3,5,7
        self.assertEqual(features['feature_C=true'].get_count(), 4)  # First half: 0,1,2,3
        
        self.assertEqual(targets['y=0'].get_count(), 4)  # Class 0: 0,1,4,5
        self.assertEqual(targets['y=1'].get_count(), 4)  # Class 1: 2,3,6,7
    
    def test_target_mapping(self):
        """Test that target values are correctly mapped to y=0, y=1."""
        universe_size = 10
        sketch_type = 'bitvector'
        create_sample_csv(self.sample_csv, universe_size, sketch_type)
        
        features, targets = parse_csv_input(self.sample_csv, sketch_type=sketch_type, universe_size=universe_size)
        
        # The sample creates:
        # outcome=success -> should become y=0 (first target)
        # outcome=failure -> should become y=1 (second target)
        
        # Verify specific row mappings
        expected_y0 = SketchFactory.create_sketch(sketch_type, {0, 1, 4, 5}, universe_size=universe_size)
        expected_y1 = SketchFactory.create_sketch(sketch_type, {2, 3, 6, 7}, universe_size=universe_size)
        
        self.assertEqual(targets['y=0'], expected_y0)
        self.assertEqual(targets['y=1'], expected_y1)
    
    def test_disjoint_targets(self):
        """Test that target classes are disjoint."""
        universe_size = 10
        sketch_type = 'bitvector'
        create_sample_csv(self.sample_csv, universe_size, sketch_type)
        
        features, targets = parse_csv_input(self.sample_csv, sketch_type=sketch_type, universe_size=universe_size)
        
        # Check that y=0 and y=1 don't overlap
        overlap = targets['y=0'].intersect(targets['y=1'])
        self.assertEqual(overlap.get_count(), 0, "Target classes should not overlap")
    
    def test_invalid_csv_missing_file(self):
        """Test error handling for missing CSV file."""
        with self.assertRaises(FileNotFoundError):
            parse_csv_input("nonexistent_file.csv", sketch_type='bitvector')
    
    def test_invalid_csv_wrong_columns(self):
        """Test error handling for CSV with wrong columns."""
        # Create CSV with wrong headers
        wrong_csv = os.path.join(self.temp_dir, "wrong.csv")
        with open(wrong_csv, 'w') as f:
            f.write("wrong,headers,here\n")
            f.write("data1,data2,data3\n")
        
        with self.assertRaises(ValueError) as cm:
            parse_csv_input(wrong_csv, sketch_type='bitvector')
        
        self.assertIn("CSV must have columns", str(cm.exception))
    
    def test_invalid_target_count(self):
        """Test error handling for wrong number of target values."""
        # Create CSV with 3 target values (should be exactly 2)
        invalid_csv = os.path.join(self.temp_dir, "invalid.csv")
        universe_size = 5
        sketch_type = 'bitvector'
        
        target1 = SketchFactory.create_sketch(sketch_type, {0, 1}, universe_size=universe_size)
        target2 = SketchFactory.create_sketch(sketch_type, {2, 3}, universe_size=universe_size)
        target3 = SketchFactory.create_sketch(sketch_type, {4}, universe_size=universe_size)
        
        with open(invalid_csv, 'w') as f:
            f.write("dim,type,base64_sketch\n")
            f.write(f"outcome=good,target,{target1.to_base64()}\n")
            f.write(f"outcome=bad,target,{target2.to_base64()}\n")
            f.write(f"outcome=neutral,target,{target3.to_base64()}\n")  # Third target - invalid
        
        with self.assertRaises(ValueError) as cm:
            parse_csv_input(invalid_csv, sketch_type=sketch_type, universe_size=universe_size)
        
        self.assertIn("Target must have exactly 2 values", str(cm.exception))
    
    def test_validate_sketches(self):
        """Test sketch validation functionality."""
        universe_size = 10
        sketch_type = 'bitvector'
        create_sample_csv(self.sample_csv, universe_size, sketch_type)
        
        features, targets = parse_csv_input(self.sample_csv, sketch_type=sketch_type, universe_size=universe_size)
        stats = validate_sketches(features, targets)
        
        # Verify statistics
        self.assertEqual(stats['total_features'], 3)
        self.assertEqual(stats['total_targets'], 2)
        self.assertEqual(stats['y0_count'], 4)
        self.assertEqual(stats['y1_count'], 4)
        self.assertEqual(stats['total_rows'], 8)
        self.assertEqual(stats['class_balance'], 0.5)  # 50% class 1
        self.assertEqual(stats['target_overlap'], 0)   # No overlap
    
    def test_empty_sketches(self):
        """Test handling of empty sketches."""
        empty_csv = os.path.join(self.temp_dir, "empty.csv")
        universe_size = 5
        sketch_type = 'bitvector'
        
        # Create feature with no rows and normal targets
        empty_feature = SketchFactory.create_sketch(sketch_type, universe_size=universe_size)  # Empty
        target1 = SketchFactory.create_sketch(sketch_type, {0, 1}, universe_size=universe_size)
        target2 = SketchFactory.create_sketch(sketch_type, {2, 3}, universe_size=universe_size)
        
        with open(empty_csv, 'w') as f:
            f.write("dim,type,base64_sketch\n")
            f.write(f"empty_feature=true,feature,{empty_feature.to_base64()}\n")
            f.write(f"outcome=class0,target,{target1.to_base64()}\n") 
            f.write(f"outcome=class1,target,{target2.to_base64()}\n")
        
        features, targets = parse_csv_input(empty_csv, sketch_type=sketch_type, universe_size=universe_size)
        
        # Should parse successfully
        self.assertEqual(len(features), 1)
        self.assertEqual(features['empty_feature=true'].get_count(), 0)
        
        # Validation should warn about empty feature
        stats = validate_sketches(features, targets)
        self.assertEqual(stats['total_features'], 1)


class TestDataUtilsIntegration(unittest.TestCase):
    """Integration tests with realistic data scenarios."""
    
    def test_realistic_scenario(self):
        """Test with a realistic decision tree scenario."""
        universe_size = 100
        sketch_type = 'bitvector'
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        try:
            # Create realistic features and targets using factory
            gender_male = SketchFactory.create_sketch(sketch_type, set(range(0, 50)), universe_size=universe_size)      # Men: rows 0-49
            age_young = SketchFactory.create_sketch(sketch_type, set(range(0, 30)) | set(range(50, 80)), universe_size=universe_size)  # Young: 0-29, 50-79
            income_high = SketchFactory.create_sketch(sketch_type, set(range(20, 70)), universe_size=universe_size)    # High income: 20-69
            
            # Binary target: approved/denied (disjoint)
            approved = SketchFactory.create_sketch(sketch_type, set(range(10, 60)), universe_size=universe_size)       # Approved: 10-59
            denied = SketchFactory.create_sketch(sketch_type, set(range(0, 10)) | set(range(60, 100)), universe_size=universe_size)  # Denied: 0-9, 60-99
            
            # Write CSV
            temp_csv.write("dim,type,base64_sketch\n")
            temp_csv.write(f"gender=male,feature,{gender_male.to_base64()}\n")
            temp_csv.write(f"age=young,feature,{age_young.to_base64()}\n")
            temp_csv.write(f"income=high,feature,{income_high.to_base64()}\n")
            temp_csv.write(f"loan=approved,target,{approved.to_base64()}\n")
            temp_csv.write(f"loan=denied,target,{denied.to_base64()}\n")
            temp_csv.close()
            
            # Parse and validate
            features, targets = parse_csv_input(temp_csv.name, sketch_type=sketch_type, universe_size=universe_size)
            stats = validate_sketches(features, targets)
            
            # Verify realistic scenario
            self.assertEqual(len(features), 3)
            self.assertEqual(stats['total_rows'], 100)
            self.assertEqual(stats['target_overlap'], 0)  # Should be disjoint
            self.assertEqual(targets['y=0'].get_count(), 50)  # loan=approved
            self.assertEqual(targets['y=1'].get_count(), 50)  # loan=denied
            
        finally:
            os.unlink(temp_csv.name)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestDataUtils, TestDataUtilsIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
