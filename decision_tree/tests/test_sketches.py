import unittest
import base64
import pickle
from typing import Set

# Import the classes we're testing
import sys
import os
# Add parent directory to path so we can import sketches module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sketches import DataSketch, BitVector
except ImportError:
    # If sketches.py is in same directory as test file
    from sketches import DataSketch, BitVector

try:
    from bitarray import bitarray
    BITARRAY_AVAILABLE = True
except ImportError:
    BITARRAY_AVAILABLE = False



class TestDataSketchInterface(unittest.TestCase):
    """Test the DataSketch abstract base class interface."""
    
    def test_cannot_instantiate_abstract_class(self):
        """DataSketch should not be instantiable directly."""
        with self.assertRaises(TypeError):
            DataSketch()


class TestBitVectorSetMode(unittest.TestCase):
    """Test BitVector using set-based implementation (fallback mode)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create BitVectors without universe_size to force set mode
        self.empty_bv = BitVector()
        self.bv1 = BitVector({1, 3, 5, 7})
        self.bv2 = BitVector({3, 4, 5, 8})
        self.bv3 = BitVector({1, 2, 3})
    
    def test_empty_bitvector_creation(self):
        """Test creating empty BitVector."""
        bv = BitVector()
        self.assertEqual(bv.get_count(), 0)
        self.assertTrue(bv.is_empty())
        self.assertEqual(len(bv), 0)
    
    def test_bitvector_creation_with_row_ids(self):
        """Test creating BitVector with initial row IDs."""
        row_ids = {1, 3, 5, 7, 9}
        bv = BitVector(row_ids)
        self.assertEqual(bv.get_count(), 5)
        self.assertFalse(bv.is_empty())
        self.assertEqual(len(bv), 5)
    
    def test_add_row_id(self):
        """Test adding row IDs to BitVector."""
        bv = BitVector()
        bv.add_row_id(5)
        bv.add_row_id(10)
        bv.add_row_id(5)  # Should not duplicate
        
        self.assertEqual(bv.get_count(), 2)
    
    def test_intersection_basic(self):
        """Test basic intersection functionality."""
        result = self.bv1.intersect(self.bv2)
        
        # bv1: {1, 3, 5, 7}, bv2: {3, 4, 5, 8}
        # Expected intersection: {3, 5}
        self.assertEqual(result.get_count(), 2)
        
        # Verify the actual row IDs (access internal for testing)
        if hasattr(result, 'row_ids'):
            self.assertEqual(result.row_ids, {3, 5})
    
    def test_intersection_empty_result(self):
        """Test intersection that results in empty set."""
        bv_disjoint = BitVector({10, 20, 30})
        result = self.bv1.intersect(bv_disjoint)
        
        self.assertEqual(result.get_count(), 0)
        self.assertTrue(result.is_empty())
    
    def test_intersection_with_empty(self):
        """Test intersection with empty BitVector."""
        result = self.bv1.intersect(self.empty_bv)
        self.assertEqual(result.get_count(), 0)
        
        result2 = self.empty_bv.intersect(self.bv1)
        self.assertEqual(result2.get_count(), 0)
    
    def test_intersection_type_error(self):
        """Test intersection with wrong type raises error."""
        with self.assertRaises(TypeError):
            self.bv1.intersect("not a bitvector")
        
        with self.assertRaises(TypeError):
            self.bv1.intersect(42)
    
    def test_equality(self):
        """Test BitVector equality comparison."""
        bv_same = BitVector({1, 3, 5, 7})
        bv_different = BitVector({1, 3, 5})
        
        self.assertEqual(self.bv1, bv_same)
        self.assertNotEqual(self.bv1, bv_different)
        self.assertNotEqual(self.bv1, "not a bitvector")
    
    def test_serialization_empty(self):
        """Test serialization of empty BitVector."""
        encoded = self.empty_bv.to_base64()
        decoded = BitVector.from_base64(encoded)
        
        self.assertEqual(decoded, self.empty_bv)
        self.assertEqual(decoded.get_count(), 0)
    
    def test_serialization_with_data(self):
        """Test serialization of BitVector with data."""
        encoded = self.bv1.to_base64()
        decoded = BitVector.from_base64(encoded)
        
        self.assertEqual(decoded, self.bv1)
        self.assertEqual(decoded.get_count(), 4)
    
    def test_serialization_roundtrip(self):
        """Test multiple serialization/deserialization cycles."""
        original = BitVector({100, 200, 300, 999, 1001})
        
        # First roundtrip
        encoded1 = original.to_base64()
        decoded1 = BitVector.from_base64(encoded1)
        self.assertEqual(original, decoded1)
        
        # Second roundtrip
        encoded2 = decoded1.to_base64()
        decoded2 = BitVector.from_base64(encoded2)
        self.assertEqual(original, decoded2)
    
    def test_from_base64_invalid_data(self):
        """Test deserialization with invalid data."""
        # Invalid base64
        with self.assertRaises(ValueError):
            BitVector.from_base64("invalid_base64!")
        
        # Valid base64 but not a pickled set
        invalid_data = base64.b64encode(b"not a pickled set").decode('utf-8')
        with self.assertRaises(ValueError):
            BitVector.from_base64(invalid_data)
        
        # Valid base64 and pickle, but wrong data type
        wrong_type = base64.b64encode(pickle.dumps([1, 2, 3])).decode('utf-8')
        with self.assertRaises(ValueError):
            BitVector.from_base64(wrong_type)
        
        # Set with non-integer values
        invalid_set = base64.b64encode(pickle.dumps({1, 2, "three"})).decode('utf-8')
        with self.assertRaises(ValueError):
            BitVector.from_base64(invalid_set)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.bv1)
        self.assertIn("BitVector", repr_str)
        # Should contain the row IDs in some form
        self.assertTrue(any(str(i) in repr_str for i in [1, 3, 5, 7]))


@unittest.skipUnless(BITARRAY_AVAILABLE, "bitarray package not available")
class TestBitVectorBitArrayMode(unittest.TestCase):
    """Test BitVector using true bitarray implementation."""
    
    def setUp(self):
        """Set up test fixtures with universe size."""
        self.universe_size = 100
        self.empty_bv = BitVector(universe_size=self.universe_size)
        self.bv1 = BitVector({1, 3, 5, 7}, universe_size=self.universe_size)
        self.bv2 = BitVector({3, 4, 5, 8}, universe_size=self.universe_size)
    
    def test_bitarray_mode_creation(self):
        """Test BitVector creation in bitarray mode."""
        bv = BitVector({10, 20, 30}, universe_size=50)
        self.assertEqual(bv.get_count(), 3)
        self.assertTrue(bv._use_bitarray)
        self.assertEqual(len(bv.bits), 50)
    
    def test_bitarray_intersection(self):
        """Test intersection using bitarray operations."""
        result = self.bv1.intersect(self.bv2)
        
        # Should use bitarray mode
        self.assertTrue(result._use_bitarray)
        self.assertEqual(result.get_count(), 2)  # {3, 5}
        
        # Verify specific bits are set
        self.assertTrue(result.bits[3])
        self.assertTrue(result.bits[5])
        self.assertFalse(result.bits[1])
        self.assertFalse(result.bits[7])
    
    def test_bitarray_union(self):
        """Test union using bitarray operations."""
        result = self.bv1.union(self.bv2)
        
        self.assertTrue(result._use_bitarray)
        self.assertEqual(result.get_count(), 6)  # {1, 3, 4, 5, 7, 8}
        
        # Verify all expected bits are set
        for bit_pos in [1, 3, 4, 5, 7, 8]:
            self.assertTrue(result.bits[bit_pos])
    
    def test_universe_size_validation(self):
        """Test validation of row IDs against universe size."""
        with self.assertRaises(ValueError):
            BitVector({50, 100, 150}, universe_size=100)  # 100, 150 out of range
    
    def test_different_universe_sizes_error(self):
        """Test error when intersecting BitVectors with different universe sizes."""
        bv_small = BitVector({1, 2}, universe_size=10)
        bv_large = BitVector({1, 2}, universe_size=20)
        
        with self.assertRaises(ValueError):
            bv_small.intersect(bv_large)
    
    def test_bitarray_serialization(self):
        """Test serialization in bitarray mode."""
        original = BitVector({5, 15, 25}, universe_size=50)
        
        encoded = original.to_base64()
        decoded = BitVector.from_base64(encoded, universe_size=50)
        
        self.assertEqual(original, decoded)
        self.assertEqual(decoded.get_count(), 3)
    
    def test_add_row_id_bitarray_mode(self):
        """Test adding row IDs in bitarray mode."""
        bv = BitVector(universe_size=20)
        bv.add_row_id(5)
        bv.add_row_id(15)
        
        self.assertEqual(bv.get_count(), 2)
        self.assertTrue(bv.bits[5])
        self.assertTrue(bv.bits[15])
        
        # Test out of range
        with self.assertRaises(ValueError):
            bv.add_row_id(25)


class TestBitVectorMixedMode(unittest.TestCase):
    """Test interactions between different BitVector modes."""
    
    def test_cannot_intersect_different_modes(self):
        """Test that different implementation modes cannot be mixed."""
        bv_set = BitVector({1, 2, 3})  # Set mode
        
        if BITARRAY_AVAILABLE:
            bv_bits = BitVector({1, 2, 3}, universe_size=10)  # Bitarray mode
            
            with self.assertRaises(ValueError):
                bv_set.intersect(bv_bits)
            
            with self.assertRaises(ValueError):
                bv_bits.intersect(bv_set)


class TestBitVectorIntegration(unittest.TestCase):
    """Integration tests for BitVector in decision tree context."""
    
    def test_decision_tree_scenario(self):
        """Test a realistic decision tree scenario."""
        # Simulate target class sketches
        y0_sketch = BitVector({0, 2, 4, 6, 8})  # Class 0 rows
        y1_sketch = BitVector({1, 3, 5, 7, 9})  # Class 1 rows
        
        # Simulate feature sketches
        feature_a_true = BitVector({1, 2, 5, 6, 9})  # feature_a=true
        feature_b_true = BitVector({3, 4, 7, 8, 9})  # feature_b=true
        
        # Test intersections for tree splitting
        y0_and_feature_a = y0_sketch.intersect(feature_a_true)
        y1_and_feature_a = y1_sketch.intersect(feature_a_true)
        
        # Verify counts make sense
        self.assertEqual(y0_and_feature_a.get_count(), 2)  # rows 2, 6
        self.assertEqual(y1_and_feature_a.get_count(), 3)  # rows 1, 5, 9
        
        # Test that intersections are disjoint
        overlap = y0_and_feature_a.intersect(y1_and_feature_a)
        self.assertEqual(overlap.get_count(), 0)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataSketchInterface,
        TestBitVectorSetMode, 
        TestBitVectorBitArrayMode,
        TestBitVectorMixedMode,
        TestBitVectorIntegration
    ]
    
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
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
