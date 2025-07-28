import math
from typing import Dict, Tuple, Optional
from enum import Enum

# Import our DataSketch abstract base class
from sketches import DataSketch


class ImpurityMethod(Enum):
    """Enumeration of supported impurity calculation methods."""
    ENTROPY = "entropy"
    GINI = "gini"


class ImpurityCalculator:
    """
    Calculator for impurity measures used in decision tree construction.
    
    Provides entropy and Gini impurity calculations using DataSketch counts,
    as well as information gain calculations for feature splitting decisions.
    """
    
    @staticmethod
    def calculate_entropy(y0_sketch: 'DataSketch', y1_sketch: 'DataSketch') -> float:
        """
        Calculate entropy for binary classification using sketch counts.
        
        Entropy = -Σ(p_i * log2(p_i)) where p_i is proportion of class i
        
        Args:
            y0_sketch (DataSketch): Sketch containing class 0 instances
            y1_sketch (DataSketch): Sketch containing class 1 instances
            
        Returns:
            float: Entropy value (0.0 = pure, 1.0 = maximum impurity for binary)
        """
        y0_count = y0_sketch.get_count()
        y1_count = y1_sketch.get_count()
        total_count = y0_count + y1_count
        
        # Handle edge case of empty dataset
        if total_count == 0:
            return 0.0
        
        # Handle pure cases (only one class present)
        if y0_count == 0 or y1_count == 0:
            return 0.0
        
        # Calculate proportions
        p0 = y0_count / total_count
        p1 = y1_count / total_count
        
        # Calculate entropy: -p0*log2(p0) - p1*log2(p1)
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        
        return entropy
    
    @staticmethod
    def calculate_gini(y0_sketch: 'DataSketch', y1_sketch: 'DataSketch') -> float:
        """
        Calculate Gini impurity for binary classification using sketch counts.
        
        Gini = 1 - Σ(p_i^2) where p_i is proportion of class i
        
        Args:
            y0_sketch (DataSketch): Sketch containing class 0 instances
            y1_sketch (DataSketch): Sketch containing class 1 instances
            
        Returns:
            float: Gini impurity (0.0 = pure, 0.5 = maximum impurity for binary)
        """
        y0_count = y0_sketch.get_count()
        y1_count = y1_sketch.get_count()
        total_count = y0_count + y1_count
        
        # Handle edge case of empty dataset
        if total_count == 0:
            return 0.0
        
        # Calculate proportions
        p0 = y0_count / total_count
        p1 = y1_count / total_count
        
        # Calculate Gini: 1 - (p0^2 + p1^2)
        gini = 1.0 - (p0**2 + p1**2)
        
        return gini
    
    @staticmethod
    def calculate_impurity(y0_sketch: 'DataSketch', y1_sketch: 'DataSketch', 
                          method: ImpurityMethod) -> float:
        """
        Calculate impurity using specified method.
        
        Args:
            y0_sketch (DataSketch): Sketch containing class 0 instances
            y1_sketch (DataSketch): Sketch containing class 1 instances
            method (ImpurityMethod): Method to use (ENTROPY or GINI)
            
        Returns:
            float: Impurity value
            
        Raises:
            ValueError: If method is not supported
        """
        if method == ImpurityMethod.ENTROPY:
            return ImpurityCalculator.calculate_entropy(y0_sketch, y1_sketch)
        elif method == ImpurityMethod.GINI:
            return ImpurityCalculator.calculate_gini(y0_sketch, y1_sketch)
        else:
            raise ValueError(f"Unsupported impurity method: {method}")
    
    @staticmethod
    def calculate_feature_split_counts(feature_sketch: 'DataSketch', 
                                     y0_sketch: 'DataSketch', 
                                     y1_sketch: 'DataSketch') -> Tuple[int, int, int, int]:
        """
        Calculate split counts for a feature against target classes.
        
        For binary split: feature=true vs feature=false (complement)
        
        Args:
            feature_sketch (DataSketch): Sketch for feature=value condition
            y0_sketch (DataSketch): Sketch for class 0 instances  
            y1_sketch (DataSketch): Sketch for class 1 instances
            
        Returns:
            Tuple[int, int, int, int]: (left_y0, left_y1, right_y0, right_y1)
                - left_y0: class 0 instances where feature=true
                - left_y1: class 1 instances where feature=true  
                - right_y0: class 0 instances where feature=false
                - right_y1: class 1 instances where feature=false
        """
        # Left side: feature=true (intersect feature with each target class)
        left_y0_sketch = feature_sketch.intersect(y0_sketch)
        left_y1_sketch = feature_sketch.intersect(y1_sketch)
        
        left_y0 = left_y0_sketch.get_count()
        left_y1 = left_y1_sketch.get_count()
        
        # Right side: feature=false (complement - total minus left)
        total_y0 = y0_sketch.get_count()
        total_y1 = y1_sketch.get_count()
        
        right_y0 = total_y0 - left_y0
        right_y1 = total_y1 - left_y1
        
        return left_y0, left_y1, right_y0, right_y1
    
    @staticmethod
    def calculate_information_gain(parent_y0_sketch: 'DataSketch',
                                 parent_y1_sketch: 'DataSketch',
                                 left_y0_count: int, left_y1_count: int,
                                 right_y0_count: int, right_y1_count: int,
                                 method: ImpurityMethod = ImpurityMethod.ENTROPY) -> float:
        """
        Calculate information gain for a binary split.
        
        Information Gain = Parent_Impurity - Weighted_Average_Child_Impurity
        
        Args:
            parent_y0_sketch (DataSketch): Parent class 0 instances
            parent_y1_sketch (DataSketch): Parent class 1 instances
            left_y0_count (int): Left child class 0 count
            left_y1_count (int): Left child class 1 count
            right_y0_count (int): Right child class 0 count
            right_y1_count (int): Right child class 1 count
            method (ImpurityMethod): Impurity calculation method
            
        Returns:
            float: Information gain (higher is better)
        """
        # Calculate parent impurity
        parent_impurity = ImpurityCalculator.calculate_impurity(
            parent_y0_sketch, parent_y1_sketch, method
        )
        
        # Calculate total counts
        total_count = parent_y0_sketch.get_count() + parent_y1_sketch.get_count()
        left_total = left_y0_count + left_y1_count
        right_total = right_y0_count + right_y1_count
        
        # Handle edge cases
        if total_count == 0:
            return 0.0
        
        if left_total == 0:
            # All instances go to right child
            return 0.0
        
        if right_total == 0:
            # All instances go to left child  
            return 0.0
        
        # Create temporary sketches for child impurity calculation
        # Note: We create minimal sketches just for count-based calculations
        from sketches import SketchFactory
        
        # Create sketches with the counts (using bitvector for simplicity)
        left_y0_sketch = SketchFactory.create_sketch('bitvector', set(range(left_y0_count)))
        left_y1_sketch = SketchFactory.create_sketch('bitvector', set(range(left_y1_count)))
        right_y0_sketch = SketchFactory.create_sketch('bitvector', set(range(right_y0_count)))
        right_y1_sketch = SketchFactory.create_sketch('bitvector', set(range(right_y1_count)))
        
        # Calculate child impurities
        left_impurity = ImpurityCalculator.calculate_impurity(
            left_y0_sketch, left_y1_sketch, method
        )
        right_impurity = ImpurityCalculator.calculate_impurity(
            right_y0_sketch, right_y1_sketch, method
        )
        
        # Calculate weighted average of child impurities
        left_weight = left_total / total_count
        right_weight = right_total / total_count
        
        weighted_child_impurity = (left_weight * left_impurity + 
                                 right_weight * right_impurity)
        
        # Information gain = parent impurity - weighted child impurity
        information_gain = parent_impurity - weighted_child_impurity
        
        return information_gain
    
    @staticmethod
    def is_valid_split(left_y0_count: int, left_y1_count: int,
                      right_y0_count: int, right_y1_count: int,
                      min_samples_leaf: int) -> bool:
        """
        Check if a split satisfies the minimum samples per leaf constraint.
        
        Args:
            left_y0_count (int): Left child class 0 count
            left_y1_count (int): Left child class 1 count
            right_y0_count (int): Right child class 0 count
            right_y1_count (int): Right child class 1 count
            min_samples_leaf (int): Minimum samples required per leaf
            
        Returns:
            bool: True if split is valid, False otherwise
        """
        left_total = left_y0_count + left_y1_count
        right_total = right_y0_count + right_y1_count
        
        return left_total >= min_samples_leaf and right_total >= min_samples_leaf
    
    @staticmethod
    def evaluate_all_features(features: Dict[str, 'DataSketch'],
                            y0_sketch: 'DataSketch', 
                            y1_sketch: 'DataSketch',
                            method: ImpurityMethod = ImpurityMethod.ENTROPY,
                            min_samples_leaf: int = 1) -> Dict[str, float]:
        """
        Evaluate information gain for all features, filtering out invalid splits.
        
        Args:
            features (Dict[str, DataSketch]): Feature sketches {"feature_name=value": sketch}
            y0_sketch (DataSketch): Target class 0 sketch
            y1_sketch (DataSketch): Target class 1 sketch  
            method (ImpurityMethod): Impurity calculation method
            min_samples_leaf (int): Minimum samples required per leaf
            
        Returns:
            Dict[str, float]: Information gains {"feature_name=value": gain}
                Only includes features that create valid splits.
        """
        gains = {}
        
        print(f"Evaluating {len(features)} features using {method.value} (min_samples_leaf={min_samples_leaf})...")
        
        for feature_name, feature_sketch in features.items():
            # Calculate split counts for this feature
            left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
                feature_sketch, y0_sketch, y1_sketch
            )
            
            # Check if this split satisfies min_samples_leaf constraint
            if not ImpurityCalculator.is_valid_split(left_y0, left_y1, right_y0, right_y1, min_samples_leaf):
                print(f"  {feature_name}: INVALID split "
                      f"(left: {left_y0}+{left_y1}={left_y0+left_y1}, "
                      f"right: {right_y0}+{right_y1}={right_y0+right_y1}) < {min_samples_leaf}")
                continue  # Skip this feature as it creates invalid split
            
            # Calculate information gain for valid split
            gain = ImpurityCalculator.calculate_information_gain(
                y0_sketch, y1_sketch, 
                left_y0, left_y1, right_y0, right_y1,
                method
            )
            
            gains[feature_name] = gain
            
            print(f"  {feature_name}: gain={gain:.4f} "
                  f"(left: {left_y0}+{left_y1}={left_y0+left_y1}, "
                  f"right: {right_y0}+{right_y1}={right_y0+right_y1})")
        
        if not gains:
            print("  No valid splits found that satisfy min_samples_leaf constraint!")
        
        return gains
    
    @staticmethod
    def find_best_split(features: Dict[str, 'DataSketch'],
                       y0_sketch: 'DataSketch',
                       y1_sketch: 'DataSketch', 
                       method: ImpurityMethod = ImpurityMethod.ENTROPY,
                       min_samples_leaf: int = 1) -> Tuple[str, float]:
        """
        Find the feature with the highest information gain that creates a valid split.
        
        Args:
            features (Dict[str, DataSketch]): Feature sketches
            y0_sketch (DataSketch): Target class 0 sketch
            y1_sketch (DataSketch): Target class 1 sketch
            method (ImpurityMethod): Impurity calculation method
            min_samples_leaf (int): Minimum samples required per leaf
            
        Returns:
            Tuple[str, float]: (best_feature_name, best_gain)
                Returns (None, 0.0) if no valid splits found
            
        Raises:
            ValueError: If no features provided
        """
        if not features:
            raise ValueError("No features provided for split evaluation")
        
        gains = ImpurityCalculator.evaluate_all_features(
            features, y0_sketch, y1_sketch, method, min_samples_leaf
        )
        
        # Check if any valid splits were found
        if not gains:
            print("No valid splits found that satisfy constraints")
            return None, 0.0
        
        # Find feature with maximum gain among valid splits
        best_feature = max(gains.items(), key=lambda x: x[1])
        
        print(f"Best split: {best_feature[0]} with gain {best_feature[1]:.4f}")
        
        return best_feature
