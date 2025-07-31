from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json

# Import our DataSketch abstract base class
from sketches import DataSketch


@dataclass
class ClassDistribution:
    """
    Class distribution information for tree nodes.
    
    Stores the count and proportion of each class at a node.
    """
    y0_count: int
    y1_count: int
    
    @property
    def total_count(self) -> int:
        """Total number of samples at this node."""
        return self.y0_count + self.y1_count
    
    @property
    def y0_probability(self) -> float:
        """Probability of class 0."""
        if self.total_count == 0:
            return 0.0
        return self.y0_count / self.total_count
    
    @property
    def y1_probability(self) -> float:
        """Probability of class 1."""
        if self.total_count == 0:
            return 0.0
        return self.y1_count / self.total_count
    
    def predicted_class(self, threshold: float = 0.5) -> int:
        """
        Predicted class based on threshold.
        
        Args:
            threshold (float): Threshold for predicting class 1 (default: 0.5)
                             If P(y=1) >= threshold, predict class 1
        
        Returns:
            int: Predicted class (0 or 1)
        """
        if self.y1_probability >= threshold:
            return 1
        else:
            return 0
    
    @property
    def confidence(self) -> float:
        """Confidence of the prediction (proportion of majority class)."""
        if self.total_count == 0:
            return 0.0
        return max(self.y0_probability, self.y1_probability)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'y0_count': self.y0_count,
            'y1_count': self.y1_count,
            'total_count': self.total_count,
            'y0_probability': round(self.y0_probability, 4),
            'y1_probability': round(self.y1_probability, 4),
            'confidence': round(self.confidence, 4)
        }
    
    def __str__(self) -> str:
        return f"ClassDist(y0={self.y0_count}, y1={self.y1_count}, P(y=1)={self.y1_probability:.3f})"


class TreeNode:
    """
    Node in a decision tree.
    
    Can be either an internal node (with a split condition and children) or 
    a leaf node (with a class prediction).
    """
    
    def __init__(self, 
                 feature: Optional[str] = None,
                 is_leaf: bool = False,
                 class_distribution: Optional[ClassDistribution] = None,
                 depth: int = 0,
                 node_id: Optional[int] = None,
                 y0_sketch: Optional['DataSketch'] = None,
                 y1_sketch: Optional['DataSketch'] = None):
        """
        Initialize a tree node.
        
        Args:
            feature (str, optional): Feature used for splitting (e.g., "income=high")
            is_leaf (bool): Whether this is a leaf node
            class_distribution (ClassDistribution, optional): Class distribution at this node
            depth (int): Depth of this node in the tree (root = 0)
            node_id (int, optional): Unique identifier for this node
            y0_sketch (DataSketch, optional): Class 0 data that reached this node
            y1_sketch (DataSketch, optional): Class 1 data that reached this node
        """
        # Node identification
        self.node_id = node_id
        self.depth = depth
        
        # Split information (for internal nodes)
        self.feature = feature  # e.g., "income=high"
        self.split_feature_name = None  # e.g., "income" 
        self.split_feature_value = None  # e.g., "high"
        
        if feature and '=' in feature:
            parts = feature.split('=', 1)
            self.split_feature_name = parts[0]
            self.split_feature_value = parts[1]
        
        # Tree structure
        self.left_child: Optional[TreeNode] = None   # feature=true path
        self.right_child: Optional[TreeNode] = None  # feature=false path
        
        # Leaf information
        self.is_leaf = is_leaf
        self.class_distribution = class_distribution
        
        # Data sketches (optional - for analysis and debugging)
        self.y0_sketch = y0_sketch
        self.y1_sketch = y1_sketch
        
        # Metadata for analysis
        self.split_gain: Optional[float] = None  # Information gain from this split
        self.impurity: Optional[float] = None    # Impurity at this node
    
    def to_json(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Convert node to JSON-serializable dictionary.
        
        Args:
            threshold (float): Threshold for predicted class
            
        Returns:
            Dict[str, Any]: JSON representation of the node
        """
        node_dict = {
            'node_id': self.node_id,
            'depth': self.depth,
            'is_leaf': self.is_leaf,
            'samples': self.get_data_count(),
        }
        
        # Add class distribution info
        if self.class_distribution:
            node_dict['class_distribution'] = self.class_distribution.to_dict()
            node_dict['predicted_class'] = self.class_distribution.predicted_class(threshold)
        
        # Add impurity if available
        if self.impurity is not None:
            node_dict['impurity'] = round(self.impurity, 4)
        
        if self.is_leaf:
            # Leaf node
            node_dict['type'] = 'leaf'
        else:
            # Internal node
            node_dict['type'] = 'internal'
            node_dict['feature'] = self.feature
            node_dict['split_feature_name'] = self.split_feature_name
            node_dict['split_feature_value'] = self.split_feature_value
            
            if self.split_gain is not None:
                node_dict['split_gain'] = round(self.split_gain, 4)
            
            # Add children
            if self.left_child:
                node_dict['left_child'] = self.left_child.to_json(threshold)
            if self.right_child:
                node_dict['right_child'] = self.right_child.to_json(threshold)
        
        return node_dict
    
    def get_data_count(self) -> int:
        """
        Get total number of data points that reached this node.
        
        Returns:
            int: Total data count at this node
        """
        if self.y0_sketch and self.y1_sketch:
            return self.y0_sketch.get_count() + self.y1_sketch.get_count()
        elif self.class_distribution:
            return self.class_distribution.total_count
        else:
            return 0
    
    def get_data_sketches(self) -> Tuple[Optional['DataSketch'], Optional['DataSketch']]:
        """
        Get the data sketches for this node.
        
        Returns:
            Tuple[DataSketch, DataSketch]: (y0_sketch, y1_sketch) or (None, None)
        """
        return self.y0_sketch, self.y1_sketch
    
    def set_children(self, left_child: 'TreeNode', right_child: 'TreeNode'):
        """
        Set the left and right children of this node.
        
        Args:
            left_child (TreeNode): Left child (feature=true)
            right_child (TreeNode): Right child (feature=false)
        """
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
    
    def make_leaf(self, class_distribution: ClassDistribution):
        """
        Convert this node to a leaf node.
        
        Args:
            class_distribution (ClassDistribution): Final class distribution
        """
        self.is_leaf = True
        self.class_distribution = class_distribution
        self.left_child = None
        self.right_child = None
        self.feature = None
    
    def predict_single(self, sample: Dict[str, Any], threshold: float = 0.5) -> int:
        """
        Predict class for a single sample by traversing the tree.
        
        Args:
            sample (Dict[str, Any]): Sample features {"feature_name": value}
            threshold (float): Threshold for predicting class 1
            
        Returns:
            int: Predicted class (0 or 1)
        """
        if self.is_leaf:
            return self.class_distribution.predicted_class(threshold)
        
        # Navigate based on feature value
        feature_present = self._check_feature_condition(sample)
        
        if feature_present:
            return self.left_child.predict_single(sample, threshold)
        else:
            return self.right_child.predict_single(sample, threshold)
    
    def predict_proba_single(self, sample: Dict[str, Any]) -> tuple[float, float]:
        """
        Predict class probabilities for a single sample.
        
        Args:
            sample (Dict[str, Any]): Sample features
            
        Returns:
            tuple[float, float]: (P(class=0), P(class=1))
        """
        if self.is_leaf:
            return (self.class_distribution.y0_probability, 
                   self.class_distribution.y1_probability)
        
        # Navigate based on feature value
        feature_present = self._check_feature_condition(sample)
        
        if feature_present:
            return self.left_child.predict_proba_single(sample)
        else:
            return self.right_child.predict_proba_single(sample)
    
    def _check_feature_condition(self, sample: Dict[str, Any]) -> bool:
        """
        Check if the sample satisfies this node's feature condition.
        
        Args:
            sample (Dict[str, Any]): Sample features
            
        Returns:
            bool: True if feature condition is met (go left), False otherwise (go right)
        """
        if not self.split_feature_name or not self.split_feature_value:
            return False
        
        # Check if sample has the feature value
        sample_value = sample.get(self.split_feature_name)
        return sample_value == self.split_feature_value
    
    def get_leaf_count(self) -> int:
        """
        Count the number of leaf nodes in the subtree rooted at this node.
        
        Returns:
            int: Number of leaf nodes
        """
        if self.is_leaf:
            return 1
        
        left_leaves = self.left_child.get_leaf_count() if self.left_child else 0
        right_leaves = self.right_child.get_leaf_count() if self.right_child else 0
        
        return left_leaves + right_leaves
    
    def get_max_depth(self) -> int:
        """
        Get the maximum depth of the subtree rooted at this node.
        
        Returns:
            int: Maximum depth
        """
        if self.is_leaf:
            return self.depth
        
        left_depth = self.left_child.get_max_depth() if self.left_child else self.depth
        right_depth = self.right_child.get_max_depth() if self.right_child else self.depth
        
        return max(left_depth, right_depth)
    
    def print_tree(self, indent: int = 0, prefix: str = "Root: ", threshold: float = 0.5) -> None:
        """
        Print a visual representation of the tree.
        
        Args:
            indent (int): Current indentation level
            prefix (str): Prefix for this node
            threshold (float): Threshold for showing predicted class
        """
        indent_str = "  " * indent
        
        if self.is_leaf:
            pred_class = self.class_distribution.predicted_class(threshold)
            print(f"{indent_str}{prefix}LEAF: {self.class_distribution} -> pred={pred_class}")
        else:
            gain_str = f"{self.split_gain:.4f}" if self.split_gain is not None else "0.0000"
            print(f"{indent_str}{prefix}{self.feature} "
                  f"(gain={gain_str}, "
                  f"samples={self.get_data_count()})")
            
            if self.left_child:
                self.left_child.print_tree(indent + 1, "├─ True:  ", threshold)
            
            if self.right_child:
                self.right_child.print_tree(indent + 1, "└─ False: ", threshold)
    
    def __str__(self) -> str:
        """String representation of the node."""
        if self.is_leaf:
            return f"LeafNode(depth={self.depth}, {self.class_distribution})"
        else:
            return f"InternalNode(depth={self.depth}, feature={self.feature}, children={bool(self.left_child and self.right_child)})"
    
    def __repr__(self) -> str:
        return self.__str__()
