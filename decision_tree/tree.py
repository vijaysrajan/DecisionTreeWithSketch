from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np

# Import our components
from sketches import DataSketch, SketchFactory
from impurity import ImpurityCalculator, ImpurityMethod  
from nodes import TreeNode, ClassDistribution


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using DataSketch abstractions.
    
    Builds decision trees using efficient sketch-based operations for 
    categorical features and binary classification.
    """
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1,
                 criterion: str = 'entropy',
                 sketch_type: str = 'bitvector',
                 store_node_sketches: bool = False,
                 random_state: Optional[int] = None,
                 prediction_threshold: float = 0.5):
        """
        Initialize the Decision Tree Classifier.
        
        Args:
            max_depth (int, optional): Maximum depth of the tree
            min_samples_leaf (int): Minimum number of samples required in a leaf
            criterion (str): Splitting criterion ('entropy' or 'gini')
            sketch_type (str): Type of sketch to use ('bitvector' or 'thetasketch')
            store_node_sketches (bool): Whether to store data sketches in nodes (for analysis)
            random_state (int, optional): Random seed for reproducibility
            prediction_threshold (float): Threshold for predicting class 1 (default: 0.5)
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.sketch_type = sketch_type
        self.store_node_sketches = store_node_sketches
        self.random_state = random_state
        self.prediction_threshold = prediction_threshold
        
        # Convert criterion string to enum
        if criterion.lower() == 'entropy':
            self.impurity_method = ImpurityMethod.ENTROPY
        elif criterion.lower() == 'gini':
            self.impurity_method = ImpurityMethod.GINI
        else:
            raise ValueError(f"Unsupported criterion: {criterion}. Use 'entropy' or 'gini'")
        
        # Tree components
        self.root: Optional[TreeNode] = None
        self.features: Optional[Dict[str, DataSketch]] = None
        self.feature_names: Optional[List[str]] = None
        self.n_features: int = 0
        self.n_samples: int = 0
        
        # Training metadata
        self.training_time: float = 0.0
        self.node_count: int = 0
        self.leaf_count: int = 0
        self.max_tree_depth: int = 0
    
    def fit(self, features: Dict[str, DataSketch], targets: Dict[str, DataSketch]) -> 'DecisionTreeClassifier':
        """
        Build a decision tree from training data.
        
        Args:
            features (Dict[str, DataSketch]): Feature sketches {"feature_name=value": sketch}
            targets (Dict[str, DataSketch]): Target sketches {"y=0": sketch, "y=1": sketch}
            
        Returns:
            DecisionTreeClassifier: Fitted classifier (self)
            
        Raises:
            ValueError: If targets don't have exactly y=0 and y=1
        """
        print(f"Training Decision Tree with {self.criterion} criterion (min_samples_leaf={self.min_samples_leaf})...")
        start_time = time.time()
        
        # Validate inputs
        if not features:
            raise ValueError("No features provided")
        
        if set(targets.keys()) != {"y=0", "y=1"}:
            raise ValueError(f"Targets must have exactly 'y=0' and 'y=1'. Got: {list(targets.keys())}")
        
        # Store training data
        self.features = features
        self.feature_names = list(features.keys())
        self.n_features = len(features)
        self.n_samples = targets["y=0"].get_count() + targets["y=1"].get_count()
        
        print(f"Training data: {self.n_samples} samples, {self.n_features} features")
        
        # Build the tree recursively
        self.root = self._build_tree(
            remaining_features=features,
            current_y0_sketch=targets["y=0"],
            current_y1_sketch=targets["y=1"],
            depth=0,
            node_id=0
        )
        
        # Update training metadata
        self.training_time = time.time() - start_time
        self.node_count = self._count_nodes(self.root)
        self.leaf_count = self.root.get_leaf_count()
        self.max_tree_depth = self.root.get_max_depth()
        
        print(f"Training completed in {self.training_time:.3f}s")
        print(f"Tree: {self.node_count} nodes, {self.leaf_count} leaves, max depth {self.max_tree_depth}")
        
        return self
    
    def _build_tree(self, 
                   remaining_features: Dict[str, DataSketch],
                   current_y0_sketch: DataSketch,
                   current_y1_sketch: DataSketch,
                   depth: int,
                   node_id: int) -> TreeNode:
        """
        Recursively build the decision tree.
        
        Args:
            remaining_features (Dict[str, DataSketch]): Available features for splitting
            current_y0_sketch (DataSketch): Class 0 instances at this node
            current_y1_sketch (DataSketch): Class 1 instances at this node
            depth (int): Current depth in the tree
            node_id (int): Unique node identifier
            
        Returns:
            TreeNode: Root of the subtree
        """
        # Create class distribution for this node
        class_dist = ClassDistribution(
            y0_count=current_y0_sketch.get_count(),
            y1_count=current_y1_sketch.get_count()
        )
        
        print(f"{'  ' * depth}Building node {node_id} at depth {depth}: {class_dist}")
        
        # Check stopping conditions
        if self._is_stopping_condition(current_y0_sketch, current_y1_sketch, depth, remaining_features):
            # Create leaf node
            leaf = TreeNode(
                is_leaf=True,
                class_distribution=class_dist,
                depth=depth,
                node_id=node_id,
                y0_sketch=current_y0_sketch if self.store_node_sketches else None,
                y1_sketch=current_y1_sketch if self.store_node_sketches else None
            )
            
            # Calculate impurity for this leaf
            leaf.impurity = ImpurityCalculator.calculate_impurity(
                current_y0_sketch, current_y1_sketch, self.impurity_method
            )
            
            print(f"{'  ' * depth}Created leaf: P(y=1)={class_dist.y1_probability:.3f}, impurity={leaf.impurity:.4f}")
            return leaf
        
        # Find best split
        best_feature, best_gain = self._find_best_split(
            remaining_features, current_y0_sketch, current_y1_sketch
        )
        
        # Check if no valid splits were found
        if best_feature is None or best_gain <= 0:
            # No valid splits found, create leaf
            print(f"{'  ' * depth}No valid splits found (best_gain={best_gain}), creating leaf")
            leaf = TreeNode(
                is_leaf=True,
                class_distribution=class_dist,
                depth=depth,
                node_id=node_id,
                y0_sketch=current_y0_sketch if self.store_node_sketches else None,
                y1_sketch=current_y1_sketch if self.store_node_sketches else None
            )
            leaf.impurity = ImpurityCalculator.calculate_impurity(
                current_y0_sketch, current_y1_sketch, self.impurity_method
            )
            return leaf
        
        # Create internal node
        node = TreeNode(
            feature=best_feature,
            is_leaf=False,
            class_distribution=class_dist,
            depth=depth,
            node_id=node_id,
            y0_sketch=current_y0_sketch if self.store_node_sketches else None,
            y1_sketch=current_y1_sketch if self.store_node_sketches else None
        )
        node.split_gain = best_gain
        node.impurity = ImpurityCalculator.calculate_impurity(
            current_y0_sketch, current_y1_sketch, self.impurity_method
        )
        
        print(f"{'  ' * depth}Splitting on {best_feature} (gain={best_gain:.4f})")
        
        # Split the data
        left_y0, left_y1, right_y0, right_y1 = self._split_sketches(
            remaining_features[best_feature], current_y0_sketch, current_y1_sketch
        )
        
        # Remove the used feature from remaining features
        remaining_features_left = {k: v for k, v in remaining_features.items() if k != best_feature}
        remaining_features_right = remaining_features_left.copy()  # Both children get same remaining features
        
        # Recursively build children
        self.node_count += 2  # Will create 2 more nodes
        
        left_child = self._build_tree(
            remaining_features_left, left_y0, left_y1, 
            depth + 1, node_id * 2 + 1
        )
        
        right_child = self._build_tree(
            remaining_features_right, right_y0, right_y1,
            depth + 1, node_id * 2 + 2
        )
        
        # Set children
        node.set_children(left_child, right_child)
        
        return node
    
    def _find_best_split(self, 
                        remaining_features: Dict[str, DataSketch],
                        current_y0_sketch: DataSketch,
                        current_y1_sketch: DataSketch) -> Tuple[Optional[str], float]:
        """
        Find the best feature to split on that satisfies min_samples_leaf constraint.
        
        Args:
            remaining_features (Dict[str, DataSketch]): Available features
            current_y0_sketch (DataSketch): Current class 0 instances
            current_y1_sketch (DataSketch): Current class 1 instances
            
        Returns:
            Tuple[Optional[str], float]: (best_feature_name, best_information_gain)
                Returns (None, 0.0) if no valid splits found
        """
        if not remaining_features:
            return None, 0.0
        
        # Use ImpurityCalculator to find best split that respects min_samples_leaf
        best_feature, best_gain = ImpurityCalculator.find_best_split(
            remaining_features, current_y0_sketch, current_y1_sketch, 
            self.impurity_method, self.min_samples_leaf
        )
        
        return best_feature, best_gain
    
    def _is_stopping_condition(self,
                              current_y0_sketch: DataSketch,
                              current_y1_sketch: DataSketch,
                              depth: int,
                              remaining_features: Dict[str, DataSketch]) -> bool:
        """
        Check if we should stop splitting at this node.
        
        Args:
            current_y0_sketch (DataSketch): Current class 0 instances
            current_y1_sketch (DataSketch): Current class 1 instances  
            depth (int): Current depth
            remaining_features (Dict[str, DataSketch]): Available features
            
        Returns:
            bool: True if we should stop splitting
        """
        # Check max depth
        if self.max_depth is not None and depth >= self.max_depth:
            print(f"{'  ' * depth}Stopping: max depth {self.max_depth} reached")
            return True
        
        # Check if node is pure (only one class)
        if current_y0_sketch.get_count() == 0 or current_y1_sketch.get_count() == 0:
            print(f"{'  ' * depth}Stopping: node is pure")
            return True
        
        # Check if no features remaining
        if not remaining_features:
            print(f"{'  ' * depth}Stopping: no features remaining")
            return True
        
        # Check minimum samples per leaf
        total_samples = current_y0_sketch.get_count() + current_y1_sketch.get_count()
        
        # If current node has fewer than 2 * min_samples_leaf samples,
        # any split will violate the min_samples_leaf constraint
        if total_samples < 2 * self.min_samples_leaf:
            print(f"{'  ' * depth}Stopping: not enough samples ({total_samples} < {2 * self.min_samples_leaf})")
            return True
        
        # Additional check: look at actual proposed splits to see if they would violate min_samples_leaf
        if self.min_samples_leaf > 1:  # Only do this check if min_samples_leaf is meaningful
            valid_splits_found = False
            for feature_name, feature_sketch in remaining_features.items():
                left_y0, left_y1, right_y0, right_y1 = ImpurityCalculator.calculate_feature_split_counts(
                    feature_sketch, current_y0_sketch, current_y1_sketch
                )
                left_total = left_y0 + left_y1
                right_total = right_y0 + right_y1
                
                # If this feature creates a valid split (both children >= min_samples_leaf), we can continue
                if left_total >= self.min_samples_leaf and right_total >= self.min_samples_leaf:
                    valid_splits_found = True
                    break  # Found at least one valid split, don't stop
            
            if not valid_splits_found:
                print(f"{'  ' * depth}Stopping: no valid splits (all violate min_samples_leaf={self.min_samples_leaf})")
                return True
        
        return False
    
    def _split_sketches(self,
                       feature_sketch: DataSketch,
                       current_y0_sketch: DataSketch,
                       current_y1_sketch: DataSketch) -> Tuple[DataSketch, DataSketch, DataSketch, DataSketch]:
        """
        Split the current sketches based on a feature using proper sketch operations.
        
        Args:
            feature_sketch (DataSketch): Feature sketch for splitting
            current_y0_sketch (DataSketch): Current class 0 instances
            current_y1_sketch (DataSketch): Current class 1 instances
            
        Returns:
            Tuple[DataSketch, DataSketch, DataSketch, DataSketch]: 
                (left_y0, left_y1, right_y0, right_y1)
        """
        # Left side: feature=true (intersection)
        left_y0 = feature_sketch.intersect(current_y0_sketch)
        left_y1 = feature_sketch.intersect(current_y1_sketch)
        
        # Right side: feature=false (set subtraction: current - left)
        try:
            right_y0 = current_y0_sketch.subtract(left_y0)
            right_y1 = current_y1_sketch.subtract(left_y1)
        except NotImplementedError:
            # If sketch type doesn't support subtraction, we have a problem
            raise NotImplementedError(
                f"Sketch type '{self.sketch_type}' does not support subtraction operation. "
                "Binary tree splits require set subtraction to compute complements."
            )
        
        return left_y0, left_y1, right_y0, right_y1
    
    def predict(self, samples: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[int]:
        """
        Predict classes for samples.
        
        Args:
            samples (List[Dict[str, Any]]): List of samples to predict
            threshold (float, optional): Threshold for predicting class 1. 
                                       If None, uses self.prediction_threshold
            
        Returns:
            List[int]: Predicted classes
        """
        if self.root is None:
            raise ValueError("Tree not trained. Call fit() first.")
        
        if threshold is None:
            threshold = self.prediction_threshold
        
        predictions = []
        for sample in samples:
            pred = self.root.predict_single(sample, threshold)
            predictions.append(pred)
        
        return predictions
    
    def predict_proba(self, samples: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """
        Predict class probabilities for samples.
        
        Args:
            samples (List[Dict[str, Any]]): List of samples to predict
            
        Returns:
            List[Tuple[float, float]]: List of (P(class=0), P(class=1))
        """
        if self.root is None:
            raise ValueError("Tree not trained. Call fit() first.")
        
        probabilities = []
        for sample in samples:
            proba = self.root.predict_proba_single(sample)
            probabilities.append(proba)
        
        return probabilities
    
    def compute_roc_auc(self, test_samples: List[Dict[str, Any]], true_labels: List[int]) -> Tuple[List[float], List[float], List[float], float]:
        """
        Compute ROC curve and AUC score.
        
        Args:
            test_samples (List[Dict[str, Any]]): Test samples
            true_labels (List[int]): True labels (0 or 1)
            
        Returns:
            Tuple containing:
                - fpr (List[float]): False positive rates
                - tpr (List[float]): True positive rates  
                - thresholds (List[float]): Thresholds used
                - auc (float): Area under the ROC curve
        """
        if self.root is None:
            raise ValueError("Tree not trained. Call fit() first.")
        
        # Get predicted probabilities for class 1
        probas = self.predict_proba(test_samples)
        y_scores = [p[1] for p in probas]  # P(y=1)
        
        # Convert to numpy arrays for easier computation
        y_true = np.array(true_labels)
        y_scores = np.array(y_scores)
        
        # Get unique thresholds from predicted probabilities
        thresholds = sorted(set(y_scores), reverse=True)
        thresholds.append(0.0)  # Add 0 to ensure we get (1, 1) point
        
        fpr_list = []
        tpr_list = []
        
        for threshold in thresholds:
            # Predict at this threshold
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
        
        return fpr_list, tpr_list, thresholds, auc
    
    def print_tree(self, threshold: Optional[float] = None) -> None:
        """
        Print a visual representation of the decision tree.
        
        Args:
            threshold (float, optional): Threshold for showing predicted classes.
                                       If None, uses self.prediction_threshold
        """
        if self.root is None:
            print("Tree not trained.")
            return
        
        if threshold is None:
            threshold = self.prediction_threshold
        
        print(f"\nDecision Tree (criterion={self.criterion}, max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf}, threshold={threshold}):")
        print(f"Training samples: {self.n_samples}, Features: {self.n_features}")
        print(f"Nodes: {self.node_count}, Leaves: {self.leaf_count}, Max depth: {self.max_tree_depth}")
        print("-" * 80)
        self.root.print_tree(threshold=threshold)
        print("-" * 80)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on information gain.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.root is None:
            return {}
        
        importance = {}
        self._calculate_feature_importance(self.root, importance)
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _calculate_feature_importance(self, node: TreeNode, importance: Dict[str, float]) -> None:
        """
        Recursively calculate feature importance.
        
        Args:
            node (TreeNode): Current node
            importance (Dict[str, float]): Importance accumulator
        """
        if node.is_leaf:
            return
        
        # Add this node's contribution to feature importance
        if node.feature and node.split_gain:
            feature_base = node.feature.split('=')[0]  # Extract base feature name
            importance[feature_base] = importance.get(feature_base, 0.0) + node.split_gain
        
        # Recurse to children
        if node.left_child:
            self._calculate_feature_importance(node.left_child, importance)
        if node.right_child:
            self._calculate_feature_importance(node.right_child, importance)
    
    def _count_nodes(self, node: TreeNode) -> int:
        """
        Count total nodes in the tree.
        
        Args:
            node (TreeNode): Root node
            
        Returns:
            int: Total node count
        """
        if node is None:
            return 0
        
        count = 1
        if node.left_child:
            count += self._count_nodes(node.left_child)
        if node.right_child:
            count += self._count_nodes(node.right_child)
        
        return count
    
    def __str__(self) -> str:
        return f"DecisionTreeClassifier(criterion={self.criterion}, max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf}, threshold={self.prediction_threshold}, trained={self.root is not None})"
