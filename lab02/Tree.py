from typing import List

from PointSet import PointSet, FeaturesTypes
    
class Node:
    """A node of a decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the node
        left_node : Node
            The left node of the tree
        right_node : Node
            The right node of the tree
    """
    def __init__(self,
                 points: PointSet,
                 left_node: 'Node' = None,
                 right_node: 'Node' = None):
        """
        Parameters
        ----------
            points : PointSet
                The training points of the node
            left_node : Node
                The left node of the tree (split_feature = true)
            right_node : Node
                The right node of the tree (split_feature = false)
        """
        self.points = points
        self.left_node = left_node
        self.right_node = right_node

    def is_leaf(self)-> bool:
        """Returns True if the node is a leaf, False otherwise"""
        return self.left_node is None and self.right_node is None
     
"""The constructor is expected to create a binary tree of height 1
(that is, one root and two children, which are leaves) by splitting the points of
its PointSet along the feature that provides best gain"""
class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        root : Node
            The root of the tree
        split_feature_index : int
            The index of the feature along which the points have been split
        split_value : float
            The value of the feature along which the points have been split
            for categorical feature, split_value(left child), the other value(right child)
            for real feature, less than split_value(left child), greater than split_value(right child)
            for boolean feature, split_value = None
    """
            
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points : int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
            h : int default=1
                The height of the tree. The tree will have a maximum
                depth of leaf (the root is at depth 0).
        """
        # root contains all points
        self.points = PointSet(features,labels,types)
        self.root = Node(self.points,None,None)
        # generate the tree from the root
        # generate function will initialize the split_feature_index and split_value
        # by how the root is split
        self.generate(self.root,h,types,min_split_points)
        
        
    def generate(self, pivot: Node, h:int, types, min_split_points : int = 1):
        """Generate the tree from the pivot node
            
            Parameters
            ----------
            h: height of the pivot node, which equals to
                the height of the tree - depth of the pivot node
            types : List[FeaturesTypes]
                The types of the features.
        """
        # if the pivot node is composed of one class or height of pivot = 0, stop and return
        if pivot.points.get_gini() == 0 or h == 0:
            return
        # else, split the pivot node along the feature that provides best gain to fill left and right nodes
        (left_class_features, left_class_labels,
         right_class_features,right_class_lables)  = pivot.points.split_with_best_gain(min_split_points) 
        self.split_feature_index = pivot.points.split_feature_index
        self.split_value = pivot.points.split_value
        # if no split can reduce gini, stop generating from this node and return
        if self.split_feature_index == None:
            return
        
        left_node_points = PointSet(left_class_features,left_class_labels,types)
        right_node_points = PointSet(right_class_features,right_class_lables,types)
        left_node = Node(left_node_points,None,None)
        right_node = Node(right_node_points,None,None)
        pivot.left_node = left_node
        pivot.right_node = right_node
        # recursively generate the left and right nodes
        self.generate(left_node,h-1,types,min_split_points)
        self.generate(right_node,h-1,types,min_split_points)
    
    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        # parse the decision tree and assign the point to a leaf
        now_node = self.root
        while now_node.is_leaf() == False:
            self.split_feature_index = now_node.points.split_feature_index
            self.split_value = now_node.points.split_value
            if now_node.points.types[self.split_feature_index] == FeaturesTypes.BOOLEAN:
                if features[self.split_feature_index]:
                    now_node = now_node.left_node
                else:
                    now_node = now_node.right_node
            elif now_node.points.types[self.split_feature_index] == FeaturesTypes.CLASSES:
                if features[self.split_feature_index] == self.split_value:
                    now_node = now_node.left_node
                else:
                    now_node = now_node.right_node
            else: # real feature
                if features[self.split_feature_index] < self.split_value:
                    now_node = now_node.left_node
                else:
                    now_node = now_node.right_node
        
        #test the leaf is true label or false label
        true_count = 0
        false_count = 0
        for label in now_node.points.labels:
            if label == True:
                true_count += 1
            else:
                false_count += 1
        
        return true_count > false_count
        
    

        