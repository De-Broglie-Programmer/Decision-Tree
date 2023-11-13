from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
        split_feature_index : int
            along which feature the points have been split
        split_value: float
            the value of the feature along which the points have been split
            for boolean feature, split_value = 1.0
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.split_feature_index = None
        self.split_value = None
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        gini = 0.0
        num_label_true = 0
        num_label_false = 0
        for label in self.labels:
            if label:
                num_label_true += 1
            else:
                num_label_false += 1
        gini = 1 - (num_label_true/len(self.labels))**2 - (num_label_false/len(self.labels))**2
        
        return gini
        

    def compute_gini_split(self, feature_index: int, split_value: float = 1.0) -> float:
        """Computes the Gini_split score of points after splitting
        them along a feature

        Parameters
        ----------
        feature_index : int
            The index of the feature along which the points have been
            split
        split_value : float
            The value of the feature along which the points have been
            split, for boolean feature, split_value = 1.0

        Returns
        -------
        float
            The Gini split of points after splitting them
            into 2 sets along the feature
        """
        feature_true_indexs = []
        feature_false_indexs = []
        temp_type = self.types[feature_index]
        if temp_type == FeaturesTypes.BOOLEAN:
            for j in range(len(self.features)):
                if self.features[j][feature_index]:
                    feature_true_indexs.append(j)
                else:   
                    feature_false_indexs.append(j)
        elif temp_type == FeaturesTypes.CLASSES:
            for j in range(len(self.features)):
                if self.features[j][feature_index] == split_value: 
                    feature_true_indexs.append(j)
                else:
                    feature_false_indexs.append(j)
                    
        ## If a feature has the same value in all the points, it cannot be used to split the set
        ##  return None beacause gini_split is not defined
        if (len(feature_true_indexs) == 0 or len(feature_false_indexs) == 0):
           return None
        
        ## calculate gini for feature_true_indexs
        num_label_true = 0
        num_label_false = 0
        for k in feature_true_indexs:
            if self.labels[k]:
                num_label_true += 1
            else:
                num_label_false += 1
        gini_true = 1 - (num_label_true/len(feature_true_indexs))**2 - (num_label_false/len(feature_true_indexs))**2
        
        ##calculate gini for feature_false_indexs
        num_label_true = 0
        num_label_false = 0
        for k in feature_false_indexs:
            if self.labels[k]:
                num_label_true += 1
            else:
                num_label_false += 1
        gini_false = 1 - (num_label_true/len(feature_false_indexs))**2 - (num_label_false/len(feature_false_indexs))**2
        
        ## calculate gini_split
        gini_split = (len(feature_true_indexs)/len(self.features))*gini_true + (len(feature_false_indexs)/len(self.features))*gini_false
        return gini_split
    
    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        max_gini_gain = 0.0
        best_feature_index = None
        gini = self.get_gini()
        
        ## split the set along each feature and calculate gini gain
        for tmp_feature_index in range(len(self.features[0])):
            gini_split = self.compute_gini_split(tmp_feature_index)
            if gini_split == None:
                continue 
            ## gini_gain = gini - gini_split
            gini_gain = gini - gini_split
            if (gini_gain > max_gini_gain):
                max_gini_gain = gini_gain
                best_feature_index = tmp_feature_index
                
        ## If no feature provides a gain well-defined (gain>0), return (None, None)        
        if (max_gini_gain == 0.0):
            return (None, None)
        
        return (best_feature_index, max_gini_gain)       
            
            
    def get_best_gain_multi_types(self) -> Tuple[int, float, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        float
            split value: The value of the feature along which splitting the set provides
                        for boolean feature, split_value = 1.0
        """
        
        max_gini_gain = 0.0
        best_feature_index = None
        best_split_value = None
        gini = self.get_gini()
        
        ## split the set along each feature (each value if type is not bool) and calculate gini gain
        for tmp_feature_index in range(len(self.features[0])):
            tmp_split_value = None
            temp_type = self.types[tmp_feature_index]
            if temp_type == FeaturesTypes.BOOLEAN:
                tmp_split_value = 1.0
                gini_split = self.compute_gini_split(tmp_feature_index, tmp_split_value)
                if gini_split == None:
                    continue 
                ## gini_gain = gini - gini_split
                gini_gain = gini - gini_split
                if (gini_gain > max_gini_gain):
                    max_gini_gain = gini_gain
                    best_feature_index = tmp_feature_index
                    best_split_value = tmp_split_value  
            elif temp_type == FeaturesTypes.CLASSES:
                feature_values = {}  # store all possible values of a feature, dict is quick for query
                for i in range(len(self.features)):
                    if self.features[i][tmp_feature_index] not in feature_values:
                        feature_values[self.features[i][tmp_feature_index]] = 1
                for tmp_split_value in feature_values:
                    gini_split = self.compute_gini_split(tmp_feature_index, tmp_split_value)
                    if gini_split == None:
                        continue
                    ## gini_gain = gini - gini_split
                    gini_gain = gini - gini_split
                    if (gini_gain > max_gini_gain):
                        max_gini_gain = gini_gain
                        best_feature_index = tmp_feature_index
                        best_split_value = tmp_split_value
        
        ## If no feature provides a gain well-defined (gain>0), return (None, None)        
        if (max_gini_gain == 0.0):
            return (None, None, None)
        
        return (best_feature_index, max_gini_gain, best_split_value)
            

    def split_with_best_gain(self)->(int,float,List[List[float]], List[bool], List[List[float]], List[bool]):
        """Split the set of points along the feature that provides best gain
        
        Returns
        -------
        int 
            The ID of the feature along which the points have been split
        float
            split value: The value of the feature along which splitting the set provides
                        for boolean feature, split_value = 1.0
        List[List[float]]
            The features of the first subset of points (split_feature = true)
        List[bool]
            The labels of the first subset of points (split_feature = true)
        List[List[float]]
            The features of the second subset of points (split_feature = false)
        List[bool]  
            The labels of the second subset of points (split_feature = false)
        """
        self.split_feature_index, self.split_value = self.get_best_gain_multi_types()[0], self.get_best_gain_multi_types()[2]
        
        # if no split can reduces gini, return (None, None, None, None, None)
        if self.split_feature_index == None:
            return (None, None, None, None, None, None)
        
        true_class_features = []
        true_class_labels = []
        false_class_features = []
        false_class_labels = []
        for i in range(len(self.labels)):
            if self.types[self.split_feature_index] == FeaturesTypes.BOOLEAN:
                if self.features[i][self.split_feature_index]:
                    true_class_features.append(self.features[i])
                    true_class_labels.append(self.labels[i])
                else:   
                    false_class_features.append(self.features[i])   
                    false_class_labels.append(self.labels[i])
            elif self.types[self.split_feature_index] == FeaturesTypes.CLASSES:
                if self.features[i][self.split_feature_index] == self.split_value:
                    true_class_features.append(self.features[i])
                    true_class_labels.append(self.labels[i])
                else:
                    false_class_features.append(self.features[i])   
                    false_class_labels.append(self.labels[i])
        
        return (self.split_feature_index, self.split_value, true_class_features, true_class_labels, false_class_features, false_class_labels)
        
