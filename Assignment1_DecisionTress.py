#!/usr/bin/env python
# coding: utf-8

# #### This assignment may be worked individually or in pairs. Enter your name/names here:
#     

# In[ ]:


#Piaoping Jiang


# # Assignment 1: Decision Trees
# 
# In this assignment we'll implement the Decision Tree algorithm to classify patients as either having or not having diabetic retinopathy. For this task we'll be using the Diabetic Retinopathy data set, which contains features from the Messidor image set to predict whether an image contains signs of diabetic retinopathy or not. This dataset has `1151` instances and `20` attributes (some categorical, some continuous). You can find additional details about the dataset [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set).

# Attribute Information:
# 
# 0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
# 
# 1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack. 
# 
# 2-7) The results of MA detection. Each feature value stand for the number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively. 
# 
# 8-15) contain the same information as 2-7) for exudates. However, as exudates are represented by a set of points rather than the number of pixels constructing the lesions, these features are normalized by dividing the 
# number of lesions with the diameter of the ROI to compensate different image sizes. 
# 
# 16) The euclidean distance of the center of the macula and the center of the optic disc to provide important information regarding the patient's condition. This feature is also normalized with the diameter of the ROI.
# 
# 17) The diameter of the optic disc. 
# 
# 18) The binary result of the AM/FM-based classification.
# 
# 19) Class label. 1 = contains signs of Diabetic Retinopathy (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of Diabetic Retinopathy.
# 
# 
# A few function prototypes are already given to you, please don't change those. You can add additional helper functions for your convenience. *Suggestion:* The dataset is substantially big, for the purpose of easy debugging work with a subset of the data and test your decision tree implementation on that.

# #### Implementation: 
# A few function prototypes are already given to you, please don't change those. You can add additional helper functions for your convenience. 
# 
# *Suggestion:* The dataset is substantially big, for the purpose of easy debugging, work with a subset of the data and test your decision tree implementation on that.
# 
# #### Notes:
# Parts of this assignment will be **autograded** so a couple of caveats :-
# - Entropy is calculated using log with base 2, `math.log2(x)`.
# - For continuous features ensure that the threshold value lies exactly between 2 buckets. For example, if for feature 2 the best split occurs between 10 and 15 then the threshold value will be set as 12.5.
# - For binary features [0/1] the threshold value will be 0.5. All values < `thresh_val` go to the left child and all values >= `thresh_val` go to the right child.

# In[2]:


# Standard Headers
# You are welcome to add additional headers if you wish
# EXCEPT for scikit-learn... You may NOT use scikit-learn for this assignment!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from random import shuffle


# In[3]:


class DataPoint:
    def __str__(self):
        return "< " + str(self.label) + ": " + str(self.features) + " >"
    def __init__(self, label, features):
        self.label = label # the classification label of this data point
        self.features = features


# Q1. Read data from a CSV file. Put it into a list of `DataPoints`.

# In[3]:


def get_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            tp = line.strip().split(',')
            label = tp[19]
            features = tp[0:19]
            data.append(DataPoint(label, features))
    return data


# In[5]:


class TreeNode:
    is_leaf = True          # boolean variable to check if the node is a leaf
    feature_idx = None      # index that identifies the feature
    thresh_val = None       # threshold value that splits the node
    prediction = None       # prediction class (only valid for leaf nodes)
    left_child = None       # left TreeNode (all values < thresh_val)
    right_child = None      # right TreeNode (all values >= thresh_val)
    
    def printTree(self):    # for debugging purposes
        if self.is_leaf:
            print ('Leaf Node:      predicts ' + str(self.prediction))
        else:
            print ('Internal Node:  splits on feature ' 
                   + str(self.feature_idx) + ' with threshold ' + str(self.thresh_val))
            self.left_child.printTree()
            self.right_child.printTree()


# Q2. Implement the function `make_prediction` that takes the decision tree root and a `DataPoint` instance and returns the prediction label.

# In[6]:


def make_prediction(tree_root, data_point):
#     your code goes here
    if tree.root.is_leaf:
        pred =  tree_root.prediction
    else:
        if data_point.features[tree_root.feature_idx] >= tree_root.thresh_val:
            pred = make_prediction(tree_root.right_child, data_point)
        else:
            pred = make_prediction(tree_root.left_child, data_point)
    return pred


# Q3. Implement the function `split_dataset` given an input data set, a `feature_idx` and the `threshold` for the feature. `left_split` will have all values < `threshold` and `right_split` will have all values >= `threshold`.

# In[7]:


def split_dataset(data, feature_idx, threshold):
    left_split = []
    right_split = []
#     your code goes here
    for d in data:
        if float(d.features[feature_idx]) < threshold:
            left_split.append(d)
        else:
            right_split.append(d)
    return (left_split, right_split)


# Q4. Implement the function `calc_entropy` to return the entropy of the input dataset.

# In[8]:


def calc_entropy(data):
    entropy = 0.0
#     your code goes here
    n = len(data)
    pos = 0
    x = 0
    for d in data:
        x += int(d.label)
    p_pos = (x+1) / (n+2)
    p_neg = 1 - p_pos
    entropy  = -(p_pos * np.log(p_pos) + p_neg * np.log(p_neg))
    return entropy


# Q5. Implement the function `calc_best_threshold` which returns the best information gain and the corresponding threshold value for one feature at `feature_idx`.

# In[4]:


def calc_best_threshold(data, feature_idx):
    best_info_gain = 0.0
    best_thresh = None
#     your code goes here
#sort the dataset according to the value of feature_idx 
    n = len(data)
    my_data = data
    for i in range(n-1):
        for j in range (n-1-i):
            if float(my_data[j].features[feature_idx]) > float(my_data[j+1].features[feature_idx]):
                my_data[j], my_data[j+1] = my_data[j+1], my_data[j]
    
    best_thresh_candidates = []
    for i in range((n-1)):
        if my_data[i].label == my_data[(i+1)].label:
            continue
        else:
            best_thresh_candidates.append(float(my_data[i].features[feature_idx]))
    
    split_impurity = []

    for i in best_thresh_candidates:
        left_split, right_split = split_dataset(data, feature_idx, i)
        l_entro = calc_entropy(left_split)
        r_entro = calc_entropy(right_split)
        split_entro = (len(left_split)/n) * l_entro +(len(right_split)/n) * r_entro
        split_impurity.append(split_entro)
    
    best_info_gain = calc_entropy(data) - min(split_impurity)
    best_thresh = best_thresh_candidates[np.argmin(split_impurity)]
    return best_info_gain, best_thresh


# Q6. Implement the function `identify_best_split` which returns the best feature to split on for an input dataset, and also returns the corresponding threshold value.

# In[5]:


def identify_best_split(data):
    if len(data) < 2:
        return (None, None)
    best_feature = None
    best_thresh = None
#     your code goes here
    n = len(data)
    list_info_gain =[]
    
    for feature_idx in range(19):
        info_gain, thresh = calc_best_threshold(data, feature_idx)
        #print('the best thresh candidates for feature[%d] is: %s' %(feature_idx,best_thresh))
        #print('the info_gain for feature[%d] is: %s' %(feature_idx,info_gain))
        list_info_gain.append(info_gain)
    
    best_info_gain = np.max(list_info_gain)
    best_feature = np.argmax(list_info_gain)
    #print(list_info_gain)
    """
    for i in range (1,len(list_info_gain)):
        print(list_info_gain[i])
        if list_info_gain[i] > best_info_gain:
            best_info_gain = list_info_gain[i]
            best_feature = i
    """
    
    return best_info_gain, best_feature


# Q7. Implement the function `createLeafNode` which returns a `TreeNode` with `is_leaf=True` and `prediction` set to whichever classification occurs most in the dataset at this node.

# In[ ]:


def createLeafNode(data):
    node = TreeNode()
    node.is_leaf = True
    labels = []
    for i in range(len(data)):
        labels.append(data[i].label)
    if np.mean(labels) >= 0.5:
        node.prediction = 1
    else:
        node.prediction = 0
    return node


# Q8. Implement the `createDecisionTree` function. `max_levels` denotes the maximum height of the tree (for example if `max_levels = 1` then the decision tree will only contain the leaf node at the root. [Hint: this is where the recursion happens.]

# In[ ]:


def createDecisionTree(data, max_levels):
#     your code goes here
    
    if max_levels == 0:
        return createLeafNode(data)
    else:
        pure = True
        for i in data:
            if i.label != data[0].label:
                pure = False
                break
        if pure:
            return createLeafNode(data)
        else:
            _, best_feature_idx = identify_best_split(data)
            _, best_thresh = calc_best_threshold(data, best_feature_idx)
            left_split, right_split = split_dataset(data, best_feature_idx, best_thresh)
            node = TreeNode()
            node.feature_idx = best_feature_idx
            node.thresh_val = best_thresh
            node.is_leaf = False
            node.left_child = createDecisionTree(left_split, max_levels - 1)
            node.right_child = createDecisionTree(right_split, max_levels - 1)

    return node
    


# Q9. Given a test set, the function `calcAccuracy` returns the accuracy of the classifier. You'll use the `makePrediction` function for this.

# In[ ]:


def calcAccuracy(tree_root, data):
#     your code goes here
    n = len(data)
    total_correct = 0
    for i in range(n):
        label_pred = make_prediction(tree_root, data[i])
        if data[i].label == label_pred:
            total_correct += 1
    return total_correct / (float(n) + 1e-16)


# Q10. Keeping the `max_levels` parameter as 10, use 5-fold cross validation to measure the accuracy of the model. Print the recall and precision of the model. Also display the confusion matrix.

# In[ ]:


# edit the code here - this is just a sample to get you started

if __name__ == '__main__':
import time

d = get_data("messidor_features.txt")

# partition data into train_set and test_set
    idx = np.arange(len(d))
    np.random.shuffle(idx)
    ratio = 0.8
    cut = int(ratio * len(d))
    train_set = [d[i] for i in range(0, cut)]
    test_set = [d[i] for i in range(cut, len(d))]

    print ('Training set size:', len(train_set))
    print ('Test set size    :', len(test_set))


# create the decision tree
    start = time.time()
    tree = createDecisionTree(train_set, 10)
    end = time.time()
    print ('Time taken:', end - start)

# calculate the accuracy of the tree
    accuracy = calcAccuracy(tree, test_set)
    print ('The accuracy on the test set is ', str(accuracy * 100.0))
    #t.printTree()

