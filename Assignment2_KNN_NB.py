import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from random import shuffle

class DataPoint:
    def __str__(self):
        return "< " + str(self.label) + ": " + str(self.features) + " >"
    def __init__(self, label, features):
        self.label = label # the classification label of this data point
        self.features = features


def get_data(filename):
    data = []
#     your code goes here
    with open(filename) as f:
        for line in f:
            tp = line.strip().split(',')
            label = int(tp[19])
            features = [float(tp[i]) for i in range(19)]
            data.append(DataPoint(label,features))
    return data


# Q1. Implement a Naive Bayes classifier
# Measure the accuracy of your classifier using 5-fold cross validation and display the confusion matrix.
# Also print the precision and recall for class label 1 (patients that have been diagnosed with the disease).
def NaiveBayesClassifier(data):
    n = len(data)
    num_1 = 0
    prior_prob_1 = 0
    for i in data:
        num_1 += i.label
    prior_prob_1 = num_1/n
    features = []
    label = []
    for i in data:
        features.append(i.features)
        label.append(i.label)
    features = np.stack(features)    
    label = np.stack(label)
    
    new_data = np.column_stack([label,features])
    feature_dict = {}
    thresh_1_dict = {}
    thresh_2_dict = {}

    for j in range(1,20):        
        if j == 1 or j == 2 or j == 19:
            n_feature_j = np.zeros([2,2])
            for i in new_data:
                if i[0] == 1:
                    if i[j] == 1:
                        n_feature_j[0,0] += 1
                    if i[j] == 0:
                        n_feature_j[0,1] += 1     
                if i[0] == 0:
                    if i[j] == 1:
                        n_feature_j[1,0] += 1
                    if i[j] == 0:
                        n_feature_j[1,1] += 1      
            #print('the j th feature number:',j,n_feature_j)
            prob_feature_j = np.zeros([2,2])
            prob_feature_j[0,:] = (n_feature_j[0,:] + 1) / (num_1+2) 
            prob_feature_j[1,:] = (n_feature_j[1,:] + 1) / (n-num_1+2)
            #print('the %d th feature probablity matrix is:',j,prob_feature_j)
            feature_dict[j] = prob_feature_j
            
        else:
            thresh_1, thresh_2 = bin(new_data[:,j])
            thresh_1_dict[j] = thresh_1
            thresh_2_dict[j] = thresh_2
            n_feature_j = np.zeros([2,3])
            for i in new_data:
                if i[0] == 1:
                    if i[j] <= thresh_1:
                        n_feature_j[0,0] += 1
                    if i[j] > thresh_1 and i[j] < thresh_2:
                        n_feature_j[0,1] += 1
                    if i[j] >= thresh_2:
                        n_feature_j[0,2] += 1
                if i[0] == 0:
                    if i[j] <= thresh_1:
                        n_feature_j[1,0] +=1
                    if i[j] >= thresh_1 and i[j] <= thresh_2:
                        n_feature_j[1,1] +=1
                    if i[j] >= thresh_2:
                        n_feature_j[1,2] +=1
            #print('the j th feature number:',j,n_feature_j,)
            prob_feature_j = np.zeros([2,3])
            prob_feature_j[0,:] = (n_feature_j[0,:] + 1) / (num_1+2)
            prob_feature_j[1,:] = (n_feature_j[1,:] + 1) / (n-num_1+2)
            #print('the %d th feature probablity matrix is: ',j,prob_feature_j)
            feature_dict[j] = prob_feature_j
    return thresh_1_dict, thresh_2_dict, feature_dict

def make_prediction(data):
    features = np.array(data.features)
    label = np.array([data.label])
    new_data = np.concatenate((label,features))
    
    prob_label_1 = []
    prob_label_0 = []
    if new_data[1] == 1:
        prob_label_1.append(feature_dict[1][0,0])
        prob_label_0.append(feature_dict[1][1,0])
    else:
        prob_label_1.append(feature_dict[1][0,1])
        prob_label_0.append(feature_dict[1][1,1])
    if new_data[2] ==1:
        prob_label_1.append(feature_dict[2][0,0])
        prob_label_0.append(feature_dict[2][1,0])
    else:
        prob_label_1.append(feature_dict[2][0,1])
        prob_label_0.append(feature_dict[2][1,1])
    for j in range(3,19):
        thresh_1 = thresh_1_dict[j]
        thresh_2 = thresh_2_dict[j]
        if new_data[j] <= thresh_1:
            prob_label_1.append(feature_dict[j][0,0])
            prob_label_0.append(feature_dict[j][1,0])
        if thresh_1 < new_data[j] < thresh_2:
            prob_label_1.append(feature_dict[j][0,1])
            prob_label_0.append(feature_dict[j][1,1])
        if new_data[j] > thresh_2:
            prob_label_1.append(feature_dict[j][0,2])
            prob_label_0.append(feature_dict[j][1,2])
    if new_data[19] == 1:
        prob_label_1.append(feature_dict[19][0,0])
        prob_label_0.append(feature_dict[19][1,0])
    else:
        prob_label_1.append(feature_dict[19][0,1])
        prob_label_0.append(feature_dict[19][1,1])
    '''
    prob_1 = 1
    prob_0 = 1
    for i in prob_label_1:
        prob_1 = i * prob_1
    '''
    if np.prod(prob_label_1 >= prob_label_0):
        return 1
    else:
        return 0
    

    

    
def bin(feature): 

    thresh_1 = max(feature)/3
    thresh_2 = 2 * thresh_1
    return thresh_1,thresh_2

#main Function  
if __name__ == '__main__':
    d = get_data("messidor_features.txt")
    
    idx = np.arange(len(d))
    np.random.shuffle(idx)
    ratio = 0.8
    cut = int(ratio * len(d))
    train_set = [d[idx[i]] for i in range(0, cut)]
    test_set = [d[idx[i]] for i in range(cut, len(d))]
    print ('Training set size:', len(train_set))
    print ('Test set size    :', len(test_set))

    thresh_1_dict, thresh_2_dict, feature_dict = NaiveBayesClassifier(train_set)
    predict_label = []
    for data in test_set:
        predict_label.append(make_prediction(data))
    import ipdb; ipdb.set_trace()
    
    

    