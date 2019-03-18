import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from random import shuffle
import argparse
import time

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

def calcAccuracy(predict_label, test_data):
    n = len(test_data)
    total_correct = 0
    
    for i in range(n):
        if predict_label[i] == test_data[i].label:           
            total_correct += 1
    return total_correct / (float(n) + 1e-16)

#Q2. Normalize the dataset so that each feature value lies between [0-1].
def Normalize(data):
    features = []
    label = []
    for i in data:
        features.append(i.features)
        label.append(i.label)
    features = np.stack(features)    
    label = np.stack(label)
    n = len(data)
    norm_features = np.zeros([n,19])
    norm_features = (features - features.min(0, keepdims=True)) / (features.max(0, keepdims=True) - features.min(0, keepdims=True) + 1e-16)
    '''
    if 0:
        for i in range(n):
            for j in range(19):
                if max(features[:,j]) != min(features[:,j]):
                    norm_features[i,j] = (features[i,j] - min(features[:,j])) / (max(features[:,j]) - min(features[:,j]))  
    '''            
    norm_data = np.column_stack([label,norm_features])
    return norm_data

#Q3. Build your KNN classifier. 
def KNN_Classifier(k,v_data,data):
    n = len(data)
    dist = np.zeros([n,2])
    dst = np.linalg.norm(v_data[1:][None, :] - data[:, 1:], axis=1)
    idx = np.argsort(dst)
    label = 0
    for i in range(k):
        label += data[idx[i],0]
    if label.mean() > 0.5:
        return 1
    else:
        return 0
    '''
    if 0:
        for i in range(n):
            
            dist[i,0] = data[i,0]
            distance = 0
            for j in range(1,20):
                distance += (v_data[j]-data[i,j]) ** 2
            dist[i,1] = math.sqrt(distance)
        t = np.argsort(dist[:,1])
        n_label_1 = 0
        n_label_0 = 0
        for i in range(k):
            if dist[t[i],0] == 0:
                n_label_0 += 1
            if dist[t[i],0] == 1:
                n_label_1 += 1
    #print(n_label_0,n_label_1)
    
    if n_label_0 <= n_label_1:
        return 1
    else:
        return 0
    '''
def calcAccuracy(predict_label, test_data):
    n = len(test_data)
    total_correct = 0
    
    for i in range(n):
        if predict_label[i] == test_data[i].label:          
            total_correct += 1
    return total_correct / (float(n) + 1e-16)

def ConfusionMatrix(predict_label, test_data):
    n = len(test_data)
    confusion_matrix = np.zeros([2,2])
    n_1_1 = n_1_0 = n_0_1 = n_0_0 = 0
    
    for i in range(n):
        if predict_label[i] == test_data[i].label == 1:
            n_1_1 += 1
        if predict_label[i] == test_data[i].label == 0:
            n_0_0 += 1
        if predict_label[i] == 1 and test_data[i].label == 0:
            n_0_1 += 1
        if predict_label[i] == 0 and test_data[i].label == 1:
            n_1_0 += 1
    
    confusion_matrix[0,0] = n_1_1
    confusion_matrix[0,1] = n_1_0
    confusion_matrix[1,0] = n_0_1
    confusion_matrix[1,1] = n_0_0
    return confusion_matrix
            

#Q4. Find the best value of k using 5-fold cross validation. 
#    In each fold of CV, divide your data into a training set and a validation set. 
#    Try k ranging from 1 to 10 and plot the accuracies using 5-fold CV. 
#    Use this plot to identify the best value of k (provide reasoning).
    
def KNN_accuracy(d,k):
    #randomize the index of the data => randomize the data
    idx = np.arange(len(d))
    #np.random.shuffle(idx)

    # 5-fold cross validation
    N = 5
    accuracy_k = 0
    Confusion_Matrix = np.zeros([2,2])
    for q in range(1, N+1):
        st = time.time()
        cut_1 = int((0.2 * (q-1))*len(d))
        cut_2 = int((0.2 * q) * len(d))
        validation_set = []
        training_set = []       
        for i in range(cut_1, cut_2):
            validation_set.append(d[idx[i]])
        for i in range(0, cut_1):
            training_set.append(d[idx[i]])
        for i in range(cut_2,len(d)):
            training_set.append(d[idx[i]])      
        #print ('Training set size:', len(training_set))
        #print ('Test set size:', len(validation_set))
        #print('time 0: %f' % (time.time() - st))
        st = time.time()
        norm_train_set = Normalize(training_set)
        norm_valid_set = Normalize(validation_set)
        #print('time 1: %f' % (time.time() - st))
        #read normalized data in the class of numpy into the KNN classifier
        st = time.time()
        predict_label = []
        for i in range(len(validation_set)):
            predict_label.append(KNN_Classifier(k,norm_valid_set[i],norm_train_set))
        confusion_matrix = ConfusionMatrix(predict_label,validation_set)
        Confusion_Matrix = Confusion_Matrix + confusion_matrix
        #print('time 2: %f' % (time.time() - st))
        # calculate the accuracy of k-th fold
        st = time.time()
        accuracy = calcAccuracy(predict_label, validation_set)
        accuracy_k = accuracy_k + accuracy
        #print('time 3: %f' % (time.time() - st))
    return accuracy_k/N, Confusion_Matrix

#main Function
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str,help="display a square of a given number")
    #import ipdb;ipdb.set_trace()
    args = parser.parse_args()

    d = get_data("messidor_features.txt")

    #Naive Bayes Classifier
    if args.name == 'nb':
        #randomize the index
        idx = np.arange(len(d))
        #np.random.shuffle(idx)
        Confusion_Matrix = np.zeros([2,2])
        # 5-fold cross validation
        for k in range(1,6):
            cut_1 = int((0.2 * (k-1))*len(d))
            cut_2 = int((0.2 * k) * len(d))
            test_set = []
            train_set = []       
            for i in range(cut_1, cut_2):
                test_set.append(d[idx[i]])
            for i in range(0, cut_1):
                train_set.append(d[idx[i]])
            for i in range(cut_2,len(d)):
                train_set.append(d[idx[i]])      
        #print ('Training set size:', len(train_set))
        #print ('Test set size:', len(test_set))
            thresh_1_dict, thresh_2_dict, feature_dict = NaiveBayesClassifier(train_set)
            predict_label = []
            for data in test_set:
                predict_label.append(make_prediction(data))

        # calculate the accuracy of the NB classifier
            accuracy = calcAccuracy(predict_label, test_set)
            confusion_matrix = ConfusionMatrix(predict_label,test_set)
            Confusion_Matrix = Confusion_Matrix + confusion_matrix
            print ('The accuracy of the %d th-fold test set is %s' % (k,str(accuracy * 100.0)))
        print('The Confusion Matrix is:',Confusion_Matrix) 

#Q5. Now measure the accuracy of your classifier using 5-fold cross validation. 
#    In each fold of this CV, divide your data into a training set and a test set. 
#    The training set should get sent through your code for Q4, resulting in a value of k to use. 
#    Using that k, calculate an accuracy on the test set. 
#    You will average the accuracy over all 5 folds to obtain the final accuracy measurement.        

    elif args.name == 'knn':
        accuracy_k = []       
        for k in range(1,11):
            accuracy, Confusion_Matrix = KNN_accuracy(d,k)
            accuracy_k.append(accuracy)
            print('the accuracy for %d-NN Classifier is: %f' % (k,accuracy_k[-1]))
        x = np.arange(10)+1
        plt.plot(x,accuracy_k)
        plt.show()
        #According to the plot, we take k=2 as the best k value.
        print('The best k value is 2 and its corresponding accuracy is:',accuracy_k[1])

        accuracy_2, Confusion_Matrix_2 = KNN_accuracy(d,2)
        print('The Confusion Matrix is:',Confusion_Matrix_2)








    