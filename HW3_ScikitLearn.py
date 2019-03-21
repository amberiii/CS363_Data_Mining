#!/usr/bin/env python
# coding: utf-8

# Enter name(s) here:

# # Assignment 3 : Using `scikit-learn`
# 
# Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python. In this assigment you'll explore how to train various classifiers using the `scikit-learn` library. The scikit-learn documentation can be found [here](http://scikit-learn.org/stable/documentation.html).
# 
# In this assignment we'll attempt to classify patients as either having or not having diabetic retinopathy, using the same Diabetic Retinopathy data set from your previous assignments. Recall that this dataset contains 1151 records and 20 attributes (some categorical, some continuous). You can find additional details about the dataset [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set).

# In[32]:


#You may add additional imports
import warnings
#warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


# Read the data from csv file
col_names = []
for i in range(20):
    if i == 0:
        col_names.append('quality')
    if i == 1:
        col_names.append('prescreen')
    if i >= 2 and i <= 7:
        col_names.append('ma' + str(i))
    if i >= 8 and i <= 15:
        col_names.append('exudate' + str(i))
    if i == 16:
        col_names.append('euDist')
    if i == 17:
        col_names.append('diameter')
    if i == 18:
        col_names.append('amfm_class')
    if i == 19:
        col_names.append('label')

data = pd.read_csv("messidor_features.txt", names = col_names)
print(data.shape)
data.head(10)


# ### A. Data prep

# Q1. All of the classifiers in `scikit-learn` require that you separate the feature columns from the class label column, so go ahead and do that first. You should end up with two separate data frames: one that contains all of the feature values and one that contains the class labels. 
# 
# Note: Later in this assignment, you may get a warning stating "a column-vector was passed when a 1d array was expected." This indicates that some function wants a _flat array_ of labels, rather than a 2D DataFrame of labels. You can go ahead and transform the labels into a flat array here by doing either `labels.values.ravel()` or `labels.iloc[:,0]`. And you can just use that flat array for everything.
# 
# Print the `shape` of your features data frame, the shape or len of your labels dataframe or array, and the `head` of the features data frame.

# In[42]:


# your code goes here
Label = pd.DataFrame(data=data['label'])
Features = pd.DataFrame(data=data.values[:,0:19],columns = data.columns[0:19])
len(Features)


# ### B. Decision Trees (DT) & Cross Validation

# **Train/Test Split**

# Q2. You can train a classifier using the holdout method by splitting your data into a  training set and a  test set, then you can evaluate the classifier on the held-out test set. 
# 
# Let's try this with a decision tree classifier. 
# 
# * Use `sklearn.model_selection.train_test_split` to split your dataset into training and test sets (do an 80%-20% split). Display how many records are in the training set and how many are in the test set.
# * Use `sklearn.tree.DecisionTreeClassifier` to fit a decision tree classifier on the training set. Use entropy as the split criterion. 
# * Now that the tree has been learned from the training data, we can run the test data through and predict classes for the test data. Use the `predict` method of `DecisionTreeClassifier` to classify the test data. 
# * Then use `sklearn.metrics.accuracy_score` to print out the accuracy of the classifier on the test set.

# In[122]:


# your code goes here
from sklearn.model_selection import train_test_split 
fea_train, fea_test, lab_train, lab_test = train_test_split(Features, Label, test_size = 0.2)
print('The size of training set is:',len(fea_train))
print('The size of test set is:',len(fea_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(fea_train, lab_train)
predict_label = clf.predict(fea_test)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(lab_test, predict_label, normalize=True)
print('The accuracy score is:',accuracy_score)


# Q3. Note that the DecisionTree classifier has many parameters that can be set. Try tweaking parameters like split criterion, max_depth, min_impurity_decrease, min_samples_leaf, min_samples_split, etc. to see how they affect accuracy. Print the accuracy of a few different variations.

# In[127]:


# your code goes here
clf_tw = tree.DecisionTreeClassifier(max_depth=10,min_samples_leaf=2)
clf_tw = clf_tw.fit(fea_train, lab_train)
pred_label = clf_tw.predict(fea_test)

from sklearn.metrics import accuracy_score
accuracy_tw = accuracy_score(lab_test, pred_label, normalize=True)
print(accuracy_tw)


# **Cross Validation**

# Q4. You have now built a decision tree and tested it's accuracy using the "holdout" method. But as discussed in class, this is not sufficient for estimating generalization accuracy. Instead, we should use Cross Validation to get a better estimate of accuracy. 
# 
# Use `sklearn.model_selection.cross_val_score` to perform 10-fold cross validation on a decision tree. You will pass the FULL dataset into `cross_val_score` which will automatically divide it into the number of folds you tell it to, train a decision tree model on the training set for each fold, and test it on the test set for each fold. It will return a numpy array with the accuracy out of each fold. Average these accuracies to print out the generalization accuracy of the model.

# In[45]:


# your code goes here
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
score = cross_val_score(clf,Features,Label,cv=10)
print(score)
print(score.mean())


# **Nested Cross Validation**

# Q5. Now we want to tune our model to use the best parameters to avoid overfitting to our training data. Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters (hyperparameters) specified in a grid. 
# * Use `sklearn.model_selection.GridSearchCV` to find the best `max_depth`, `max_features`, and `min_samples_leaf` for your tree. Use a 5-fold-CV and 'accuracy' for the scoring criteria.
# * Try the values [5,10,15,20] for `max_depth` and `min_samples_leaf`. Try [5,10,15] for `max_features`. 
# * Print out the best value for each of the tested parameters (`best_params_`).
# * Print out the accuracy of the model with these best values (`best_score_`).

# In[130]:


# your code goes here
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':[5,10,15,20], 'min_samples_leaf':[5,10,15,20],'max_features':[5,10,15]}
clf = tree.DecisionTreeClassifier(criterion='entropy')

clf_grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5)
clfresult = clf_grid.fit(Features,Label)
print('The best values are:\n',clfresult.best_params_)
print('The accuracy is:\n',clfresult.best_score_)


# Q6. What you did in Q5 performed the _inner_ loop of a nested CV (no test set was held out). What you did in Q4 performed an _outer_ loop of CV (holds out a test set). Now we need to combine them to perform the nested cross-validation that we discussed in class. To do this, you'll need to pass the a `GridSearchCV` into a `cross_val_score`. 
# 
# What this does is: the `cross_val_score` splits the data in to train and test sets for the first outer fold, and it passes the train set into `GridSearchCV`. `GridSearchCV` then splits that set into train and validation sets for k number of folds (the inner CV loop). The hyper-parameters for which the average score over all inner iterations is best, is reported as the `best_params_`, `best_score_`, and `best_estimator_`(best decision tree). This best decision tree is then evaluated with the test set from the `cross_val_score` (the outer CV loop). And this whole thing is repeated for the remaining k folds of the `cross_val_score` (the outer CV loop). 
# 
# That is a lot of explanation for a very complex (but IMPORTANT) process, which can all be performed with a single line of code!
# 
# Be patient for this one to run. The nested cross-validation loop can take some time. A [ * ] next to the cell indicates that it is still running.
# 
# Print the accuracy of your tuned, cross-validated model. This is the official accuracy that you would report for your model.

# In[131]:


# your code goes here
print(cross_val_score(clf_grid,Features,Label,cv=10))


# ### C. Naive Bayes (NB) & Evaluation Metrics
# 
# `sklearn.naive_bayes.GaussianNB` implements the Gaussian Naive Bayes algorithm for classification. This means that the liklihood of continuous features is estimated using a Gaussian distribution. (Refer to slide 13 of the Naive Bayes powerpoint notes.)

# Q7. Create a `sklearn.naive_bayes.GaussianNB` classifier. Use `sklearn.model_selection.cross_val_score` to do a 10-fold cross validation on the classifier. Display the accuracy.

# In[48]:


# your code goes here
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
Label = np.ravel(Label)
NB_clf = clf.fit(Features, Label)
print(NB_clf)
score_NB = cross_val_score(NB_clf,Features,Label,cv=10)
print(score_NB)


# Q8. `cross_val_score` returns the scores of every test fold. There is another function called `cross_val_predict` that returns predicted y values for every record in the test fold. In other words, for each element in the input, `cross_val_predict` returns the prediction that was obtained for that element when it was in the test set. 
# 
# * Use `cross_val_predict` and `sklearn.metrics.confusion_matrix` to print the confusion matrix for the classifier.
# 
# * Sckit-learn also provides a useful function `sklearn.metrics.classification_report` for evaluating the classifier on a per-class basis. It is a summary of the precision, recall, and F1 score for each class (and support is just the actual class count). Display the classification report for your Naive Bayes classifier.

# In[49]:


# your code goes here
from sklearn.model_selection import cross_val_predict
Label_pred = cross_val_predict(NB_clf, Features, Label, cv=10)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Label, Label_pred))

from sklearn.metrics import classification_report
print(classification_report(Label, Label_pred))


# Q9. Using `sklearn.metrics.roc_curve` plot a ROC curve for the Naive Bayes classifier. Also calculate the area under the curve (AUC) using `sklearn.metrics.roc_auc_score`.
# 
# * We will just do this on a single holdout test set (because it gets more complicated to put this inside of a cross-validation). So, split your data into training and test sets using `sklearn.model_selection.train_test_split`. Do an 80/20 split.
# * Fit the Naive Bayes classifier to the training data by calling the `fit` method on the trainng data.
# * Now call the `predict_proba` method on your classifier and pass in the test data. This will return a 2D numpy array with one row for each datapoint in the test set and 2 columns. Column index 0 is the probability that this datapoint is in class 0, and column index 1 is the probability that this datapoint is in class 1.
# * We are going to say that class 1 (having the disease) is the rare/positive class. To create a ROC curve, pass the actual Y labels and the probabilites of class 1 (column index 1 out of your predict_proba result) into `sklearn.metrics.roc_curve`
# * Pass the FPR and TPR that `roc_curve` returns into the plotting code that we have provided you.
# * Print the AUC (area under the curve) by using `sklearn.metrics.roc_auc_score`

# In[50]:


clf = GaussianNB()
lab_train = np.ravel(lab_train)
nb_clf = clf.fit(fea_train,lab_train)
cl = nb_clf.predict_proba(fea_test)
class_1 = cl[:,1]
lab_test = np.ravel(lab_test)

from sklearn import metrics
fpr, tpr, thresh = metrics.roc_curve(lab_test, class_1)


# In[51]:


# your code goes here

#replace these fpr and tpr with the results of your roc_curve
#fpr, tpr = [], []

# Do not change this code! This plots the ROC curve.
# Just replace the fpr and tpr above with the values from your roc_curve
plt.plot([0,1],[0,1],'k--') #plot the diagonal line
plt.plot(fpr, tpr, label='NB') #plot the ROC curve
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Naive Bayes')
plt.show()

from sklearn.metrics import roc_auc_score
print('The AUC is:',roc_auc_score(lab_test, class_1))


# ### D. k-Nearest Neighbor (KNN) & Pipelines 

# For some classification algorithms, scaling of the data is critical (like KNN, SVM, Neural Nets). For other classification algorithms, data scaling is not necessary (like Naive Bayes and Decision Trees). _Take a minute to think about why this is the case!!_ But using scaled data with an algorithm that doesn't explicitly need it to be scaled does not hurt the results of that algorithm.

# Q10. The distance calculation method is central to the KNN algorithm. By default, `KNeighborsClassifier` uses  Euclidean distance as its metric (but this can be changed). Because of the distance calculations, it is critical to scale the data before running Nearest Neighbor!
# 
# We discussed why dimensionality reduction may also be needed with KNN because of the curse of dimensionality. So we may want to also perform a dimensionality reduction with PCA before running KNN. PCA should only be performed on scaled data! (Remember that you can also reduce dimensionality by performing feature selection and feature engineering.) 
# 
# An important note about scaling data and dimensionality reduction is that they should only be performed on the **training** data, then you transform the test data into the scaled, PCA space that was found on the training data. (Refer to the concept of [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/).)
# 
# So when you are doing cross-validation, the scaling and PCA needs to happen *inside of your CV loop*. This way, it is performed on the training set for the first fold, then the test set is put into that space. On the second fold, it is performed on the trainng set for the second fold, and the test set is put into that space. And so on for the remaining folds. 
# 
# In order to do this with scikit-learn, you must create what's called a `Pipeline` and pass that in to the cross validation. This is a very important concept for Data Mining and Machine Learning, so let's practice it here.
# 
# Do the following:
# * Create a `sklearn.preprocessing.StandardScaler` object to standardize the dataset’s features (mean = 0 and variance = 1). (Do not call `fit` on it yet. Just create the `StandardScaler` object.)
# * Create a `sklearn.decomposition.PCA` object to perform PCA dimensionality reduction. (Do not call `fit` on it yet. Just create the `PCA` object.)
# * Create a `sklearn.neighbors.KNeighborsClassifier`. The number of neighbors defaults to 5 (k=5). Go ahead and change it to 7. (Do not call `fit` on it yet. Just create the `KNeighborsClassifier` object.)
# * Create a `sklearn.pipeline.Pipeline` object and set the `steps` to the scaler, the PCA, and the KNN objects that you just created. 
# * Pass the `pipeline` object in to a `cross_val_score` as the estimator, along with the features and the labels, and use a 5-fold-CV. 
# 
# In each fold of the cross validation, the training phase will use _only_ the training data for scaling, PCA, and training the model. Then the testing phase will scale & transform the test data into the PCA space (found on the training data) and run the test data through the trained classifier, to return an accuracy measurement for each fold. Print the average accuracy across all 5 folds. 

# In[57]:


# your code goes here
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler.fit(Features)
#sd_features = scaler.transform(Features)

from sklearn.decomposition import PCA
pca = PCA()
#pca.fit()
#pca_features = pca.transform(sd_features)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
#knn_clf = neigh.fit(pca_features,Label)

from sklearn.pipeline import Pipeline
ppl = Pipeline([('scaler', scaler), ('PCA', pca),('KNN', neigh)])

average_accuracy = cross_val_score(ppl,Features,Label,cv=5)
print(average_accuracy.mean())


# Q11. Another important part of KNN is choosing the best number of neighbors (tuning the hyperparameter, k). We can use nested cross validation to do this. Let's try k values from 1-25 to find the best one. 
# 
# We _also_ want to find the best number of dimensions to project down onto using PCA. We can use nested cross validation to do this as well. Let's try from 5-19 dimensions.
# 
# * Starter code is provided to create the "parameter grid" to search. You will need to change this code! Where I have "knn__n_neighbors", this indicates that I want to tune the "n_neighbors" parameter in the "knn" part of the pipeline. When you created your pipeline above, you named the KNN part of the pipeline with a string. You should replace "knn" in the param_grid below with whatever you named your KNN part of the pipeline: **<replace_this>__n_neighbors.** Do the same for the PCA part of the pipeline.
# * Create a `sklearn.model_selection.GridSearchCV` and pass in the pipeline, the param_grid, and set it to a 5-fold-CV.
# * Now, on that `GridSearchCV` object, call `fit` and pass in the features and labels.
# * Show the best number of dimensions and best number of neighbors for this dataset by printing the `best_params_` from the `GridSearchCV`.
# * Also print the accuracy when using this best number of dimensions and neighbors by printing the `best_score_` from the `GridSearchCV`.
# 
# Be patient, this can take some time to run. It is trying every combination of dimensions from 5-19 with every k from 1-25! A [ * ] next to the cell indicates that it is still running.

# In[55]:


'''
On the "pca" part of the pipeline, 
tune the n_components parameter,
by trying the values 1-19.

On the "knn" part of the pipeline, 
tune the n_neighbors parameter,
by trying the values 1-30.
'''
param_grid = {
    'pca__n_components': list(range(5, 19)),
    'knn__n_neighbors': list(range(1, 25))
}

# your code goes here
neigh = KNeighborsClassifier()
ppl_2 = Pipeline([('scaler', scaler), ('pca', pca),('knn', neigh)])
#ppl = Pipeline([('pca', pca),('knn', neigh)])
grid_ppl = GridSearchCV(ppl_2, param_grid, cv=5)
#print(grid_ppl)
results = grid_ppl.fit(Features,Label)
print('The dimentions and best numbers of neighbors are:',results.best_params_)


# Q12. In Q11, we did not hold out a test set. The accuracy reported out is on the _validation_ set. So now we need to wrap the whole process in another cross-validation to perform a nested cross-validation and report the accuarcy of this KNN model on unseen test data. This is the official accuracy you would report on this model.
# 
# You'll need to pass the `GridSearchCV` into a `cross_val_score`, just as you did with the decision tree. Use a 5-fold-CV for the outer loop. 
# 
# Again, be patient for this one to run. The nested cross-validation loop can take some time. It is doing what it did above in Q11 five times. A [ * ] next to the cell indicates that it is still running. (Just for comparison, mine takes about 2 mins to run and the fan revs up so it sounds like my computer is going to explode. All computers are different, so yours could take shorter or longer...)
# 
# <img src="model_is_training.png" width="250">

# In[58]:


# your code goes here
official_accuracy = cross_val_score(grid_ppl,Features,Label,cv=5)
print('The average accuracy of the final 5-fold-VS is:',official_accuracy.mean())


# ### E. Support Vector Machines (SVM)

# Q13. Now put it all together with an SVM. 
# * Create a `pipeline` that includes scaling, PCA, and an `sklearn.svm.SVC`.
# * Create a parameter grid that tries number of dimensions from 5-19 and SVM kernels `linear`, `rbf` and `poly`.
# * Create a `GridSearchCV` for the inner CV loop. Use a 5-fold CV.
# * Run a `cross_val_predict` with a 10-fold CV for the outer loop. 
# * Print out the accuracy and the classification report of using an SVM classifier on this data.

# In[61]:


# your code goes here
from sklearn.svm import SVC
svc_clf = SVC()
param_grid = {
    'pca__n_components': list(range(5, 19)),
    'svc__kernel':('linear', 'rbf','poly')
}
ppl_svc = Pipeline([('scaler', scaler), ('pca', pca),('svc', svc_clf)])
grid_ppl_svc = GridSearchCV(ppl_svc, param_grid, cv=5)
svc_results = grid_ppl_svc.fit(Features,Label)
label_pred_svm = grid_ppl_svc.predict(Features)

svm_accuracy = cross_val_score(grid_ppl_svc,Features,Label,cv=10)
svm_report = classification_report(Label, label_pred_svm)
print('The accuracy of SVM classifier is:',svm_accuracy.mean())
print('The classification report is:',svm_report)


# ### F. Neural Networks (NN)

# Q14. Train a multi-layer perceptron with a single hidden layer using `sklearn.neural_network.MLPClassifier`. 
# * Create a pipeline with scaling and a neural net. (No PCA on this one. But scaling is critical to neural nets.)
# * Use `GridSearchCV` with 5 fold cross validation to find the best hidden layer size and the best activation function. 
# * Try values of `hidden_layer_sizes` ranging from `(30,)` to `(60,)` by increments of 10.
# * Try activation functions `logistic`, `tanh`, `relu`.
# * Wrap your `GridSearchCV` in a 5-fold `cross_val_score` and report the accuracy of your neural net.
# 
# Be patient, as this can take a few minutes to run. You may get ConvergenceWarnings as it runs - that is fine.

# In[71]:


# your code goes here
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier()
param_grid = {
    'mlp__hidden_layer_sizes': ((30,),(40,),(50)),
    'mlp__activation':('logistic', 'tanh','relu')
} 
ppl_mlp = Pipeline([('scaler', scaler), ('mlp', mlp_clf)])
grid_ppl_mlp = GridSearchCV(ppl_mlp, param_grid, cv=5)
mlp_accuracy = cross_val_score(grid_ppl_mlp,Features,Label,cv=5)
print('The accuracy of the MLP classifier is:',mlp_accuracy.mean())
#The accuracy of the MLP classifier is: 0.7375983436853002


# ### G. Ensemble Classifiers
# 
# Ensemble classifiers combine the predictions of multiple base estimators to improve the accuracy of the predictions. One of the key assumptions that ensemble classifiers make is that the base estimators are built independently (so they are diverse).

# **Random Forests**
# 
# Q15. Use `sklearn.ensemble.RandomForestClassifier` to classify the data. Scaling the data is not necessary for Decision Trees (take a minute to think about why). So, no need for a pipeline here.
# 
# Use a `GridSearchCV` with a 5-fold CV to tune the hyperparameters to get the best results. 
# * Try `max_depth` ranging from 35-55
# * Try `min_samples_leaf` of 8, 10, 12
# * Try `max_features` of `"sqrt"` and `"log2"`
# 
# Wrap your GridSearchCV in a cross_val_score with 5-fold CV to report the accuracy of the model.
# 
# Be patient, this can take a few minutes to run.

# In[73]:


# your code goes here
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
param_grid = {
    'max_depth': list(range(35,55)),
    'min_samples_leaf':(8,10,12),
    'max_features':('sqrt','log2')
} 

grid_rf = GridSearchCV(rf_clf, param_grid, cv=5)
rf_accuracy = cross_val_score(grid_rf,Features,Label,cv=5)
print('The accuracy of the ramdom forest model is:',rf_accuracy.mean())


# **AdaBoost**
# 
# Random Forests are a kind of averaging ensemble classifier, where several estimators are built independently and then to average their predictions (by taking a vote). There is another method of training ensemble classifiers called *boosting*. Here the classifiers are trained sequentially and each time the sampling of the training set depends on the performance of previously generated models.
# 
# Q16. Evaluate a `sklearn.ensemble.AdaBoostClassifier` classifier on the data. By default, `AdaBoostClassifier` uses decision trees as the base classifiers (but this can be changed). Use 150 base classifiers to make an `AdaBoostClassifier` and evaluate it's accuracy with a 5-fold-CV.

# In[76]:


# your code goes here
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(n_estimators=150)
ab_accuracy = cross_val_score(ab_clf,Features,Label,cv=5)
print('The accuracy of AdaBoost classifier is:',ab_accuracy.mean())


# ### H. Build your final model

# Now you have tested all kinds of classifiers on this data. Some have performed better than others. 
# 
# Q17. We may not want to deploy any of these models in the real world to actually diagnose patients because the accuracies are not high enough. What can we do to improve the accuracy rates? Answer as a comment:

# In[ ]:


'''
Answer here as a comment.
1. add more data and features
2. treating missing and outlier values
3. do feature selection 
3. use different class of models to compare and tune

'''


# Q18. Let's say we *did* get to the point where we had a model with very high accuracy and we want to deploy that model and use it for real-world predictions.
# 
# * Let's say we're going to deploy our SVM classifier.
# * We need to make one final version of this model, where we use ALL of our available data for training (we do not hold out a test set this time, so no outer cross-validation loop). 
# * We need to tune the parameters of the model on the FULL dataset, so copy the code you entered for Q13, but remove the outer cross validation loop (remove `cross_val_score`). Just run the `GridSearchCV` by calling `fit` on it and passing in the full dataset. This results in the final trained model with the best parameters for the full dataset. You can print out `best_params_` to see what they are.
# * The accuracy of this model is what you assessed and reported in Q13.
# 
# 
# * Use the `pickle` package to save your model. We have provided the lines of code for you, just make sure your final model gets passed in to `pickle.dump()`. This will save your model to a file called finalized_model.sav in your current working directory. 

# In[78]:


import pickle

# your code goes here
from sklearn.svm import SVC
svc_clf = SVC()
param_grid = {
    'pca__n_components': list(range(5, 19)),
    'svc__kernel':('linear', 'rbf','poly')
}
ppl_svc = Pipeline([('scaler', scaler), ('pca', pca),('svc', svc_clf)])
grid_ppl_svc = GridSearchCV(ppl_svc, param_grid, cv=5)
svc_results = grid_ppl_svc.fit(Features,Label)

print('The best parameters are',svc_results.best_params_)
#replace this final_model with your final model
final_model = grid_ppl_svc

filename = 'finalized_model.sav'
pickle.dump(final_model, open(filename, 'wb'))


# Q19. Now if someone wants to use your trained, saved classifier to classify a new record, they can load the saved model and just call predict on it. 
# * Given this new record, classify it with your saved model and print out either "Negative for disease" or "Positive for disease."

# In[103]:


# some time later...

# use this as the new record to classify
record = [ 0.05905386, 0.2982129, 0.68613149, 0.75078865, 0.87119216, 0.88615694,
  0.93600623, 0.98369184, -0.47426472, -0.57642756, -0.53115361, -0.42789774,
 -0.21907738, -0.20090532, -0.21496782, -0.2080998, 0.06692373, -2.81681183,
 -0.7117194 ]

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# your code goes here
record = np.stack(record)
n_record = np.transpose(record[:,None])
pred = loaded_model.predict(n_record)

if pred == 0:
    print("Negative for disease.")
else:
    print('Positive for disease.')

