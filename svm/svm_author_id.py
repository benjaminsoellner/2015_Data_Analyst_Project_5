#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

clf = SVC(kernel="rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

print "Accuracy: ", acc
print "Training time: ", round(t1-t0,3), "s"
print "Prediction for element 10:", pred[10]
print "Prediction for element 26:", pred[26]
print "Prediction for element 50:", pred[50]
print "Number of elements classified as 1:", sum(pred)

#########################################################
### your code goes here ###

#########################################################


