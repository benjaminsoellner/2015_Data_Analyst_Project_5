import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################

from sklearn import tree
clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)

clf50 = tree.DecisionTreeClassifier(min_samples_split=50)
clf50 = clf50.fit(features_train, labels_train)
pred50 = clf50.predict(features_test)

from sklearn.metrics import accuracy_score
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred50, labels_test)

print "Accuracy min_samples_split=2: ", acc_min_samples_split_2
print "Accuracy min_samples_split=50: ", acc_min_samples_split_50

### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
