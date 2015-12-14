#!/usr/bin/python

import matplotlib.pyplot as plt
import getopt
import sys
import math

from sklearn.metrics import accuracy_score
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

optlist, _ = getopt.gnu_getopt(sys.argv, 'kar')
options = [o[0] for o in optlist]

print "Training set: ", len(features_train)
print "Test set: ", len(features_test)

clf = None 
if "-k" in options:
    from sklearn.neighbors import KNeighborsClassifier
    # good to have k be sqrt of observations
    k = int(math.floor(math.sqrt(len(features_train)+len(features_test))/3))
    k = k+((k+1)%2) # make sure k is odd number
    print "Using k-Nearest Neighbor Classifier with k: ", k
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
elif "-a" in options:
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import tree
    clfbase = tree.DecisionTreeClassifier(min_samples_split=40)
    print "Using AdaBoost with Decision Trees (50 iterations)"
    clf = AdaBoostClassifier(base_estimator=clfbase, n_estimators=50, algorithm='SAMME')
elif "-r" in options:
    from sklearn.ensemble import RandomForestClassifier
    print "Using Random Forest Classifier"
    clf = RandomForestClassifier(n_estimators=20, criterion='gini', min_samples_split=20)

    
if clf is not None:
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print "Accuracy score: ", acc
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
else:
    print "Usage: "
    print "  python your_algorithm.py -k    Perform k nearest neighbours"
    print "  python your_algorithm.py -a    Perform ada-boost"
    print "  python your_algorithm.py -r    Perform random forest"
