#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

fp = 0
fn = 0
tp = 0
tn = 0
for i in range(0, len(labels_test), 1):
    if labels_test[i] == 1.0: # positive
        if pred[i] == 1.0: # true
            tp += 1
        else: # false
            fp += 1
    else: # negative
        if pred[i] != 1.0: # false
            tn += 1
        else:
            fn += 1
print "FP: ", fp
print "FN: ", fn
print "TP: ", tp
print "TN: ", tn
print "Accuracy: ", acc
print "Number of persons predicted as POIs: ", (tp+fp)
print "Total number of people in test set: ", (tp+fp+tn+fn)
print "Accuracy if everyone would be predicted '0': ", (tp*1.0)/(tp+fp+tn+fn)
print "Precision: ", (tp*1.0)/(tp+fp)
print "Recall: ", (tp*1.0)/(tp+fn)