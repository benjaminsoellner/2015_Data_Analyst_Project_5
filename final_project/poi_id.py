#!/usr/bin/python

import getopt
import pickle
import sys
import math

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

optlist, _ = getopt.getopt(sys.argv[1:], 'g:')
options = {o[0]: o[1] for o in optlist}

from poi_id_gui import PoiIdGui
gui = PoiIdGui(options["-g"] if "-g" in options else None)


##
## Task 1: Select what features you'll use.
##

# Available to us are:
#
# financial features: ['salary', 'deferral_payments', 'total_payments',
# 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
# 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
# 'long_term_incentive', 'restricted_stock', 'director_fees']
# (all units are in US dollars)
#
# email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
# 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
# (units are generally number of emails messages; notable exception is
# 'email_address', which is a text string)

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'loan_advances', 'expenses', 'exercised_stock_options',
                 'deferred_income', 'other']


##
## Load the dictionary containing the dataset
##

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#
# "data_dict" structure at this point:
#
# { 'PERSON0': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   'PERSON1': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   ...
#   'PRESONn': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...} }
#

##
## Task 2: Remove outliers: the row containing the "TOTAL" value
##

data_dict.pop("TOTAL", None)


##
## Task 3: Create new feature(s) (also add them to features_list!)
##

for person in data_dict:
    from_messages = float(data_dict[person]['from_messages'])
    to_messages = float(data_dict[person]['to_messages'])
    if math.isnan(from_messages) or from_messages is None or from_messages == 0.0:
        # in dubio pro reo
        data_dict[person]['rate_poi_to_this_person'] = 0.0
    else:
        data_dict[person]['rate_poi_to_this_person'] = float(data_dict[person]['from_poi_to_this_person']) / from_messages
    if math.isnan(to_messages) or to_messages is None or to_messages == 0.0:
        # in dubio pro reo
        data_dict[person]['rate_this_person_to_poi'] = 0.0
        data_dict[person]['rate_shared_receipt_with_poi'] = 0.0
    else:
        data_dict[person]['rate_this_person_to_poi'] = float(data_dict[person]['from_this_person_to_poi']) / to_messages
        data_dict[person]['rate_shared_receipt_with_poi'] = float(data_dict[person]['shared_receipt_with_poi']) / to_messages
features_list.extend(['rate_poi_to_this_person', 'rate_this_person_to_poi', 'rate_shared_receipt_with_poi'])


##
## Task 3a: Explore all features (visualize them)
##

facetting_conditions = {
        "POIs": (lambda x: x['poi']),
        "non-POIs": (lambda x: not x['poi'])
    }
facetting_colors = ['red', 'green']
univariate_feature_list = features_list + ['long_term_incentive', 'restricted_stock', 'restricted_stock_deferred',
                                           'total_stock_value', 'total_payments']
gui.prepare_univariate_analysis(data_dict, univariate_feature_list , facetting_conditions)

bivariate_feature_list = [('salary','bonus'), ('total_payments','salary'), ('total_payments','bonus'),
                          ('total_payments','deferral_payments'), ('deferral_payments','loan_advances'),
                          ('total_payments','exercised_stock_options'), ('total_payments','deferred_income'),
                          ('total_payments','other'), ('rate_poi_to_this_person','rate_this_person_to_poi')]
gui.prepare_bivariate_analysis(data_dict, bivariate_feature_list, facetting_conditions, facetting_colors)


##
## Store data set and extract features / labels
##

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

#
# with feature_list = [ 'featureX0', 'featureX1', ..., 'featureXk' ]
# "data" / "labels" / "features" structure at this point:
#
#  data[i][j]
#       featureXi --->
#        0     1     2             k
#     +-----+-----+-----+- ... -+-----+
#  0  |     |     |     |       |     |   PERSONj
#     +-----+-----+-----+- ... -+-----+   |
#  1  |     |     |     |       |     |   |
#     +-----+-----+-----+- ... -+-----+   v
#     |     |     |     |  .    |     |
#       ...   ...   ...     .     ...
#     |     |     |     |    .  |     |
#     +-----+-----+-----+- ... -+-----+
#  n  |     |     |     |       |     |
#     +-----+-----+-----+- ... -+-----+
#
#     \____/ \_____________...________/
#     labels      features
#


##
## Task 4: Try a varity of classifiers (here we also think a lot and go through the course notes again)
## Task 5: Tune your classifier to achieve better than .3 precision and recall
##

# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines.
#
# For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
# http://scikit-learn.org/stable/modules/grid_search.html#grid-search
# Insights:
# - for GridSearchCV...
#   * when doing GridSearchCV, you can specify other evaluation metrics using the "score" parameter
#   * you can run threads in parallel using the "n_jobs" parameter
#   * set error_score=0 to avoid the algorithm to break if the classifier behaves unstable
# - consider LassoCV or LassoLarsCV

# Provided to give you a starting point. Try a variety of classifiers.

clf_id = options["-c"] if "-c" in options.keys() else None

# Fallback / Default
from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

pca = RandomizedPCA(n_components=3)
svm = SVC(kernel="linear")
clf = Pipeline(steps=[('RandomizedPCA',pca),('SVC',svm)])
#clf = AdaBoostClassifier(base_estimator=svm, n_estimators=50, algorithm='SAMME')

# History of other classifiers we tried...
if clf_id == 5:
    clf = SVC(kernel="rbf")
elif clf_id == 4:
    pca = RandomizedPCA(n_components=8)
    tree = DecisionTreeClassifier(min_samples_split=20)
    adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=50, algorithm='SAMME')
    clf = Pipeline(steps=[('RandomizedPCA',pca),('AdaboostDecisionTree',adaboost)])
elif clf_id == 3:
    tree = DecisionTreeClassifier(min_samples_split=20)
    clf = AdaBoostClassifier(base_estimator=tree, n_estimators=50, algorithm='SAMME')
elif clf_id == 2:
    pca = RandomizedPCA(n_components=8)
    gnb = GaussianNB()
    clf = Pipeline(steps=[('RandomizedPCA',pca),('GaussianNB',gnb)])
elif clf_id == 1:
    clf = GaussianNB()

# Use our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation.
#
# For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


##
## Task 4a: Do train/test split, potentially with advanced splitting methods
##

# e.g. StratifiedShuffleSplit(labels, 1000, random_state = 42)

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pprint

train_test_split_folds = 1000
shuffler = StratifiedShuffleSplit(labels, train_test_split_folds, random_state = 42)

f1_sum = precision_sum = accuracy_sum = recall_sum = 0.0
for train_indices, test_indices in shuffler:
    features_train = []
    labels_train = []
    features_test = []
    labels_test = []
    for train_index in train_indices:
        features_train.append(features[train_index])
        labels_train.append(labels[train_index])
    for test_index in test_indices:
        features_test.append(features[test_index])
        labels_test.append(labels[test_index])
    clf.fit(features_train, labels_train)
    labels_pred = clf.predict(features_test)
    f1_sum += f1_score(labels_test, labels_pred)
    precision_sum += precision_score(labels_test, labels_pred)
    recall_sum += recall_score(labels_test, labels_pred)
    accuracy_sum += accuracy_score(labels_test, labels_pred)
print clf
print "F1 score:  ", f1_sum / train_test_split_folds
print "Precision: ", precision_sum / train_test_split_folds
print "Recall:    ", recall_sum / train_test_split_folds
print "Accuracy:  ", accuracy_sum / train_test_split_folds


##
## Task 6: Dump your classifier, dataset, and features_list
##

# ... so anyone can check your results. You do not need to change anything
# below, but make sure that the version of poi_id.py that you submit can
# be run on its own and generates the necessary .pkl files for validating
# your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# Show GUI if parameter "-g" is present
gui.exec_()