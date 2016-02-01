#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## TODO Task 1: Select what features you'll use.

# Available to us:
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
features_list = ['poi', 'salary', 'bonus', '']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## TODO Task 1a: Explore features (visualize them)

#
# "data_dict" structure at this point:
#
# { 'PERSON0': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   'PERSON1': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   ...
#   'PRESONn': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...} }
#

## TODO Task 2: Remove outliers
## TODO Task 3: Create new feature(s) (also add them to features_list!)

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

## TODO Task 4: Try a varity of classifiers (here we also think a lot and go through the course notes again)

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
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

## TODO Task 5: Tune your classifier to achieve better than .3 precision and recall
# ... using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation.
#
# For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## TODO Task 4a: Do train/test split, potentially with advanced splitting methods

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## TODO Task 6: Dump your classifier, dataset, and features_list
# ... so anyone can check your results. You do not need to change anything
# below, but make sure that the version of poi_id.py that you submit can
# be run on its own and generates the necessary .pkl files for validating
# your results.

dump_classifier_and_data(clf, my_dataset, features_list)
