#!/usr/bin/python
# coding=utf-8

"""
The "Detect Fraud from the Enron Scandal with Machine Learning"
classifier trainer/tester.

Creates and selects features from the Enron Scandal Dataset, builds a POI
classifier pipeline, trains and tests it with multiple train/test splits,
writes dataset and classifier to pickle files and reports performance of the
classifier.

Usage:
    python poi_id.py [-l <filename>]
                     [-g {univariate_analysis|bivariate_analysis}]
                     [-s {on|off|pca}] [-f {True|False}] [-c <clf_id>]
                     [-x {in|out|off}] [-t <train_test_split_folds>]
                     [-w <filename>] [-n] [-h]

Options:
    -l <filename>
        Write the whole CSV file to file <filename> for inspection
    -g {univariate_analysis|bivariate_analysis}
        Shows univariate or bivariate analysis as Qt window
    -d {True|False}                         default: False
        Go through dataset and reject all features where more than half of
        the records have no data ("NaN") about this feature
    -s {on|off|pca}                         default: pca
        none:   Disables Feature Scaling and Principle Component Analysis
        minmax: Uses Feature Scaling (MinMaxScaler)
        pca:    Uses Principal Component Analysis but no feature scaling
    -f {True|False}                         default: False
        Explicitly enables / disables automatic feature selection (SelectKBest)
    -c <clf_id>                             default: 4
        Explicitly sets one of the following classifiers for training/testing:
        <clf_id>: 0: Gaussian Naive Bayes
                  1: AdaBoosted Decision Tree
                  2: Support Vector Classifier
                  3: Adaboosted RandomForest (warning: slow and useless)
                  4: KNeighbors
                  5: LogisticRegression
                  6: LDA
    -x {in|out|off}                         default: in
        Sets how to handle GridSearchCV for a few algorithms (2, 4, 5, 6)
        out: Finds the best grid parameter for the algorithm and writes it to
             poi_id_grids.csv. Automatically sets -n and disables -w.
        in:  Reads the parameters from poi_id_grids.csv. If file is not found
             falls back to "off".
        off: Uses the default parameter for the algorithms.
    -t <train_test_split_folds>             default: 10
        Explicitly set the number of train-test-splits which will be done
        during StratifiedShuffleSplit for assessing classifier performance.
    -w <filename>
        Write performance as CSV record to end of file <filename>
    -F <filename>
        Write selected features to CSV record to end of file <filename>
        (only if automatic feature selection enabled)
    -n
        Suppress writing of pickle files (*.pkl) of dataset, classifier etc.
    -h
        Show this screen.

Author:
    Benjamin Soellner <post@benkku.com>
    from the "Intro to Machine Learning" Class
    of Udacity's "Data Analyst" Nanodegree
"""

import sys
import math
import pickle
import pprint
import csv

from poi_id_gui import PoiIdGui


# ------------
# STEP 0
# Preparations
# ------------
import getopt

# Default classifier, feature scaling (on/off), feature selection (on/off) and
# number of train-test-split-folds to use if no command line parameters are
# specified. These should be set to the best-performing configuration so that
# the script can be run w/o command line arguments as specified in the
# project rubric.
default_clf_id = 4
default_f_scaling = "pca"
default_f_selection = False
default_f_rejection = True
default_f_rejection_threshold = 0.75 # not modifiable via options
default_train_test_split_folds = 10
default_gridsearchcv = "in"
gridsearchcv_file_name = "poi_id_gridsearchcv.csv"

# This is in case we wish to use / import tools.
sys.path.append("../tools/")

# Get supported options from the command line which we will store into the
# dictionary "options" access throughout the script
optlist, _ = getopt.getopt(sys.argv[1:], 'r:l:g:s:f:c:t:w:F:nh?x:')
options = {o[0]: o[1] for o in optlist}

# Set up the graphical user interface and supply the GUI mode to it
# perhaps we don't wish to use the GUI at all (if the "-g" option is abscent)
# but PoiIdGui should handle this case too, in this case, just specify None
gui = PoiIdGui(options["-g"] if "-g" in options else None)

# We use this variable to store all our machine learning classifiers in
# during training/testing they will be chained together
pipe = []
features_list = []

# Show help and exit if "-h" present
if "-h" in options.keys() or "-?" in options.keys():
    print __doc__
    exit(0)

# Helper function to load JSON as byte
# from http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )
def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
            return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

# Helper function to calculate subtotal of a few features in a row
def calc_total_value(data_dict_row, features):
    total_value = 'NaN'
    for feature in features:
        value = data_dict_row[feature]
        if type(value) is float or type(value) is int:
            if total_value == 'NaN':
                total_value = 0.0
            total_value += value
    return total_value

# ------------
# STEP 1
# Load the features into a data_dict (dictionary)
# Also: print data set, if asked
# ------------
import pandas

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

if "-l" in options:
    write_filename = options["-l"]
    with open(write_filename, "wb") as write_file:
        header = ["PERSON"]
        fields = data_dict[data_dict.keys()[0]].keys()
        header.extend(fields)
        writer = csv.DictWriter(write_file, delimiter=",", fieldnames=header)
        nanalysis = pandas.DataFrame(
                        columns=["nans","values"], index=fields).fillna(0.0)
        for person in data_dict:
            row = {'PERSON': person}
            row.update(data_dict[person])
            for field in data_dict[person]:
                if data_dict[person][field] == 'NaN':
                    nanalysis["nans"][field] += 1.0
                nanalysis["values"][field] += 1.0
            writer.writerow(row)
        nanalysis["frac"] = nanalysis["nans"]/nanalysis["values"]
        print "I went through the dataset and found this fraction of NaNs"
        pprint.pprint(nanalysis.sort("frac", ascending=False)["frac"])
        print

# "data_dict" structure at this point:
#
# { 'PERSON0': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   'PERSON1': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...},
#   ...
#   'PRESONn': {'feature0': ..., 'feature1': ..., ..., 'featureK': ...} }


# ------------
# STEP 2 (Task 2)
# Remove outliers
# ------------

# Skimming the dataset, there are two records which are outliers
data_dict.pop("TOTAL", None)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)

# ------------
# STEP 3 (Task 3)
# Create new feature(s) (also add them to features_list!)
# Also: re-calculate some totals and see if they match
# ------------

# The absolute values from_poi_to_this_person / from_poi_to_this_person /
# shared_receipt_with_poi is not helpful (some people write / get send more
# or less emails). We should probably find the ratio (how "tightly" they were
# in contact with a POI) instead. We will add these variables as "rate_*"
# features.
for person in data_dict:
    from_messages = float(data_dict[person]['from_messages'])
    to_messages = float(data_dict[person]['to_messages'])
    # Messages received by the user
    if math.isnan(from_messages) or from_messages is None or \
                    from_messages == 0.0:
        # If person received no messages, he didn't receive any from POIs either
        data_dict[person]['rate_poi_to_this_person'] = 0.0
    else:
        data_dict[person]['rate_poi_to_this_person'] = \
            float(data_dict[person]['from_poi_to_this_person']) / from_messages
    # Messages sent by the user
    if math.isnan(to_messages) or to_messages is None or to_messages == 0.0:
        # If person sent no messages, he didn't send any to POIs either
        data_dict[person]['rate_this_person_to_poi'] = 0.0
        data_dict[person]['rate_shared_receipt_with_poi'] = 0.0
    else:
        data_dict[person]['rate_this_person_to_poi'] = \
            float(data_dict[person]['from_this_person_to_poi']) / to_messages
        data_dict[person]['rate_shared_receipt_with_poi'] = \
            float(data_dict[person]['shared_receipt_with_poi']) / to_messages

# Add these three features to the feature_list
new_features_list = ['rate_poi_to_this_person', 'rate_this_person_to_poi',
                      'rate_shared_receipt_with_poi']

# Correct the total_* values of all the persons
total_payments_features = ['salary', 'bonus', 'long_term_incentive',
        'deferred_income', 'deferral_payments', 'loan_advances', 'other',
        'expenses', 'director_fees']
total_stock_value_features = ['exercised_stock_options', 'restricted_stock',
                              'restricted_stock_deferred']

# Fix values total_stock_value and total_payments_features
for person in data_dict:
    total_payments = data_dict[person]['total_payments']
    total_payments_calc = calc_total_value(data_dict[person],
                                              total_payments_features)
    total_stock_value = data_dict[person]['total_stock_value']
    total_stock_value_calc = calc_total_value(data_dict[person],
                                              total_stock_value_features)
    # We only will do something if we could calculate a total and there was none
    # before. Mistakes, where the totals don't match are set by one-offs (below)
    if total_payments == 'NaN' and total_payments_calc != total_payments:
        data_dict[person]['total_payments'] = total_payments_calc
    if total_stock_value == 'NaN' and \
                    total_stock_value_calc != total_stock_value:
        data_dict[person]['total_stock_value'] = total_stock_value_calc

# One-Offs, see comments in data_dict.xlsx
data_dict['BELFER ROBERT']['deferral_payments'] *= -1
data_dict['BELFER ROBERT']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['other'] = \
    data_dict['BHATNAGAR SANJAY']['total_payments'] - \
    data_dict['BHATNAGAR SANJAY']['director_fees']
data_dict['BELFER ROBERT']['total_stock_value'] = \
    calc_total_value(data_dict['BELFER ROBERT'], total_stock_value_features)


# ------------
# STEP 4 (Task 1)
# Select what features you'll use.
# Also: Scale the features.
# ------------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA

# The features we found to work best during exploratory analysis.
# They will be used when automatic feature selection is disabled.
# First MUST be poi.
preferred_features_list = ['poi', 'salary', 'bonus', 'deferral_payments',
                           'loan_advances', 'expenses',
                           'exercised_stock_options',
                           'deferred_income', 'other'] + new_features_list

# This list includes _all_ features. This list is used for exploratory
# (univariate) analysis and in case automatic feature selection is enabled.
# Note: missing are: 'to_messages', 'email_address', 'from_messages'
#   as they are obviously not indicative of a persons POI-status.
all_features_list = preferred_features_list + \
                    ['long_term_incentive', 'restricted_stock',
                     'restricted_stock_deferred', 'total_stock_value',
                     'total_payments']

# Was the command called with a specific value set for automatic feature
# selection? Instead use the default value.
try:
    feature_selection = (options["-f"] != "False")
except:
    feature_selection = default_f_selection

# Do we want to throw out all features where more than half of the features
# are ill-defined?
try:
    feature_rejection = (options["-r"] != "False")
except:
    feature_rejection = default_f_rejection

# Reject all the features where more than half of the features have no
# value defined for a record
if feature_rejection:
    fields = data_dict[data_dict.keys()[0]].keys()
    nanalysis = pandas.DataFrame(
                    columns=["nans","values"], index=fields).fillna(0.0)
    for person in data_dict:
        for field in data_dict[person]:
            if data_dict[person][field] == 'NaN':
                nanalysis["nans"][field] += 1.0
            nanalysis["values"][field] += 1.0
    nanalysis["frac"] = nanalysis["nans"]/nanalysis["values"]
    print "I will drop the following features with fraction of NaNs > ", \
            default_f_rejection_threshold
    nanalysis_rejected = \
        nanalysis[nanalysis["frac"]>default_f_rejection_threshold].\
            sort("frac", ascending=False)["frac"]
    pprint.pprint(nanalysis_rejected)
    rejected_fields = list(nanalysis_rejected.index)
    for rejected_field in rejected_fields:
        if rejected_field in all_features_list:
            all_features_list.remove(rejected_field)
        if rejected_field in preferred_features_list:
            preferred_features_list.remove(rejected_field)

# Select the features either explicitly (preferred_features_list) or with
# as SelectKBest feature selection algorithm.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
if feature_selection:
    # source:
    features_list = all_features_list
    kbest = SelectKBest(f_regression, k=len(preferred_features_list)-1)
    pipe.extend([('SelectKBest',kbest)])
else:
    features_list = preferred_features_list

# Was the command called with a specific value set for automatic feature
# scaling? Instead use the default value.
try:
    feature_scaling = options["-s"]
except:
    feature_scaling = default_f_scaling

# If we do use feature scaling, add as min-max-scaler to the pipeline
if feature_scaling == "on":
    scaler = MinMaxScaler()
    pipe.extend([('MinMaxScaler',scaler)])
elif feature_scaling == "pca":
    pca = RandomizedPCA(n_components=(len(preferred_features_list)-1)/2,
                        random_state=815)
    pipe.extend([('RandomizedPCA',pca)])

# ------------
# STEP 5
# Explore all features (visualize them) - this is completely done by the
# PoiIdGui class (in poi_id_gui module)
# ------------

# Highlight POIs and non-POIs in different colors
facetting_conditions = {
        "POIs": (lambda x: x['poi']),
        "non-POIs": (lambda x: not x['poi'])
    }
# What should be those colors?
facetting_colors = ['red', 'green']

# Prepare univariate feature analysis window
# (will only be done if appropriate gui mode was set to gui object).
gui.prepare_univariate_analysis(data_dict, all_features_list,
                                facetting_conditions, facetting_colors)

# In case we do a bivariate analysis, there is a range of feature pairs we
# would like to explore as scatter plot.
bivariate_features_list = \
    [('salary','bonus'), ('total_payments','salary'),
     ('total_payments','bonus'), ('total_payments','deferral_payments'),
     ('deferral_payments','loan_advances'),
     ('total_payments','exercised_stock_options'),
     ('total_payments','deferred_income'), ('total_payments','other'),
     ('rate_poi_to_this_person','rate_this_person_to_poi')]

# Prepare bivariatefeature analysis window
# (will only be done if appropriate gui mode was set to gui object).
gui.prepare_bivariate_analysis(data_dict, bivariate_features_list,
                               facetting_conditions, facetting_colors)


# ------------
# STEP 6
# Store data set and separate features / labels
# ------------
from feature_format import featureFormat, targetFeatureSplit

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# with features_list = [ 'featureX0', 'featureX1', ..., 'featureXk' ]
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


# ------------
# STEP 7 (Task 4 & 5)
# Try a varity of classifiers, tune classifier to achieve better than .3
# precision and recall.
# ------------
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

# Did the caller specify as specific classifier they want to use? Otherwise
# use default classifier.
try:
    clf_id = int(options["-c"])
except:
    clf_id = default_clf_id

# Did the caller specify how to handle cross validation?
try:
    gridsearchcv = options["-x"]
except:
    gridsearchcv = default_gridsearchcv

# For some algorithms, we do GridSearchCV, for those, the variables
# grid and default_grid are filled
search_grid = {}
run_grid = {}
algorithm = None

# Which classifier did we choose?
if clf_id == 6:
    algorithm = LDA()
elif clf_id == 5:
    # Logistic Regression
    algorithm = LogisticRegression()
    search_grid = {'C': [1e3, 1e4, 1e5]}
    run_grid = {'C': [1000.0]}
elif clf_id == 4:
    # PCA + k Neighbors Classifier
    algorithm = KNeighborsClassifier(n_neighbors=1,weights='uniform',p=1)
    search_grid = {'algorithm': ['ball_tree', 'kd_tree', 'auto'],
                   'weights': ['uniform','distance'],
                   'n_neighbors': [1,2,3]}
    run_grid = {'n_neighbors': [3], 'weights': ['uniform'],
                'algorithm': ['ball_tree']}
elif clf_id == 3:
    # Adaboost Random Forest
    forest = RandomForestClassifier(n_estimators=4, min_samples_split=20,
                                       n_jobs=4, random_state=42)
    algorithm = AdaBoostClassifier(base_estimator=forest, n_estimators=50,
                                  algorithm='SAMME', random_state=23)
elif clf_id == 2:
    # SVC with grid search for parameters and PCA
    search_grid = {'C': [1e3, 1e4, 1e5],
                   'gamma': [0.0001, 0.001, 0.01, 0.1],
                   'kernel': ['rbf', 'poly']}
    run_grid = {'kernel': ['rbf'], 'C': [1000.0], 'gamma': [0.0001]}
    algorithm = SVC(class_weight='auto')
    if feature_scaling != "on":
        raise Exception("Error: Support Vector Classifier must use " +
                "feature_scaling.")
elif clf_id == 1:
    # Decision Tree Classifier + AdaBoost + PCA
    tree = DecisionTreeClassifier(min_samples_split=20)
    algorithm = AdaBoostClassifier(base_estimator=tree, n_estimators=50,
                                  algorithm='SAMME', random_state=23)
elif clf_id == 0:
    # Gaussian Naive Bayes: starting point and minimum viable product
    algorithm = GaussianNB()

# How should we handle grid search or should we just run with the
# standard parameters?
if gridsearchcv == "in":
    # The optimal cross validation values should be found from the csv file
    try:
        with open(gridsearchcv_file_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            for record in reader:
                if feature_scaling == str(record[0]) and \
                                feature_selection == (record[1]=="True") and \
                                clf_id == int(record[2]):
                    run_grid = json_loads_byteified(record[3])
                    print "Fetched best GridSearchCV values: ", run_grid
    except:
        print "Could not read grid filename: ", gridsearchcv_file_name
        print "Defaulting to parameter standard settings"
    # Check if we do have now parameters for our algorithm and set them
    if len(run_grid) > 0:
        algorithm.set_params(**run_grid)
    # Just use the standard algorithm
    wrapped_algorithm = algorithm
elif gridsearchcv == "out":
    # The optimal grid search values should be found and written (later)
    # to the CSV file
    wrapped_algorithm = GridSearchCV(algorithm, search_grid,
                                     scoring='precision')

pipe.extend([('Algorithm',wrapped_algorithm)])

# Put all the pipeline steps into an pipeline object and assign to "clf" for
# further processing and later on, serialization
clf = Pipeline(steps=pipe)

# ------------
# STEP 8
# Do train/test split with Stratified Shuffle Split
# ------------
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score
import time

# Did the user specify a specific number of splitting folds they want to do?
# Otherwise, use default number
try:
    train_test_split_folds = int(options["-t"])
except:
    train_test_split_folds = default_train_test_split_folds

# We are using Stratified Shuffle Split, the same cross-validation method as
# used in the tester
shuffler = StratifiedShuffleSplit(labels, train_test_split_folds,
                                  random_state = 42)

# Run through all the train-test splits, train, test and measure key metrics
if gridsearchcv == "out":
    if len(search_grid)>0:
        # Specify CV option
        wrapped_algorithm.cv = shuffler
        # Fit classifier
        clf.fit(features, labels)
        # Get the best parameters of cross validation
        params = wrapped_algorithm.best_params_
        # Convert them to a JSON object with every parameter in a
        # one-element array
        grid_json = json.dumps(params, ensure_ascii=True)
        # Write them to the csv file
        with open(gridsearchcv_file_name, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([feature_scaling, feature_selection,
                             clf_id, grid_json])
        # Print clf pipeline on the screen so we see what the script is using.
        pprint.pprint(clf)
        # Print parameters we found
        print "We found these best params: ", params
    else:
        print "Algorithm does not support cross-validation analysis; " + \
                "try another '-c <clf_id>' or running without '-x out'."
else:
    labels_pred = []
    labels_truth = []
    starttime = time.time()
    for train_indices, test_indices in shuffler:
        features_train = []
        labels_train = []
        features_test = []
        labels_test = []
        # build train and test set
        for train_index in train_indices:
            features_train.append(features[train_index])
            labels_train.append(labels[train_index])
        for test_index in test_indices:
            features_test.append(features[test_index])
            labels_test.append(labels[test_index])
        # fit classifier (train)
        clf.fit(features_train, labels_train)
        # predict
        labels_pred.extend(clf.predict(features_test))
        labels_truth.extend(labels_test)
    # calculate average runtime and average performance metrics
    runtime = (time.time() - starttime) / train_test_split_folds
    f1 = f1_score(labels_truth, labels_pred)
    precision = precision_score(labels_truth, labels_pred)
    recall = recall_score(labels_truth, labels_pred)
    accuracy = accuracy_score(labels_truth, labels_pred)
    # Print clf pipeline on the screen so we see what the script is using.
    pprint.pprint(clf)
    # Print metrics and runtime on screen
    print "Predictions: ", len(labels_pred)
    print "F1 score:    ", f1
    print "Precision:   ", precision
    print "Recall:      ", recall
    print "Accuracy:    ", accuracy
    print "Runtime:     ", runtime

# ------------
# STEP 9 (Task 6)
# Dump your classifier, dataset, and features_list
# Also:
# - Write classifier performance to statistics file for comparison of
#   different classifiers
# - Write selected features to statistics file
# - Write GridSearchCV stuff to a file
# - Show the exploratory GUI at the end of the step
# ------------
from tester import dump_classifier_and_data
from collections import Counter

# Write this classifiers features to another CSV file (only if automatic
# feature selection is turned on
if "-F" in options and feature_selection:
    filename = options["-F"]
    # Append Configuration
    head = [feature_scaling, clf_id]
    # For getting the feature scores...
    kbest = clf.named_steps['SelectKBest']
    scores_dict = {feature: score
                   for feature, score in zip(features_list[1:], kbest.scores_)}
    scores_top = (Counter(scores_dict).most_common())[0:kbest.k]
    # Write file
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Write every top score
        for score_top in scores_top:
            writer.writerow(head + [score_top[0], score_top[1]])

# Write this classifiers configuration and performance as as single line to a
# CSV file for later classifier evaluation.
# (Only if filename is specified with command line option "-w"!)
if gridsearchcv != "out" and "-w" in options:
    filename = options["-w"]
    row = [
            # Configuration
            feature_scaling, feature_selection, clf_id,
            # Performance Metrics
            f1, precision, recall, accuracy, runtime
        ]
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(row)

# Dump classifier for using alongside with tester.
if gridsearchcv != "out" and "-n" not in options:
    print "Dumping classifier and data."
    dump_classifier_and_data(clf, my_dataset, features_list)
else:
    print "Dumping classifier and data skipped."

# Show GUI (if parameter "-g" is present, gui object will take care about that)
gui.exec_()