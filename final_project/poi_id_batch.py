#!/usr/bin/python
# coding=utf-8

"""
The "Detect Fraud from the Enron Scandal with Machine Learning"
classifier batch runner.

Runs multiple takes of "python poi_id.py" with varying command line arguments,
causing it to run in different configurations (with differently configured
machine learning pipelines) in order to compare the performance of those.
Edit poi_id_batch.py's configuration variables to modify how
"python poi_id.py" is called. Also implements a timeout after which execution
of each pipeline would be stopped.

Usage:
    python poi_id_batch.py [-h]

Options:
    -h: Show this help page.
    -x: Switch to GridSearchCV learning mode and retrieve the best
        GridSearchCV values for all algorithms supporting it

Author:
    Benjamin Soellner <post@benkku.com>
    from the "Intro to Machine Learning" Class
    of Udacity's "Data Analyst" Nanodegree
"""
import multiprocessing
import subprocess
import getopt
import sys

# Modify these variables to control which configurations of poi_id.py are run
f_scaling_options = ['on','off','pca'] # Explicitly set feature scalingon/off
f_selection_options = [True,False] # Explicitly set feature selection on/off
classifier_options = range(0,7) # Use one of the 7 defined classifiers (0..6)
train_test_split_folds = 1000 # How many train / test split folds?
train_test_split_folds_x = 1000 # How many during cross validation training?
output_filename = 'poi_id_batch.csv' # Where to write performance report?
featurescores_filename = 'poi_id_featurescores.csv' # Where to write feature
                                                    # scores
timeout = 60*45 # Seconds until execution of any classifier run is stopped
timeout_x = 60*45 # How long during cross-validation training?


def run(command):
    subprocess.call(command)



if __name__ == '__main__':
    # Get supported options from the command line which we will store into the
    # dictionary "options" access throughout the script
    optlist, _ = getopt.getopt(sys.argv[1:], 'h?x')
    options = {o[0]: o[1] for o in optlist}
    gridsearchcv_learn = ("-x" in options)

    # Show help page and exit if "-h" option set
    if "-h" in options.keys() or "-?" in options.keys():
        print __doc__
        exit(0)

    # Now, run through all the options and call "poi_id.py".
    print "======"
    for classifier in classifier_options:
        for f_scaling in f_scaling_options:
            for f_selection in f_selection_options:
                command = ['python', 'poi_id.py',
                           '-s', str(f_scaling),
                           '-f', str(f_selection),
                           '-c', str(classifier),
                           '-r', 'True',
                           '-n']
                if gridsearchcv_learn:
                    command += ['-x', 'out',
                                '-t', str(train_test_split_folds_x)]
                else:
                    command += ['-x', 'in',
                                '-w', output_filename,
                                '-F', featurescores_filename,
                                '-t', str(train_test_split_folds)]
                # Will be started as a subprocess to ensure we can watching it
                # and abort after timeout
                p = multiprocessing.Process(target=run, name="Run",
                                            args=(command,))
                print "Calling ... ", " ".join(command)
                print "------"
                p.start()
                t = timeout_x if gridsearchcv_learn else timeout
                p.join(t)
                if p.is_alive():
                    print "Still running after timeout of " + \
                            str(t) + " seconds. Aborting."
                    p.terminate()
                    p.join()
                print "======"
