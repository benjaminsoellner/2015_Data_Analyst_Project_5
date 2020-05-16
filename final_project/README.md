The Code of the "Classify POIs from the Enron Scandal" Final Project
====================================================================

This folder contains my take on the final project of Udacity's Machine Learning
class.

**Note:** This is just an overview of contents of this folder. Refer to the
project report, especially section [System Design][1] in order to get a deeper
understanding of what I am talking about.

Code files:

* ```poi_id.py```: generates classification pipeline, test, write performance
data and output (run with ```-h``` option for more info)
* ```feature_format.py```: helper file to handle dataset preprocessing
* ```poi_id_gui.py```: helper file to display data for explorative analysis
* ```poi_id_batch.py```: wrapper script to run multiple ```poi_id.py```s in
a row

Ressource files:

* ```final_project_dataset.pkl```, ```final_project_dataset_modified.pkl```:
files supplied for our analysis
* ```my_classifier.pkl```, ```my_dataset.pkl```, ```my_feature_list.pkl```:
dumped classifier and data for later evaluation

Ressource files used for project report:

* ```*.vsd```, ```*.png```: Image files
* ```poi_id_batch.csv```, ```poi_id_featurescores.csv```: Output of
```poi_id_batch.py``` containing pipeline performance information
* ```poi_id_clf_ids.csv```: static information about the classifiers used

[1]: https://benjaminsoellner.github.io/DAND_5_MachineLearningEnronData/Data_Analyst_Project_5_-_Classify_POIs_with_Machine_Learning.html#System-Design

by Benjamin Söllner - http://www.benkku.com´