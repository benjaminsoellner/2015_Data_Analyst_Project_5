#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of observations: ", len(enron_data)
for person in enron_data:
    print person, len(enron_data[person])
print "People who are POI:", \
    sum(1 if enron_data[person]["poi"] else 0 for person in enron_data)
print "Stock value of James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Emails from Wesley Collwell:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Stock options exercised by Jeffrey Skilling:", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "Money Skilling/Lay/Fastow", \
    [enron_data[person]["total_payments"] for person in ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"]]
print "People with a quantified salary:", \
    sum(1 if enron_data[person]["salary"] != 'NaN' else 0 for person in enron_data)
print "People with an email address:", \
    sum(1 if enron_data[person]["email_address"] != 'NaN' else 0 for person in enron_data)
n = sum(1 if enron_data[person]["total_payments"] == 'NaN' else 0 for person in enron_data)
print "Number of people with NaN as total payments", n, " - fraction: ", n/float(len(enron_data))
n = sum(1 if enron_data[person]["total_payments"] == 'NaN' and enron_data[person]["poi"] else 0 for person in enron_data)
print "Number of POIs with NaN as total payments", n, " - fraction: ", n/float(len(enron_data))