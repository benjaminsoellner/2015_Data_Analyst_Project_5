def classify(features_train, labels_train):   
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    return clf.fit(features_train, labels_train)
    
