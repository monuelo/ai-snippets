def classify(features_train, labels_train, kernel_train, c_train=1.0, gamma_train='auto'):   
    
    ### import the sklearn module for SVM
    from sklearn.svm import SVC
    
    ### create classifier
    clf = SVC(kernel=kernel_train, gamma=gamma_train, C=float(c_train))

    ### fit the classifier on the training features and labels
    ### return the fit classifier
    return clf.fit(features_train, labels_train)
    