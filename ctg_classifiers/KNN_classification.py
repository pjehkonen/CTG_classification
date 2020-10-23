from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os

def CTG_KNN(X_train, X_test, y_train, y_test):
    nn_neighbors = 5
    os.environ['OMP_NUM_THREADS'] = '16'
    nn_jobs = -1
    knn = KNeighborsClassifier(n_neighbors=nn_neighbors, n_jobs=nn_jobs)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy is ",knn.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    