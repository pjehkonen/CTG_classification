from sklearn.neighbors import KNeighborsClassifier
import os

def CTG_KNN(X_train, y_train, X_test, y_test, nn_neigbors):
    os.environ['OMP_NUM_THREADS'] = '4'
    nn_jobs = -1
    knn = KNeighborsClassifier(n_neigbors=nn_neigbors, n_jobs=nn_jobs)
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    print("Accuracy is ",knn.score(X_test, y_test))
    