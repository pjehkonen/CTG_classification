from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from ctg_lib.ctg_time import now_time_string


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import os


def plot_roc(fpr, tpr, classifier, logger=None,  myEnv=None):
    time_now = now_time_string()
    if myEnv is None:
        logger.info("Displaying figure at IDE")
    else:
        logger.info("Generating a figure at {} for {}".format(time_now, classifier))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {}".format(auc(fpr,tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} Regression Curve".format(classifier))
    if myEnv is None:
        plt.show();
    else:
        plt.savefig(Path(myEnv.out_dir,time_now+"-"+classifier+".png"))


def CTG_KNN(X_train, X_test, y_train, y_test, logger, classifier, myEnv):

    param_grid = {'n_neighbors': np.arange(1,9)}

    nn_neighbors = 5
    os.environ['OMP_NUM_THREADS'] = '16'
    nn_jobs = -1
    knn = KNeighborsClassifier(n_jobs=nn_jobs)

    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(X_train, y_train)

    print(knn_cv.best_params_)

    # use best parameters
    optimal = KNeighborsClassifier(**knn_cv.best_params_,n_jobs=nn_jobs)
    optimal.fit(X_train, y_train)


    y_pred = optimal.predict(X_test)

    y_pred_prob = optimal.predict_proba(X_test)[:,1]
    fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

    plot_roc(fpr, tpr, logger, classifier, myEnv)

    logger.info("This is log file for classification algorithm of {}".format(classifier))
    print("Accuracy is ",optimal.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    print("Area under ROC")
    print(roc_auc_score(y_test, y_pred_prob))