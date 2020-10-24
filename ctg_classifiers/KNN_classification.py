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


def plot_roc(fpr, tpr, classifier, logger=None, myEnv=None):
    time_now = now_time_string()
    if myEnv is None:
        logger.info("Displaying figure at IDE")
    else:
        logger.info("Generating a figure at {} for {}".format(time_now, classifier))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {}".format(auc(fpr, tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} Regression Curve".format(classifier))
    if myEnv is None:
        plt.show();
    else:
        plt.savefig(Path(myEnv.out_dir, time_now + "-" + classifier + ".png"))


def log_results(logger, optimal, X_test, y_test, y_pred, y_pred_prob):
    logger.info("Accuracy is {}".format(optimal.score(X_test, y_test)))
    logger.info("Confusion matrix")
    logger.info("{}".format(confusion_matrix(y_test, y_pred)))
    logger.info("Classification report")
    logger.info("{}".format(classification_report(y_test, y_pred)))
    logger.info("Area under ROC")
    logger.info("{}".format(roc_auc_score(y_test, y_pred_prob)))


def CTG_KNN(X_train, X_test, y_train, y_test, logger, classifier, myEnv):
    param_grid = {'n_neighbors': np.arange(1, 9)}

    nn_neighbors = 5
    num_threads = '16'
    N_jobs = -1
    os.environ['OMP_NUM_THREADS'] = num_threads
    nn_jobs = N_jobs
    N_cv = 5

    knn = KNeighborsClassifier(n_jobs=nn_jobs)

    logger.info("Setting following parameters")
    logger.info("OMP_NUM_THREADS = {}".format(num_threads))
    logger.info("nn_jobs = {}".format(N_jobs))
    logger.info("cv = {}".format(N_cv))

    knn_cv = GridSearchCV(knn, param_grid, cv=5)

    logger.info("Using Grid search CV with parameters")
    logger.info("{}".format(param_grid))

    logger.info("Starting classifier grid search fit")
    knn_cv.fit(X_train, y_train)

    print(knn_cv.best_params_)
    logger.info("Found best parameters as")
    logger.info("{}".format(knn_cv.best_params_))

    # use best parameters
    logger.info("Creating optimal classifer with best parameters")
    optimal = KNeighborsClassifier(**knn_cv.best_params_, n_jobs=nn_jobs)
    optimal.fit(X_train, y_train)

    logger.info("Calculating y_pred")
    y_pred = optimal.predict(X_test)

    logger.info("Calculating y_pred_prob")
    y_pred_prob = optimal.predict_proba(X_test)[:, 1]

    logger.info("Generating roc_curve with y_test, y_pred_prob")
    fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

    logger.info("Calling figure generation with this classifier")
    plot_roc(fpr, tpr, logger, classifier, myEnv)
    logger.info("Figure generated")

    # Printing stuff
    print("Accuracy is ", optimal.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    print("Area under ROC")
    print(roc_auc_score(y_test, y_pred_prob))

    log_results(logger, optimal, X_test, y_test, y_pred, y_pred_prob)
