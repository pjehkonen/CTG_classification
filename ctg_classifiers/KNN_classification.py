from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

import os


def plot_roc(fpr, tpr, classifier, logger=None, my_env=None, start_time=None):
    if my_env is None:
        logger.info("Displaying figure at IDE")
    else:
        logger.info("Generating a figure at {} for {}".format(start_time, classifier))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {}".format(auc(fpr, tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} Regression Curve".format(classifier))
    if my_env is None:
        plt.show()
    else:
        plt.savefig(Path(Path(my_env.log_dir, start_time), classifier + ".png"))


def log_results(logger, cv, X_test, y_test, y_pred, y_pred_prob):

    logger.info("Accuracy is {}".format(cv.score(X_test, y_test)))
    logger.info("Confusion matrix")
    logger.info("\n{}".format(confusion_matrix(y_test, y_pred)))
    logger.info("Classification report")
    logger.info("\n{}".format(classification_report(y_test, y_pred)))
    logger.info("Area under ROC {}".format(roc_auc_score(y_test, y_pred_prob)))
    logger.info("The cv_results_")
    for key in cv.cv_results_.keys():
        logger.info("key: {}".format(key))
        logger.info("values\n{}".format(cv.cv_results_[key]))

def play_with_results(cv_object):
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    cv_results_df = pd.DataFrame(cv_object.cv_results_)
    cols = cv_results_df.columns
    time_fields = [element for element in cols if 'time' in element]
    param_fields = [element for element in cols if 'param' in element]
    test_fields = [element for element in cols if 'test' in element]
    train_fields = [element for element in cols if 'train' in element]


def CTG_KNN(X_train, X_test, y_train, y_test, logger, classifier, myEnv, start_time):


    num_neighbors = np.arange(1, 15)
    metrics = ["euclidean", "manhattan", "chebyshev"]

    #parameter_grid = {'n_neighbors': num_neighbors, 'metric': metrics}
    my_scoring = 'roc_auc' # "accuracy, neg_log_loss, jaccard, f1"

    num_threads = '16'
    N_jobs = -1
    os.environ['OMP_NUM_THREADS'] = num_threads
    nn_jobs = N_jobs
    N_cv = 5

    #knn = KNeighborsClassifier(n_jobs=nn_jobs)
    steps = [('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier())]
    pipeline = Pipeline(steps)

    parameters = {'knn__n_neighbors': num_neighbors,
                  'knn__metric': metrics
                  }

    print("Parameters set for environment and classifier")
    logger.info("Setting following parameters")
    logger.info("OMP_NUM_THREADS = {}".format(num_threads))
    logger.info("nn_jobs = {}".format(N_jobs))
    logger.info("cv = {}".format(N_cv))

    knn_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=my_scoring,
        n_jobs=nn_jobs,
        cv=N_cv,
        refit=True,
        return_train_score=True
    )

    logger.info("Using Grid search CV with parameters")
    logger.info("{}".format(parameters))

    logger.info("Starting classifier grid search fit")
    print("Starting Grid Search Cross validation")
    knn_cv.fit(X_train, y_train)

    print("Found best parameters for {} which are {}".format(classifier, knn_cv.best_params_))
    logger.info("Found best parameters {}".format(knn_cv.best_params_))

    play_with_results(knn_cv)

    # use best parameters    print("Found best parameters")
    print("{}".format(knn_cv.best_params_))
    logger.info("Creating optimal classifier with best parameters")
    print("Creating optimal filter")
    #optimal = KNeighborsClassifier(**knn_cv.best_params_, n_jobs=nn_jobs)
    #optimal.fit(X_train, y_train)

    logger.info("Calculating y_pred")
    y_pred = knn_cv.predict(X_test)

    logger.info("Calculating y_pred_prob")
    y_pred_prob = knn_cv.predict_proba(X_test)[:, 1]

    logger.info("Generating roc_curve with y_test, y_pred_prob")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    logger.info("Calling figure generation with this classifier")
    print("Creating ROC curve")
    plot_roc(fpr, tpr, classifier, logger, myEnv, start_time)
    logger.info("Figure generated")

    # Printing stuff
    print("Accuracy is ", knn_cv.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    print("Area under ROC")
    print(roc_auc_score(y_test, y_pred_prob))
    print("Found best parameters")
    print("{}".format(knn_cv.best_params_))


    # Write same stuff to log
    log_results(logger, knn_cv, X_test, y_test, y_pred, y_pred_prob)
