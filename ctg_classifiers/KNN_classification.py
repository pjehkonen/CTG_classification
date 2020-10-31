from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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


def print_stuff(classifier, cv, X_test, y_test, y_pred, y_pred_prob):
    print("**** Created ROC curve plot ****")

    print("Found best parameters for {} which are {}".format(classifier, cv.best_params_))
    print("Found best parameters {}".format(cv.best_params_))
    print("Best score {}".format(cv.best_score_))
    print("Best index {}".format(cv.best_index_))

    print("Accuracy is ", cv.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    print("Area under ROC")
    print(roc_auc_score(y_test, y_pred_prob))
    print("Found best parameters")
    print("{}".format(cv.best_params_))


def log_parameters(logger, max_neighbors, metrics, my_scoring, num_threads, N_jobs, N_cv):
    logger.info("Setting following parameters")
    logger.info("max_neighbors = {}".format(max_neighbors))
    logger.info("metrics = {}".format(metrics))
    logger.info("my_scoring = {}".format(my_scoring))
    logger.info("OMP_NUM_THREADS = {}".format(num_threads))
    logger.info("nn_jobs = {}".format(N_jobs))
    logger.info("cv = {}".format(N_cv))


def log_results(logger, cv, X_test, y_test, y_pred, y_pred_prob):
    logger.info("*********** Logging results  *****************")
    logger.info("Found best parameters {}".format(cv.best_params_))
    logger.info("Best score {}".format(cv.best_score_))
    logger.info("Best index {}".format(cv.best_index_))

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


def make_grid_cv(pipeline, parameters, my_scoring, nn_jobs, N_cv, logger):
    gs_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=my_scoring,
        n_jobs=nn_jobs,
        cv=N_cv,
        refit=True,
        return_train_score=True
    )
    logger.info("Using Grid search CV with parameters")
    logger.info("\n{}".format(parameters))
    logger.info(
        "gs_cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=my_scoring, n_jobs=nn_jobs, cv=N_cv, refit=True, return_train_score=True)")
    return gs_cv


def CTG_KNN(X_train, X_test, y_train, y_test, logger, classifier, myEnv, start_time):
    # Setting parameters for both hyperparameter search and criteria
    max_neighbors = 3
    num_neighbors = np.arange(1, max_neighbors)
    metrics = ["euclidean", "manhattan", "chebyshev"]

    my_scoring = 'roc_auc'  # "accuracy, neg_log_loss, jaccard, f1"

    # For Triton, break the wall
    num_threads = '16'
    os.environ['OMP_NUM_THREADS'] = num_threads

    # For my computer, use all processors
    N_jobs = -1
    nn_jobs = N_jobs
    N_cv = 2

    # knn = KNeighborsClassifier(n_jobs=nn_jobs)
    # Make pipeline with steps
    steps = [('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier())]  # <- change this to SVM and parameters below accordingly
    pipeline = Pipeline(steps)
    # Define KNN_related grid search parameters
    parameters = {'knn__n_neighbors': num_neighbors,
                  'knn__metric': metrics}

    print("Parameters set for environment and classifier")
    log_parameters(logger, max_neighbors, metrics, my_scoring, num_threads, N_jobs, N_cv)

    # Make grid cv object
    my_cv = make_grid_cv(pipeline, parameters, my_scoring, nn_jobs, N_cv, logger)

    logger.info("Starting classifier search fit")
    print("Starting Cross Validation")

    # Actual pipeline fit takes place here
    my_cv.fit(X_train, y_train)

    # play_with_results(gs_cv)

    # use best parameters    print("Found best parameters")
    # optimal = KNeighborsClassifier(**knn_cv.best_params_, n_jobs=nn_jobs)
    # optimal.fit(X_train, y_train)

    y_pred = my_cv.predict(X_test)
    y_pred_prob = my_cv.predict_proba(X_test)[:, 1]

    logger.info("Generating roc_curve with y_test, y_pred_prob")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plot_roc(fpr, tpr, classifier, logger, myEnv, start_time)

    # Printing stuff
    print_stuff(classifier, my_cv, X_test, y_test, y_pred, y_pred_prob)

    # Write same stuff to log
    log_results(logger, my_cv, X_test, y_test, y_pred, y_pred_prob)
