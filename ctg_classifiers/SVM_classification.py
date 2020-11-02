from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ctg_lib.ctg_path_env import in_triton

import matplotlib
if in_triton():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path
import os

from ctg_lib.ctg_path_env import in_triton

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


def log_svc_parameters(logger, my_C_params, kernels, my_degrees, my_cache_size, my_max_iter, my_class_weight):
    logger.info("Setting following parameters")
    logger.info("C = {}".format(my_C_params))
    logger.info("Kenels = {}".format(kernels))
    logger.info("Polynom degrees = {}".format(my_degrees))
    logger.info("Cache size = {}".format(my_cache_size))
    logger.info("Maximum iterations = {}".format(my_max_iter))
    logger.info("Class weights used = {}".format(my_class_weight))


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


def make_grid_cv_svc(pipeline, parameters, my_scoring, nn_jobs, N_cv, logger):
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


def CTG_SVC(X_train, X_test, y_train, y_test, logger, classifier, myEnv, start_time):


    # Setting SVC parameters for both hyperparameter search and criteria
    C = 1
    my_C_params = np.linspace(0.75, C,4)
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    my_degrees = [3, 4, 5]
    my_cache_size = 2000
    my_class_weight = 'balanced'
    my_max_iter = 1000000

    # For Triton, break the wall
    num_threads = '32'
    os.environ['OMP_NUM_THREADS'] = num_threads


    # Pipeline settings
    N_jobs = -1 # use all cores
    nn_jobs = N_jobs
    N_cv = 7  # Number of cross validations in Grid Search
    my_scoring = 'roc_auc'  # Metric from list of "roc_auc, accuracy, neg_log_loss, jaccard, f1"
    my_kernel = 'rbf'

    # Make pipeline with steps
    steps = [('scaler', StandardScaler()),
             ('svc', SVC(kernel=my_kernel, probability=True, cache_size=my_cache_size, max_iter=my_max_iter, class_weight='balanced'))]
    pipeline = Pipeline(steps)

    # Define SVC_related grid search parameters
    parameters = {'svc__C': my_C_params,
                  'svc__degree': my_degrees
                  }

    print("Parameters set for environment and classifier")
    log_svc_parameters(logger, my_C_params, my_kernel, my_degrees, my_cache_size, my_max_iter, my_class_weight)

    # Make grid cv object
    my_cv = make_grid_cv_svc(pipeline, parameters, my_scoring, nn_jobs, N_cv, logger)

    logger.info("Starting classifier search fit")
    print("Starting Cross Validation")

    #my_cv = make_pipeline(StandardScaler(), SVC(probability=True))
    # Actual pipeline fit takes place here
    my_cv.fit(X_train, y_train)


    y_pred = my_cv.predict(X_test)
    y_pred_prob = my_cv.predict_proba(X_test)[:, 1]

    logger.info("Generating roc_curve with y_test, y_pred_prob")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plot_roc(fpr, tpr, classifier, logger, myEnv, start_time)

    # Printing stuff
    print_stuff(classifier, my_cv, X_test, y_test, y_pred, y_pred_prob)

    # Write same stuff to log
    log_results(logger, my_cv, X_test, y_test, y_pred, y_pred_prob)
