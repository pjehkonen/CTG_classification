from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix

from ctg_lib.ctg_path_env import in_triton

import matplotlib
if in_triton():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path
import os


def plot_matrix(my_cv, X_test, y_test, classifier, my_scoring,  my_env, start_time):
    plt.style.use('default')
    fig = plt.figure(figsize=(8,8), dpi=150)

    my_title = "{} Confusion Matrix ({})".format(classifier, my_scoring)
    class_names = ['normal','zigzag']
    plot_confusion_matrix(my_cv, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    plt.title(my_title)

    plt.savefig(Path(Path(my_env.log_dir, start_time), 'CFM_unnormalized_'+classifier + ".png"))
    print(my_title)
    print(confusion_matrix)

    plot_confusion_matrix(my_cv, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='all')
    plt.title(my_title)

    plt.savefig(Path(Path(my_env.log_dir, start_time), 'CFM_normalized_'+classifier + ".png"))

def plot_roc(fpr, tpr, classifier, my_scoring, training, logger=None, my_env=None, start_time=None):
    if my_env is None:
        logger.info("Displaying figure at IDE")
    else:
        logger.info("Generating a figure at {} for {}".format(start_time, classifier))
    fig = plt.figure(figsize=(10,10), dpi=100)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {:.4f}".format(auc(fpr, tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    if training:
        tset = 'training'
    else:
        tset = 'test'
    plt.title("{} AUC with {} ({})".format(classifier, tset, my_scoring))
    plt.savefig(Path(Path(my_env.log_dir, start_time), 'AUC_'+classifier + ".png"))


def print_stuff(classifier, cv, my_scoring, X_test, y_test, y_pred, y_pred_prob):
    print("**** Created ROC curve plot ****")

    print("Found best parameters for {} which are {}".format(classifier, cv.best_params_))
    print("Found best parameters {}".format(cv.best_params_))
    print("Best score {}".format(cv.best_score_))
    print("Best index {}".format(cv.best_index_))

    print("Metric applied as target: {}".format(my_scoring))

    print("Accuracy is ", cv.score(X_test, y_test))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report")
    print(classification_report(y_test, y_pred))
    print("Area under ROC")
    print(roc_auc_score(y_test, y_pred_prob))
    print("Found best parameters")
    print("{}".format(cv.best_params_))


def log_rf_parameters(logger, my_n_estimators, my_criterion, my_max_depth, my_min_samples_split, my_min_samples_leaf, my_boostrap, my_n_jobs, my_class_weight, my_max_samples):
    logger.info("Setting following parameters")
    logger.info("n_estimators = {}".format(my_n_estimators))
    logger.info("criterion = {}".format(my_criterion))
    logger.info("max_depth = {}".format(my_max_depth))
    logger.info("min_samples_split = {}".format(my_min_samples_split))
    logger.info("min_samples_leaf = {}".format(my_min_samples_leaf))
    logger.info("my_boostrap = {}".format(my_boostrap))
    logger.info("n_jobs = {}".format(my_n_jobs))
    logger.info("class_weight = {}".format(my_class_weight))
    logger.info("max_samples = {}".format(my_max_samples))


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


def CTG_RF(X_train, X_test, y_train, y_test, logger, classifier, myEnv, start_time):

    # Setting Random Forest parameters for both hyperparameter search and criteria
    my_n_estimators = 100
    my_criterion = 'gini' # 'gini' or 'entropy'
    my_criterions = ['gini','entropy']
    my_max_depth = 10
    my_min_samples_split = np.max([int(y_test.sum()/3),2]) # make so that max third of test zigzags are minimum split
    my_min_samples_leaf = 1
    my_boostrap = True
    my_n_jobs = -1
    my_class_weight = 'balanced_subsample'
    my_max_samples = None

    # For Triton, break the wall
    num_threads = '32'
    os.environ['OMP_NUM_THREADS'] = num_threads


    # Pipeline settingsfrom joblib import dump, load
    my_cv = 7  # Number of cross validations in Grid Search
    my_scoring = 'f1'  # Metric from list of "roc_auc, accuracy, neg_log_loss, jaccard, f1"

    training_set = True

    # Make pipeline with steps
    steps = [('scaler', StandardScaler()),
             ('RFC', RandomForestClassifier(n_estimators=my_n_estimators,
                                            min_samples_split=my_min_samples_split,
                                            max_features="auto",
                                            min_samples_leaf=my_min_samples_leaf,
                                            bootstrap=my_boostrap,
                                            n_jobs=my_n_jobs,
                                            verbose=2,
                                            class_weight=my_class_weight,
                                            max_samples=my_max_samples))]
    pipeline = Pipeline(steps)

    # Define SVC_related grid search parameters
    parameters = {'RFC__max_depth': np.arange(1, my_max_depth),
                  'RFC__criterion': my_criterions
                  }

    print("Parameters set for environment and classifier")
    log_rf_parameters(logger, my_n_estimators, my_criterion, my_max_depth, my_min_samples_split, my_min_samples_leaf, my_boostrap, my_n_jobs, my_class_weight, my_max_samples)

    # Make grid cv object
    my_cv = make_grid_cv_svc(pipeline, parameters, my_scoring, my_n_jobs, my_cv, logger)

    logger.info("Starting classifier search fit")
    print("Starting Cross Validation")

    #my_cv = make_pipeline(StandardScaler(), SVC(probability=True))
    # Actual pipeline fit takes place here
    my_cv.fit(X_train, y_train)

    dump(my_cv, Path(myEnv.log_dir, classifier+'_'+start_time +'.joblib'))

    y_pred = my_cv.predict(X_test)
    y_pred_prob = my_cv.predict_proba(X_test)[:, 1]

    logger.info("Generating roc_curve with y_test, y_pred_prob")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plot_roc(fpr, tpr, classifier, my_scoring, training_set, logger, myEnv, start_time)
    plot_matrix(my_cv, X_test, y_test, classifier, my_scoring, myEnv, start_time)

    # Printing stuff
    print_stuff(classifier, my_cv, my_scoring, X_test, y_test, y_pred, y_pred_prob)

    # Write same stuff to log
    log_results(logger, my_cv, X_test, y_test, y_pred, y_pred_prob)
