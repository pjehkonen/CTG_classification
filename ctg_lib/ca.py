# Classification analysis

from ctg_features.base_features import base_feat, spectrum_feat
from set_env_dirs import setup_env
from set_env_dirs import setup_log
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import plot_confusion_matrix
from ctg_lib.ctg_path_env import in_triton
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from ctg_lib.viz_two_minutes import vis_sample


import matplotlib

if in_triton():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import dump, load



def vizz(train_i, test_i, y_train, y_train_pred, y_test, y_test_pred, classifier, start_time, logger, my_env):


     # First find the index where predicted labels do not match, original X and y rows are the index
    index_of_training_errors = train_i[y_train!=y_train_pred]
    index_of_testing_errors =  test_i[y_test!=y_test_pred]

    vis_sample(True, classifier, start_time, logger, my_env, index_of_training_errors, index_of_testing_errors)


def plot_matrix(model, X_test, y_test):
    plt.style.use('default')
    fig = plt.figure(figsize=(8,8), dpi=150)

    my_title = "{} Confusion Matrix ({})".format(model.estimator.steps[1][0], model.scoring)
    class_names = ['normal','zigzag']
    plot_confusion_matrix(model, X_test, y_test,
                          display_labels=class_names,
                          cmap=plt.cm.Blues,
                          normalize=None,
                          colorbar=False
                          )
    plt.title(my_title)

    plt.show()
    fig = plt.figure(figsize=(8, 8), dpi=150)

    plt.figure(figsize=(8, 8), dpi=150)
    plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='all')
    plt.title(my_title)
    plt.show()


def plot_roc(pipe_model, ground_truth, estimate, message):
    fpr, tpr, thresholds = roc_curve(ground_truth, estimate)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {:.4f}".format(auc(fpr, tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    #plt.title("{} AUC ({}) {}".format(pipe_model.estimator.steps[1][0], pipe_model.scoring, message))
    plt.title("{} AUC".format(message))
    plt.show()


def ca_one(classifier, start_time, my_env, logger, operating_in_triton):
    #orig_time = '2020-11-12_20-20-28'
    #orig_time = '2020-11-15_19-15-20'
    #orig_time = '2020-12-01_08-55-29'
    #orig_time = '2020-11-03_13-54-01'
    #orig_time = '2020-11-03_13-46-04'
    #orig_time = '2020-11-05_10-03-09'
    orig_time ='2020-12-01_08-55-29'
    if operating_in_triton:
        tt_path = Path('/run/user/1000/gvfs/smb-share:server=data.triton.aalto.fi,share=scratch/cs/salka/Results/'+classifier+'/log/'+orig_time)
    else:
        tt_path = Path('/media/jehkonp1/SecureLacie/DATA2h/Scratch/' + classifier + '/log/' + orig_time)
    trained_classifier = Path(tt_path.parent, tt_path.parent.parent.name + '_' + tt_path.name + '.joblib')

    model = load(trained_classifier)

    X, y = base_feat(my_env, logger)
    #X, y = spectrum_feat(my_env, logger)

    train_i = np.loadtxt(Path(tt_path, 'train_group.csv'), dtype=int)
    test_i = np.loadtxt(Path(tt_path, 'test_group.csv'), dtype=int)
    X_train = X.loc[train_i]
    X_test = X.loc[test_i]
    y_train = y[train_i]
    y_test = y[test_i]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    best_model = model.best_estimator_[1]
    #best_model.fit(X_train_scaled, y_train)

    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    y_train_pred_prob = best_model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    plot_roc(model, y_train, y_train_pred_prob, "Training")
    plot_roc(model, y_test, y_test_pred_prob, "Testing")

    plot_matrix(model, X_train, y_train)
    plot_matrix(model, X_test, y_test)

    # printing stuff
    print("Training Accuracy is ", best_model.score(X_train_scaled, y_train))
    print("Test Accuracy is ", best_model.score(X_test_scaled, y_test))
    print("Training Confusion matrix")
    print(confusion_matrix(y_train, y_train_pred))
    print("Test confusion matrix")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification report training")
    print(classification_report(y_train, y_train_pred))
    print("Classification report testing")
    print(classification_report(y_test, y_test_pred))

    #vizz(train_i, test_i, y_train, y_train_pred, y_test, y_test_pred, classifier, start_time, logger, my_env)

    print("end ca_one function")
