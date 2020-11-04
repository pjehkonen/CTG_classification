# Classification analysis

from ctg_features.base_features import base_feat
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

import matplotlib

if in_triton():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import dump, load


def plot_roc(model, ground_truth, estimate, message):
    fpr, tpr, thresholds = roc_curve(ground_truth, estimate)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC {:.4f}".format(auc(fpr, tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} AUC ({}) {}".format(model.estimator.steps[1][0], model.scoring, message))
    plt.show()


def ca_one(classifier, start_time):
    tt_path = Path('/media/jehkonp1/SecureLacie/DATA2h/Scratch/' + classifier + '/log/' + start_time)
    trained_classifier = Path(tt_path.parent, tt_path.parent.parent.name + '_' + tt_path.name + '.joblib')

    model = load(trained_classifier)
    pdg = True
    out_dir = classifier
    start_time = '0101dummy'
    operating_in_triton, my_env = setup_env.setup_env(pdg, output_dir=out_dir, log_start=start_time)
    logger = setup_log.setup_log(pdg, my_env, start_time)

    X, y = base_feat(my_env, logger)

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns].to_numpy())

    train_i = np.loadtxt(Path(tt_path, 'train_group.csv'), dtype=int)
    test_i = np.loadtxt(Path(tt_path, 'test_group.csv'), dtype=int)
    X_train = X.loc[train_i]
    X_test = X.loc[test_i]
    y_train = y[train_i]
    y_test = y[test_i]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    plot_roc(model, y_test, y_test_pred, "y test")
    plot_roc(model, y_test, y_test_pred_prob, "y test prob")
    plot_roc(model, y_train, y_train_pred, "y train")
    plot_roc(model, y_train, y_train_pred_prob, "y tarin prob")
