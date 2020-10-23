import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from set_env_dirs import in_triton
from ctg_lib import import_data
#from ctg_classifiers.random_train_test_indices import train_test_split
from ctg_classifiers.make_feature import make_feats
from ctg_lib.ctg_time import now_time_string
from ctg_features.spectrum import make_welch_psd

def make_y_df(n_size, s_size):
    y_array = np.zeros(n_size+s_size, dtype=int)
    y_array[-s_size:] = 1
    yy = {'y':y_array}
    return pd.DataFrame(yy)
    
def demo_spect():
    # make_spectrogram(pdg, salt_df)
    # make_demo(salt_df)
    # make_welch_psd(pdg, salt_df)
    # Create indices for elements available for training and testing
    return

def main(pdg, classifier):

    plt.style.use('ggplot')

    if in_triton.in_triton():
        sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
        print("lib appended to Triton path")

    out_dir = classifier
    start_time = now_time_string()

    operating_in_triton, my_env = setup_env.setup_env(pdg, output_dir=out_dir, log_start=start_time)
    logger = setup_log.setup_log(pdg, my_env, start_time)

    logger.info("This is log file for classification algorithm of {}".format(out_dir))

    # Read in dataframes
    normal_df, salt_df = import_data.import_data(False, my_env)


    X = pd.concat([normal_df, salt_df], ignore_index=True, axis=1).T
    y = make_y_df(normal_df.shape[1],salt_df.shape[1])

    # Now raw data is in X, where rows 0...normal_df.shape[1] contain cases with normal
    # and rows rows loc[-salt_df.shape[1]:] contain ZigZag cases.

    # Setting up classifier environment
    os.environ['OMP_NUM_THREADS'] = '4'
    nn_jobs = -1

    # Make one shot split with shuffling enabled, otherwise splitting occurs linearly.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # set up parameters for knn
    nn_neighbors = 3
    nn_jobs = -1
    knn = KNeighborsClassifier(n_neighbors=nn_neighbors, n_jobs=nn_jobs)
    knn.fit(X_train,y_train.values.ravel())
    y_pred =knn.predict(X_test)

    print("Tarkkuus:", metrics.accuracy_score(y_test.values.ravel(), y_pred))
    print("Score: ", knn.score(X_test, y_test.values.ravel()))
    ''''
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

    skf_acc = cross_val_score(knn, X, y, cv=skf)
    print(skf_acc)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    '''

    print("All done here, with {}".format(out_dir))
    logger.info("Finalized script classifying {}".format(out_dir))


if __name__ == '__main__':
    PrintDebuggingInfo = True
    classifier = "K-NearestNeighbor"

    if PrintDebuggingInfo:
        print("Printing debugging information")

    main(PrintDebuggingInfo, classifier)
