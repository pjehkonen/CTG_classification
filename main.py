import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ctg_classifiers.KNN_classification import CTG_KNN
from ctg_classifiers.SVM_classification import CTG_SVC
from ctg_classifiers.RF_classification import CTG_RF
from ctg_features.base_features import base_feat, spectrum_feat, autocorr_feat

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from set_env_dirs import in_triton
from ctg_lib.ca import ca_one
from ctg_lib.viz_two_minutes import vis_sample

from ctg_lib.ctg_time import now_time_string


def demo_spect():
    # make_spectrogram(pdg, salt_df)
    # make_demo(salt_df)
    # make_welch_psd(pdg, salt_df)
    # Create indices for elements available for training and testing
    return


def logging_data(logger, X, X_train, X_test, y, y_train, y_test, my_env, start_time):
    logger.info("X shape is    {}".format(X.shape))
    logger.info("X_train shape {}".format(X_train.shape))
    logger.info("X_test  shape {}".format(X_test.shape))
    logger.info("y shape is    {}".format(y.shape))
    logger.info("y_train shape {}".format(y_train.shape))
    logger.info("y_test shape  {}".format(y_test.shape))
    logger.info("Number of ZigZags in training set {}".format(np.sum(y_train)))
    logger.info("Number of ZigZags in test set     {}".format(np.sum(y_test)))

    np.savetxt(Path(my_env.log_dir, start_time + "/test_group.csv"), X_test.index.values, fmt="%d")
    np.savetxt(Path(my_env.log_dir, start_time + "/train_group.csv"), X_train.index.values, fmt="%d")

    logger.info("Test and Training indices written to log with this time_now as identifier")


def v_main(PrintDebuggingInfo, classifier, start_time, logger, my_env):
    vis_sample(PrintDebuggingInfo, classifier, start_time, logger, my_env)
    print("end v_sample")


def a_main(PrintDebuggingInfo, classifier, start_time, logger, operating_in_triton, my_env):
    ca_one(classifier, start_time, my_env, logger, operating_in_triton)
    print("end ca_one")


def c_main(pdg, classifier, start_time, logger, operating_in_triton, my_env):
    if operating_in_triton:
        full_data = True
    else:
        full_data = False

    plt.style.use('ggplot')

    logger.info("This is log file for classification algorithm of {}".format(out_dir))

    if full_data:
        #X, y = base_feat(my_env, logger)
        X, y = spectrum_feat(my_env, logger)
        #X, y = autocorr_feat(my_env, logger)

    else:
        #X, y = base_feat(my_env, logger, 10000)
        X, y = spectrum_feat(my_env, logger, 10000)
        #X, y = autocorr_feat(my_env, logger, 10000)

    my_test_size = 0.2
    use_shuffle = True

    # Now raw data is in X, where rows 0...normal_df.shape[1] contain cases with normal
    # and rows rows loc[-salt_df.shape[1]:] contain ZigZag cases.

    logger.info("Generating test and train sets with split {} and suffling set to {}".format(my_test_size, use_shuffle))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_test_size, shuffle=use_shuffle, stratify=y)

    logging_data(logger, X, X_train, X_test, y, y_train, y_test, my_env, start_time)

    # set up parameters for knn
    logger.info("Calling ctg classifier {}".format(classifier))

    if classifier == "K-NearestNeighbor":
        CTG_KNN(X_train, X_test, y_train, y_test, logger, classifier, my_env, start_time)
    elif classifier == "SupportVector":
        CTG_SVC(X_train, X_test, y_train, y_test, logger, classifier, my_env, start_time)
    elif classifier == 'RandomForest':
        CTG_RF(X_train, X_test, y_train, y_test, logger, classifier, my_env, start_time)

    ''''
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

    skf_acc = cross_val_score(knn, X, y, cv=skf)
    print(skf_acc)

    '''

    print("All done here, with {}".format(out_dir))
    logger.info("Finalized script classifying {}".format(out_dir))


if __name__ == '__main__':
    TASKS = ['classify', 'analyze', 'visualize']
    TASK = TASKS[1]
    classifiers = ["K-NearestNeighbor", "SupportVector", "RandomForest"]
    classifier = classifiers[2]

    start_time = now_time_string()
    PrintDebuggingInfo = True

    if PrintDebuggingInfo:
        print("Printing debugging information")

    if in_triton.in_triton():
        sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
        print("lib appended to Triton path")

    if TASK == 'classify':
        out_dir = classifier

    elif TASK == 'analyze':
        out_dir = 'Analysis_of_' + classifier

    elif TASK == 'visualize':
        out_dir = 'Visualization_of_' + classifier

    else:
        print("unknown task, exiting")
        sys.exit()

    operating_in_triton, my_env = setup_env.setup_env(PrintDebuggingInfo, output_dir=out_dir, log_start=start_time)
    logger = setup_log.setup_log(PrintDebuggingInfo, my_env, start_time)

    if TASK == 'classify':
        c_main(PrintDebuggingInfo, classifier, start_time, logger, operating_in_triton, my_env)
    elif TASK == 'analyze':
        a_main(PrintDebuggingInfo, classifier, start_time, logger, True, my_env)
    elif TASK == 'visualize':
        v_main(PrintDebuggingInfo, classifier, start_time, logger, my_env)
    else:
        print("error on selecting main, exiting")
        sys.exit()
