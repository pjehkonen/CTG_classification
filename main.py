import sys

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from ctg_lib import write_log
from set_env_dirs import in_triton
from ctg_lib import import_data
from ctg_classifiers.random_train_test_indices import train_test_split
from ctg_lib.ctg_time import now_time_string

from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from sklearn import preprocessing


def main(pdg, classifier):
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

    # Create indices for elements available for training and testing
    X_train, X_test, y_train, y_test = train_test_split(pdg, my_env, logger, classifier, start_time,
                                                        normal_df.shape[1], salt_df.shape[1], split=0.2)


    print("All done here, with {}".format(out_dir))
    logger.info("Finalized script classifying {}".format(out_dir))


if __name__ == '__main__':
    PrintDebuggingInfo = True
    classifier = "LogisticRegression"

    if PrintDebuggingInfo:
        print("Printing debugging information")

    main(PrintDebuggingInfo, classifier)
