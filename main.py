import sys

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from set_env_dirs import in_triton

import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt


from sklearn import preprocessing

if __name__ == '__main__':

    # Making sure
    if in_triton.in_triton():
        sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
        print("lib appended to Triton path")

    out_dir = "LogisticRegression"
    inTriton, myEnv = setup_env.setup_env(True, output_dir=out_dir)
    logger = setup_log.setup_log(True, myEnv)
    logger.info("This is log file for classification algorithm of {}".format(out_dir))

    print("All done here, with {}".format(out_dir))
    logger.info("Finalized script classifying {}".format(out_dir))