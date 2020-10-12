import sys

from set_env_dirs import setup_env
from set_env_dirs import setup_log
from set_env_dirs import in_triton
from ctg_lib import import_data

import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from sklearn import preprocessing




def main(pdg):
    if in_triton.in_triton():
        sys.path.append('/scratch/cs/salka/PJ_SALKA/CTG_classification/ctg_lib')
        print("lib appended to Triton path")

    out_dir = "LogisticRegression"
    inTriton, myEnv = setup_env.setup_env(pdg, output_dir=out_dir)
    logger = setup_log.setup_log(pdg, myEnv)
    logger.info("This is log file for classification algorithm of {}".format(out_dir))

    normal_df, salt_df = import_data(pdg, myEnv)


    print("All done here, with {}".format(out_dir))
    logger.info("Finalized script classifying {}".format(out_dir))


if __name__ == '__main__':
    PrintDebuggingInfo = True
    if PrintDebuggingInfo:
        print("Pringing debugging information")
    main(PrintDebuggingInfo)
