from pathlib import Path
import os
import sys
import git
import logging

from ctg_path_env import CTGPaths
from set_env_dirs import setup_env

import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt


from sklearn import preprocessing

if __name__ == '__main__':

    out_dir = "LogisticRegression"
    inTriton, myEnv = setup_env.setup_env(True, output_dir=out_dir)

    print("Hiphei")