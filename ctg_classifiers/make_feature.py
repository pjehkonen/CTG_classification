from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from sklearn import preprocessing


def make_feats(pdg, start_time, my_env, normal_df, salt_df, X_train, X_test, y_train, y_test):

    n_df = pd.DataFrame()

    n_df['STD'] = normal_df.std()
    n_df['Mean'] = normal_df.mean()

    s_df = pd.DataFrame()

