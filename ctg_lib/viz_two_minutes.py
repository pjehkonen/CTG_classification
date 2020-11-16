import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import json
import sys
from scipy.stats import median_absolute_deviation
from ctg_features.base_features import base_feat, raw_vectors




def vis_sample(PrintDebuggingInfo, classifier, start_time, logger, my_env):
    X, y = raw_vectors(my_env, logger)

    index1 = 333
    index2 = 109


    x_axis = np.arange(0,480)
    x_label_index = [i for i in range(0,481, 40)]
    x_labels = [str(int(i/4)) for i in x_label_index]

    plt.rcParams["figure.dpi"] = 100
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    axes[0].set_ylim(40,220)
    axes[1].set_ylim(40,220)
    axes[0].title.set_text('Sample {}'.format(index1))
    axes[1].title.set_text('Sample {}'.format(index2))

    axes[0].minorticks_on()
    axes[0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes[0].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    axes[1].minorticks_on()
    axes[1].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes[1].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    X.iloc[index1].plot(x=x_axis)