from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import json
import sys
from scipy.stats import median_absolute_deviation
from ctg_features.base_features import base_feat, raw_vectors

def display_spectrum(sample):
    sample = sample-np.median(sample)
    ps = np.abs(np.fft.fft(sample)) ** 2
    time_step = 0.25
    zoom = 2
    freqs = np.fft.fftfreq(sample.size, time_step)
    idx = np.argsort(freqs)
    half_way = len(freqs) // 2
    ps2 = 2 * ps[:half_way]
    ps2 = ps2/np.max(ps2)
    plt.plot(freqs[idx[half_way:half_way+(half_way//zoom)]], ps2[:half_way//zoom])
    plt.show()
    bin = []
    bin.append(np.sum(ps2[2:5]))   # lowest frequency bin excluding near DC
    bin.append(np.sum(ps2[5:12]))  # second lowest bin of frequencies
    bin.append(np.sum(ps2[12:]))   # rest of the spectral energy
    print(bin)

    print("finished plotting")


def display_just_fhr(X, y, index1, index2):

    x_label_index = [i for i in range(0,481, 40)]
    x_labels = [str(int(i/4)) for i in x_label_index]

    plt.rcParams["figure.dpi"] = 100
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    axes[0].set_ylim(40,220)
    axes[1].set_ylim(40,220)
    axes[0].title.set_text('Training Sample {} {}'.format(index1, 'zigzag' if y[index1]==1 else 'normal'))
    axes[1].title.set_text('Test Sample {} {}'.format(index2, 'zigzag' if y[index2]==1 else 'normal'))

    axes[0].minorticks_on()
    axes[0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes[0].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    axes[1].minorticks_on()
    axes[1].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes[1].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    zero_check = [0, 479]

    for zz in zero_check:
        if X.iloc[zz][index1] == 0:
            X.at[zz, index1] = 1
        if X.iloc[zz][index2] == 0:
            X.at[zz, index2] = 1

    X = X.replace(0, np.nan)
    X.plot(y=[index1],ax=axes[0], grid=True)
    X.plot(y=[index2],ax=axes[1], grid=True)

    fig.suptitle("Misspredicted training {} and test {}".format(index1, index2))
    axes[0].set_xticks(x_label_index)
    axes[0].set_xticklabels(x_labels)
    axes[1].set_xticks(x_label_index)
    axes[1].set_xticklabels(x_labels)

    axes[0].set_ylabel('Fetal heartbeat rate')
    axes[1].set_ylabel('Fetal heartbeat rate')

    axes[0].set_xlabel('Time (s)')
    axes[1].set_xlabel('Time (s)')

    plt.show()


def vis_sample(PrintDebuggingInfo, classifier, start_time, logger, my_env, train_list=[12345], test_list=[235234]):
    X, y = raw_vectors(my_env, logger)

    debug_this = True

    if debug_this:

        display_spectrum(X[0].values)
        display_spectrum(X[12345].values)
        display_spectrum(X[235234].values)
        display_spectrum(X[100000].values)

    else:
        for i, element in enumerate(test_list):
            if i>10:
                break
            display_just_fhr(X, y, train_list[i], element)

    print('end of vis_sample')

