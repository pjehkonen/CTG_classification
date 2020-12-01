from ctg_lib import import_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def log_features(X, logger, feature_func):
    feats = X.columns
    logger.info("****** Feature Creation **********")
    logger.info("Created features with {}".format(feature_func))
    logger.info("Features include {}".format(feats))
    logger.info("Feature describe")
    logger.info("\n{}".format(X.describe()))


def make_y_df(n_size, s_size):
    y_array = np.zeros(n_size + s_size, dtype=int)
    y_array[-s_size:] = 1
    return y_array


def raw_vectors(my_env, logger):
    normal_df, salt_df = import_data.import_data(False, my_env)
    X = pd.concat([normal_df, salt_df], ignore_index=True, axis=1)
    y = make_y_df(normal_df.shape[1], salt_df.shape[1])
    return X, y


def base_feat(my_env, logger, dsetsize=None):
    normal_df, salt_df = import_data.import_data(False, my_env)
    X_in = pd.concat([normal_df, salt_df], ignore_index=True, axis=1)
    y = make_y_df(normal_df.shape[1], salt_df.shape[1])
    X = pd.DataFrame()
    X['mean'] = X_in.mean()
    X['std'] = X_in.std()
    X['mad'] = X_in.mad(axis=0)
    X['diff'] = X_in.diff().sum()
    X['cumsum'] = X_in.cumsum(axis=0).iloc[-1].values

    log_features(X, logger, "base_feat")

    if dsetsize is not None:
        X = X.sample(dsetsize)
        y = y[X.index.values]

    return X, y


def smart_scale(my_vector):
    my_vector = my_vector - np.median(my_vector) # center

    if my_vector.max() < 1.0:
        return my_vector
    else:
        return my_vector / np.max(my_vector) # scale


def gen_spect(sample):
    sample = sample-np.median(sample)
    ps = np.abs(np.fft.fft(sample)) ** 2
    time_step = 0.25
    zoom = 2
    freqs = np.fft.fftfreq(sample.size, time_step)
    idx = np.argsort(freqs)
    half_way = len(freqs) // 2
    ps2 = 2 * ps[:half_way]

    ps2 = smart_scale(ps2)

    bin = []
    bin.append(np.sum(ps2[2:5]))   # lowest frequency bin excluding near DC
    bin.append(np.sum(ps2[5:12]))  # second lowest bin of frequencies
    bin.append(np.sum(ps2[12:30])) # mid segment of the spectral energy
    bin.append(np.sum(ps2[12:30])) # rest of the spectral energy

    return bin


def spectrum_feat(my_env, logger, dsetsize=None):
    normal_df, salt_df = import_data.import_data(False, my_env)
    X_in = pd.concat([normal_df, salt_df], ignore_index=True, axis=1)
    y = make_y_df(normal_df.shape[1], salt_df.shape[1])

    dc, low, mid, rest = [], [], [], []
    for column in X_in.columns:
        bins = gen_spect(X_in[column].values)
        dc.append(bins[0])
        low.append(bins[1])
        mid.append(bins[2])
        rest.append(bins[3])

    X = pd.DataFrame(np.array([dc, low, mid, rest]).T, columns=['FRQ_DC','FRQ_LOW', 'FRQ_MID','FRQ_HIGH'])

    if dsetsize is not None:
        X = X.sample(dsetsize)
        y = y[X.index.values]

    log_features(X, logger, "spectrum_feat")
    return X, y


def autocorr_feat(my_env, logger, dsetsize=None):
    normal_df, salt_df = import_data.import_data(False, my_env)
    X_in = pd.concat([normal_df, salt_df], ignore_index=True, axis=1)
    y = make_y_df(normal_df.shape[1], salt_df.shape[1])

    arvot = smart_scale(X_in[2].values)

    dc, low, mid, rest = [], [], [], []
    for column in X_in.columns:
        bins = gen_spect(X_in[column].values)
        dc.append(bins[0])
        low.append(bins[1])
        mid.append(bins[2])
        rest.append(bins[3])

    X = pd.DataFrame(np.array([dc, low, mid, rest]).T, columns=['FRQ_DC','FRQ_LOW', 'FRQ_MID','FRQ_HIGH'])

    if dsetsize is not None:
        X = X.sample(dsetsize)
        y = y[X.index.values]

    log_features(X, logger, "spectrum_feat")
    return X, y