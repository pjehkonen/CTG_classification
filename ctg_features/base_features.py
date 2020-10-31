from ctg_lib import import_data
import pandas as pd
import numpy as np


def log_features(X, logger, feature_func):
    feats = X.columns
    logger.info("****** Feature Creation **********")
    logger.info("Created features with {}".format(feature_func))
    logger.info("Features include {}".format(feats))
    logger.info("Feature describe")
    logger.info("\n{}".format(X.describe()))



def make_y_df(n_size, s_size):
    y_array = np.zeros(n_size+s_size, dtype=int)
    y_array[-s_size:] = 1
    return y_array

def base_feat(my_env, logger):
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
    return X, y


def spectrum_feat(my_env, logger):
    normal_df, salt_df = import_data.import_data(False, my_env)
    X_in = pd.concat([normal_df, salt_df], ignore_index=True, axis=1)
    y = make_y_df(normal_df.shape[1], salt_df.shape[1])
    X = pd.DataFrame()
    X['mean'] = X_in.mean()
    X['std'] = X_in.std()
    X['mad'] = X_in.mad(axis=0)
    X['diff'] = X_in.diff().sum()
    X['cumsum'] = X_in.cumsum(axis=0).iloc[-1].values

    log_features(X, logger, "spectrum_feat")
    return X, y