import numpy as np
from pathlib import Path

def train_test_split(pdg, my_env, logger, classifier, start_time, X, y, split=0.2):

    X_train_end = int(X*(1-split))
    y_train_end = int(y*(1-split))

    rng = np.random.default_rng()
    X_permuted = rng.permutation(range(X))
    y_permuted = rng.permutation(range(y))

    X_train = X_permuted[:X_train_end]
    X_test = X_permuted[X_train_end:]

    y_train = y_permuted[:y_train_end]
    y_test = y_permuted[y_train_end:]


    logger.info("Writing X training set indices to file {}".format(classifier+'/'+start_time+'/X_train.indices'))
    logger.info("Writing X test set indices to file {}".format(classifier+'/'+start_time+'/X_test.indices'))
    logger.info("Writing y training set indices to file {}".format(classifier+'/'+start_time+'/y_train.indices'))
    logger.info("Writing y test set indices to file {}".format(classifier+'/'+start_time+'/y_test.indices'))


    np.savetxt(Path(my_env.sets_dir,'X_train.indices'), X_train, delimiter=',', fmt='%d')
    np.savetxt(Path(my_env.sets_dir, 'X_test.indices'), X_test, delimiter=',', fmt='%d')
    np.savetxt(Path(my_env.sets_dir, 'y_train.indices'), y_train, delimiter=',', fmt='%d')
    np.savetxt(Path(my_env.sets_dir, 'y_test.indices'), y_test, delimiter=',', fmt='%d')


    if pdg:
        print("Sets are divided to {}% training and {}% testing".format(int(100-int(split*100)),int(100*split)))

        print("Total length of set X is {} entries".format(X))
        print("X training contains {} entries".format(len(X_train)))
        print("X test contains {} entries".format(len(X_test)))

        print("Total length of set y is {} entries".format(y))
        print("y training contains {} entries".format(len(y_train)))
        print("y test contains {} entries".format(len(y_test)))


    return X_train, X_test, y_train, y_test