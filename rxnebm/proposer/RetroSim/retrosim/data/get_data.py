from __future__ import absolute_import, print_function

import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data_df(fpath='data_processed.csv'):
    return pd.read_csv(fpath)

def split_data_df(data, val_frac=0.1, test_frac=0.1, shuffle=False, seed=None):
    # Define shuffling
    if shuffle:
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)

        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass

    # Go through each class
    classes = sorted(np.unique(data['class']))
    for class_ in classes:
        indeces = data.loc[data['class'] == class_].index
        N = len(indeces)
        print('{} rows with class value {}'.format(N, class_))

        shuffle_func(indeces)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)

        for i in indeces[:train_end]:
            data.at[i, 'dataset'] = 'train' 
            # data.set_value(i, 'dataset', 'train')
        for i in indeces[train_end:val_end]:
            data.at[i, 'dataset'] = 'val' 
            # data.set_value(i, 'dataset', 'val')
        for i in indeces[val_end:]:
            data.at[i, 'dataset'] = 'test'
            # data.set_value(i, 'dataset', 'test')
    print(data['dataset'].value_counts())

def split_data_df_old(data, val_frac=0.1, test_frac=0.1, shuffle=False, seed=None):
    # Define shuffling
    if shuffle:
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)

        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass

    # Go through each class
    classes = sorted(np.unique(data['class']))
    for class_ in classes:
        indeces = data.loc[data['class'] == class_].index
        N = len(indeces)
        print('{} rows with class value {}'.format(N, class_))

        shuffle_func(indeces)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)

        for i in indeces[:train_end]:
            # data.at[i, 'dataset'] = 'train' 
            data.set_value(i, 'dataset', 'train')
        for i in indeces[train_end:val_end]:
            # data.at[i, 'dataset'] = 'val' 
            data.set_value(i, 'dataset', 'val')
        for i in indeces[val_end:]:
            # data.at[i, 'dataset'] = 'test'
            data.set_value(i, 'dataset', 'test')
    print(data['dataset'].value_counts())


if __name__ == '__main__':
    data = get_data_df()
    split_data_df(data)
    write_to_files(data)
