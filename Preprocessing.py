import random
import numpy as np
import pandas as pd


def fill_missing_value(data):
    gender = ['male', 'female']
    x = random.randint(0, 1)
    data['gender'].fillna(value=gender[x], inplace=True)
    return data


def lbl_encoder(col):
    col = np.array(col)
    for i in range(len(col)):
        if col[i] == 'male':
            col[i] = 1
        else:
            col[i] = -1
    return pd.Series(col)


def species_encoder(c1, c2, col):
    col = np.array(col)
    for i in range(len(col)):
        if col[i] == c1:
            col[i] = 1
        elif col[i] == c2:
            col[i] = -1
    return pd.Series(col)


def scaling(col, min, max):
    col = np.array(col)
    s = []
    for i in range(len(col)):
        s.append(min + (np.float32(col[i] - np.min(col)) / np.float32(np.max(col) - np.min(col))) * (max - min))
    return pd.Series(s)
