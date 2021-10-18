from pennylane import numpy as np
import pandas as pd
import math
import os

def load_data(path_name, n_signal=None, n_background=None):
    for filename in os.listdir(path_name):
        if "background" in filename and filename.endswith(".csv"):
            data_background = pd.read_csv(os.path.join(path_name, filename)).to_numpy()[:,1:]
        if "signal" in filename and filename.endswith(".csv"):
            data_signal = pd.read_csv(os.path.join(path_name, filename)).to_numpy()[:,1:]
    if not (n_background is None):
        data_background = data_background[:n_background]
    if not (n_signal is None):
        data_signal = data_signal[:n_signal]
    return data_signal, data_background

def normalize(data_signal, data_background):
    data_total = np.concatenate((data_background, data_signal), axis=0)
    data_background = (data_background - np.mean(data_total, axis=0))/np.std(data_total, axis=0)
    data_signal = (data_signal - np.mean(data_total, axis=0))/np.std(data_total, axis=0)
    return data_signal, data_background


def split_data(data_signal, data_background, test_rat=0.2, train_rat=0.8):
    data_train = np.concatenate([data_signal[:math.ceil(train_rat*len(data_signal))], data_background[:math.ceil(train_rat*len(data_background))]], axis=0)
    labels_train = np.concatenate([np.ones(math.ceil(train_rat*len(data_signal))), -np.ones(math.ceil(train_rat*len(data_background)))], axis=0)
    data_val = np.concatenate([data_signal[math.ceil(train_rat*len(data_signal)):], data_background[math.ceil(train_rat*len(data_background)):]], axis=0)
    labels_val = np.concatenate([np.ones(math.floor(test_rat*len(data_signal))), -np.ones(math.floor(test_rat*len(data_background)))], axis=0)
    return data_train, labels_train, data_val, labels_val

def cost(v, model, X, Y):
    out = np.array([model(x, v[0], v[1]) for x in X])
    return loss(Y, out)


def loss(y, out):
    return np.mean((y - out)**2)
