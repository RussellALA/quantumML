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
    data_min = np.min(data_signal, axis=0), np.min(data_background, axis=0)
    min_pt, min_e_miss = min(data_min[0][0], data_min[1][0]), min(data_min[0][1], data_min[1][1])
    data_max = np.max(data_signal, axis=0), np.max(data_background, axis=0)
    max_pt, max_e_miss = max(data_max[0][0], data_max[1][0]), max(data_max[0][1], data_max[1][1])
    data_signal[:, 0] -= min_pt
    data_signal[:, 1] -= min_e_miss
    data_background[:, 0] -= min_pt
    data_background[:, 1] -= min_e_miss

    data_signal[:, 0] /= (max_pt - min_pt)/np.pi
    data_signal[:, 1] /= (max_e_miss - min_e_miss)/np.pi
    data_background[:, 0] /= (max_pt - min_pt)/np.pi
    data_background[:, 1] /= (max_e_miss - min_e_miss)/np.pi
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

def cost_gaussian(v, model, X, Y):
    out, jacobian = np.array([model(x, v[0], v[1]) for x in X])
    return gaussian_loss(out, jacobian)

def gaussian_loss(out, jacobian):
    return np.mean(out**2)/2 - np.mean(np.log(jacobian))/out.shape[1]
