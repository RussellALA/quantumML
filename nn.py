from pennylane import numpy as np
from datetime import datetime
import os

class ClassicalNN:
    def __init__(self, circuit, params):
        self.n_features = params["dim_x"]
        self.n_layers = params.get("n_layers", 2)
        self.hidden_dim = params.get("hidden_dim", 3)
        self.n_params = self.hidden_dim*self.n_features + self.hidden_dim + (self.n_layers - 1)*(self.hidden_dim**2)
        self.parameters = [np.random.randn(self.n_features, self.hidden_dim) * 0.01] + \
                          [np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01] * (self.n_layers -1) + \
                          [np.random.randn(self.hidden_dim, 1) * 0.01]

        self.bias = [np.zeros(self.hidden_dim)] * (self.n_layers) + \
                    [np.zeros(1)]
        self.relu = lambda x : np.where(x > 0, x, 0.1*x)
        self.sigmoid = lambda x : 1/(1+np.exp(-x))
        self.modelname = params["name"] + "_" + datetime.now().strftime("%m%d_%H%M")

    def __call__(self, x, parameters=None, bias=None):
        if parameters is None:
            parameters = self.parameters
        if bias is None:
            bias = self.bias


        out = (parameters[0].T @ x) + bias[0]


        for l in range(1, self.n_layers+1):
            out = self.relu(out)
            out = (parameters[l].T @ out) + bias[l]


        out = self.sigmoid(out)
        out = (out - 0.5)*2

        return out

    def update(self, var):
        self.parameters, self.bias = var[0], var[1]

    def var(self):
        return self.parameters, self.bias

    def save(self):
        os.makedirs("outputs", exist_ok=True)
        os.makedirs(os.path.join("outputs", self.modelname), exist_ok=True)
        np.savez(os.path.join("outputs", self.modelname, "model"), parameters=self.parameters, bias=self.bias)

    def load(self, path):
        modeldata = np.load(path, allow_pickle=True)
        self.parameters = modeldata["parameters"]
        self.bias = modeldata["bias"]
