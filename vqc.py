import pennylane as qml
from pennylane import numpy as np
from datetime import datetime
import os

def bind_to_device(circ, device):
    @qml.qnode(device)
    def wrapped_circuit(*args, **kwargs):
        return circ(*args, **kwargs)

    return wrapped_circuit

def circuit(x, parameters):
    #State Preparation
    for i in range(parameters.shape[1]):
        qml.RY(x[i], wires=i)
        qml.RY(x[i], wires=i)
    #Add layers
    for i in range(parameters.shape[0]):
        for j in range(parameters.shape[1]):
            qml.Rot(parameters[i, j, 0], parameters[i, j, 1], parameters[i, j, 2], wires=j)


        for j in range(parameters.shape[1]):
            qml.CNOT(wires=[0,1])
            qml.CNOT(wires=[1,0])
    #For measurement apply Pauli-Z to first qubit
    return qml.expval(qml.PauliZ(wires=0))

class Model:
    def __init__(self, circuit, params):
        self.n_features = params["dim_x"]
        self.n_layers = params.get("n_layers", 2)
        self.n_params = self.n_layers*self.n_features*3
        self.parameters = np.random.randn(self.n_layers, self.n_features, 3) * 0.01
        self.bias = 0.0
        self.modelname = params["name"] + "_" + datetime.now().strftime("%m%d_%H%M")
        self.circuit = circuit

    def __call__(self, x, parameters=None, bias=None):
        if parameters is None:
            parameters = self.parameters
        if bias is None:
            bias = self.bias
        return self.circuit(x, parameters) + bias

    def update(self, var):
        self.parameters, self.bias = var[0], var[1]

    def var(self):
        return self.parameters, self.bias

    def save(self):
        os.makedirs("models", exist_ok=True)
        np.savez(os.path.join("models", self.modelname), parameters=self.parameters, bias=self.bias)

    def load(self, path):
        modeldata = np.load(path)
        self.parameters = modeldata["parameters"]
        self.bias = modeldata["bias"]
