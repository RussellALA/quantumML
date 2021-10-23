import pennylane as qml
from pennylane import numpy as np
from datetime import datetime
import os
from vqc import Model

def circuit(x, parameters):
    #State Preparation
    for i in range(parameters.shape[1]):
        qml.RY(x[i], wires=i)
    #Add layers
    for i in range(parameters.shape[0]):
        for j in range(parameters.shape[1]):
            qml.Rot(parameters[i, j, 0], parameters[i, j, 1], parameters[i, j, 2], wires=j)


        for j in range(parameters.shape[1]):
            if j + 1 < parameters.shape[1]:
                qml.CNOT(wires=[j,j+1])
            else:
                qml.CNOT(wires=[j,0])
    #For measurement apply Pauli-Z to first qubit
    return qml.expval(qml.PauliZ(wires=0))

class IQNN(VQC):
    def __init__(self, circuit, inverse_circuit, params):
        super().__init__()
        self.bias1 = np.zeros(self.n_features)
        self.bias2 = np.zeros(self.n_features)
        self.inverse_circuit = self.inverse_circuit
        self.sigmoid = lambda x : 1/(1 + np.exp(-x))
        self.inverse_sigmoid = lambda x : np.log(x/(1-x))

    def __call__(self, x, parameters=None, bias1=None, bias2=None, rev=False):
        if parameters is None:
            parameters = self.parameters
        if bias1 is None:
            bias1 = self.bias1
        if bias2 is None:
            bias2 = self.bias2
        if not rev:
            inp = self.sigmoid(x + bias1)
            measurement = self.circuit(inp)
            return self.inverse_sigmoid(measurement + bias2)
        else:
            inp = self.sigmoid(x - bias2)
            measurement = self.inverse_circuit(inp)
            return self.inverse_sigmoid(measurement - bias1)

    def update(self, var):
        self.parameters, self.bias1, self.bias2 = var[0], var[1], var[2]

    def var(self):
        return self.parameters, self.bias1, self.bias2


    def save(self):
        os.makedirs("outputs", exist_ok=True)
        os.makedirs(os.path.join("outputs", self.modelname), exist_ok=True)
        np.savez(os.path.join("models", self.modelname, "model"),
                parameters=self.parameters,
                bias1=self.bias1,
                bias2=self.bias2)

    def load(self, path):
        modeldata = np.load(path, allow_pickle=True)
        self.parameters = modeldata["parameters"]
        self.bias1 = modeldata["bias1"]
        self.bias2 = modeldata["bias2"]
