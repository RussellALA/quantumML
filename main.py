import pennylane as qml
from pennylane import numpy as np
import yaml
import sys
from util import load_data, normalize, split_data
from vqc import Model, circuit, bind_to_device
from dataloader import Dataloader
from training import get_optimizer, train
from plotting import create_plots
import os




def main():
    data_store = {}

    with open(sys.argv[1]) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    data_signal, data_background = load_data(params["data"],
                                            n_signal=100000,
                                            n_background=100000)
    data_signal, data_background = normalize(data_signal, data_background)
    params["dim_x"] = data_signal.shape[-1]
    device = qml.device('default.qubit',
                        wires=params["dim_x"],
                        shots=params.get("n_shots", 1))
    bound_circ = bind_to_device(circuit, device)
    model = Model(bound_circ, params)
    data_train, labels_train, data_val, labels_val = split_data(data_signal,
                                            data_background,
                                            params.get("test_ratio", 0.2),
                                            1 - params.get("test_ratio", 0.2))
    trainloader = Dataloader(data_train, labels_train, params.get("bs", 1024))
    valloader = Dataloader(data_val, labels_val, params.get("bs", 1024))
    opt = get_optimizer(params)
    train(trainloader, valloader, model, opt, params.get("n_epochs", 5), data_store)
    os.makedirs(os.path.join("plots", model.modelname), exist_ok=True)
    create_plots(model, data_signal, data_background, data_store, params)
    print("All done \U0001F919")

if __name__ == "__main__":
    main()
