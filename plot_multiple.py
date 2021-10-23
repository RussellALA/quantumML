from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import yaml
import sys
from util import load_data, normalize, split_data
from vqc import VQC, circuit
from nn import ClassicalNN
import os
from datetime import datetime
import pennylane as qml


def create_multiple(models, data_signal, data_background, data_stores, paramcards):
    loss_plots(models, data_signal, data_background, data_stores)
    accuracy_plots(models, data_signal, data_background, data_stores)
    ROC_curves(models, data_signal, data_background, data_stores)


def loss_plots(models, data_signal, data_background, data_stores):
    with PdfPages(os.path.join("plots", "run_" +  datetime.now().strftime("%m%d_%H%M"), "losses.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        for model, data_store in zip(models, data_stores):
            if data_store.get("loss") is None:
                print("No losses found in data store, skipping loss plots")
                continue
            plt.plot(data_store["loss"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()

def accuracy_plots(models, data_signal, data_background, data_stores):
    with PdfPages(os.path.join("plots", "run_" +  datetime.now().strftime("%m%d_%H%M"), "accuracy.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        for model, data_store in zip(models, data_stores):
            if data_store.get("accuracy") is None:
                print("No accuracy found in data store, skipping accuracy plots")
                continue
            plt.plot(data_store["accuracy"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()

def ROC_curves(models, data_signal, data_background, data_stores, num_points=10000):
    with PdfPages(os.path.join("plots", "run_" +  datetime.now().strftime("%m%d_%H%M"), "ROC_curves.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        for model, data_store in zip(models, data_stores):
            num_points = min(len(data_signal), len(data_background), num_points)
            sig_pred = np.array([model(event) for event in data_signal[:num_points]])
            back_pred = np.array([model(event) for event in data_background[:num_points]])
            min_label, max_label = min(np.min(sig_pred), np.min(back_pred)), max(np.max(sig_pred), np.max(back_pred))
            acc_sig = []
            acc_back = []
            for label in np.linspace(min_label, max_label, 50):
                acc_sig.append(np.sum(sig_pred > label)/num_points)
                acc_back.append(np.sum(back_pred < label)/num_points)
            plt.plot(acc_back, acc_sig, label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()


def main():

    models = []
    data_stores = []
    paramcards = []
    for modelname in sys.argv[1:]:
        with open(os.path.join(modelname, "params.yaml")) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            data_signal, data_background = load_data(params["data"],
                                                    n_signal=1501,
                                                    n_background=1501)
            data_signal, data_background = normalize(data_signal, data_background)
            params["dim_x"] = data_signal.shape[-1]
            device = qml.device('default.qubit',
                                wires=params["dim_x"],
                                shots=params.get("n_shots", 1000))
            bound_circ = qml.QNode(circuit, device)
            paramcards.append(params)
            model = eval(params.get("model_type", "VQC") + "(bound_circ, params)")
            model.load(os.path.join(modelname, "model.npz"))
            models.append(model)
            data_store = np.load(os.path.join(modelname, "data_store.npz"), allow_pickle=True)["data_store"].item()
            data_stores.append(data_store)

    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("plots", "run_" +  datetime.now().strftime("%m%d_%H%M")), exist_ok=True)
    create_multiple(models, data_signal, data_background, data_stores, paramcards)


if __name__ == "__main__":
    main()
