from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def create_plots(model, data_signal, data_background, data_store, params):
    for plotname in params["plots"]:
        eval(plotname + "(model, data_signal, data_background, data_store)")


def decision_regions(model, data_signal, data_background, data_store, n_points=1000, grid_res=150):
    with PdfPages(os.path.join("outputs", model.modelname, "decision_regions.pdf")) as pp:
        pixels = np.empty((grid_res, grid_res))

        for i, x in enumerate(np.linspace(0, 4, 150)):
            for j, y in enumerate(np.linspace(0, 4, 150)):
                pixels[i, j] = model(np.array([x,y]))
        plt.figure(figsize=(15, 15))
        plt.imshow(pixels, cmap="bwr", alpha=0.3, extent=[0, 4, 0, 4])
        plt.scatter(data_signal[:n_points,0], data_signal[:n_points,1], color="red", alpha=0.5)
        plt.scatter(data_background[:n_points,0], data_background[:n_points,1], alpha=0.5)
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()


def loss_plots(model, data_signal, data_background, data_store):
    if data_store.get("loss") is None:
        print("No losses found in data store, skipping loss plots")
        return
    with PdfPages(os.path.join("outputs", model.modelname, "losses.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        plt.plot(data_store["loss"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()

def accuracy_plots(model, data_signal, data_background, data_store):
    if data_store.get("accuracy") is None:
        print("No accuracy found in data store, skipping accuracy plots")
        return
    with PdfPages(os.path.join("outputs", model.modelname, "accuracy.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        plt.plot(data_store["accuracy"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()

def ROC_curves(model, data_signal, data_background, data_store, num_points=10000):
    with PdfPages(os.path.join("outputs", model.modelname, "ROC_curves.pdf")) as pp:
        num_points = min(len(data_signal), len(data_background), num_points)
        sig_pred = np.array([model(event) for event in data_signal[:num_points]])
        back_pred = np.array([model(event) for event in data_background[:num_points]])
        min_label, max_label = min(np.min(sig_pred), np.min(back_pred)), max(np.max(sig_pred), np.max(back_pred))
        acc_sig = []
        acc_back = []
        for label in np.linspace(min_label, max_label, 50):
            acc_sig.append(np.sum(sig_pred > label)/num_points)
            acc_back.append(np.sum(back_pred < label)/num_points)
        plt.figure(figsize=(15, 15))
        plt.plot(acc_back, acc_sig, label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()
