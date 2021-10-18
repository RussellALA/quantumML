from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def create_plots(model, data_signal, data_background, data_store, params):
    for plotname in params["plots"]:
        eval(plotname + "(model, data_signal, data_background, data_store)")


def decision_regions(model, data_signal, data_background, data_store, n_points=1000, grid_res=150):
    with PdfPages(os.path.join("plots", model.modelname, "decision_regions.pdf")) as pp:
        pixels = np.empty((grid_res, grid_res))

        for i, x in enumerate(np.linspace(-3, 6, 150)):
            for j, y in enumerate(np.linspace(-3, 6, 150)):
                pixels[i, j] = model(np.array([x,y]))
        plt.figure(figsize=(15, 15))
        plt.imshow(pixels, cmap="bwr", alpha=0.3, extent=[-3, 6, -3, 6])
        plt.scatter(data_signal[:n_points,0], data_signal[:n_points,1], color="red", alpha=0.5)
        plt.scatter(data_background[:n_points,0], data_background[:n_points,1], alpha=0.5)
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()


def loss_plots(model, data_signal, data_background, data_store):
    with PdfPages(os.path.join("plots", model.modelname, "losses.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        plt.plot(data_store["loss"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)

def accuracy_plots(model, data_signal, data_background, data_store):
    with PdfPages(os.path.join("plots", model.modelname, "accuracy.pdf")) as pp:
        plt.figure(figsize=(15, 15))
        plt.plot(data_store["accuracy"], label=model.modelname.split("_")[0])
        plt.legend()
        plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
        plt.close()
