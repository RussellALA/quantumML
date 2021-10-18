from util import cost
from tqdm import tqdm, trange
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np


def validate(valloader, model):
    val_loss = 0
    for x, y in valloader:
        val_loss += cost(model.var(), model, x, y)/len(valloader)
    return val_loss

def accuracy(valloader, model):
    acc = 0.0
    for x, y in valloader:
        out = np.array([model(data) for data in x])
        acc += np.sum(np.sign(y) == np.sign(out))/(len(valloader)*x.shape[0])
    return acc

def get_optimizer(params):
    return AdamOptimizer(params.get("lr", 0.005),
                        beta1=params.get("beta1", 0.9),
                        beta2=params.get("beta2", 0.999))


def train(trainloader, valloader, model, opt, n_epochs, data_store):
    data_store["loss"] = []
    data_store["accuracy"] = []
    for epoch in trange(n_epochs):
        for i, (x, y) in enumerate(tqdm(trainloader, leave=False)):
            model.update(opt.step(lambda v: cost(v, model, x, y), model.var()))
        model.save()
        data_store["loss"].append(validate(valloader, model))
        data_store["accuracy"].append(accuracy(valloader, model))
        loss = data_store["loss"][-1]
        tqdm.write(f"Validation loss in epoch {epoch}: {loss}")
    model.save()
