from util import cost
from tqdm import tqdm, trange
from pennylane.optimize import AdamOptimizer, QNGOptimizer
from pennylane import numpy as np
from vqc import pin_x, circuit
import pennylane as qml

def validate(valloader, model):
    val_loss = 0
    for x, y in valloader:
        batch_loss = cost(model.var(), model, x, y)
        val_loss += batch_loss/len(valloader)
    return val_loss

def accuracy(valloader, model):
    acc = 0.0
    for x, y in valloader:
        out = np.squeeze(np.array([model(data) for data in x]))
        acc += np.sum(np.sign(y) == np.sign(out))/(len(valloader)*x.shape[0])
    return acc

def get_optimizer(params):
    if params.get("optim_type", "Adam") == "Adam":
        return AdamOptimizer(params.get("lr", 0.005),
                            beta1=params.get("beta1", 0.9),
                            beta2=params.get("beta2", 0.999))
    elif params.get("optim_type") == "Quantum":
        return QNGOptimizer(params.get("lr", 0.005))

def pad(array):
    ret = np.concatenate((array, np.zeros((len(array), 1))), axis=1)
    ret = np.concatenate((ret, np.zeros((1, len(ret)+1))), axis=0)
    ret[-1][-1] = 1
    return ret

def train(trainloader, valloader, model, opt, data_store, device, params):
    data_store["loss"] = []
    data_store["accuracy"] = []
    for epoch in trange(params.get("n_epochs", 5)):
        train_loss = 0
        for i, (x, y) in enumerate(tqdm(trainloader, leave=False)):
            if params.get("optim_type") == "Quantum":
                metric_tensor = lambda parameters: pad(qml.metric_tensor(pin_x(x, model.circuit, device))(parameters[0]))
                new_var, batch_loss = opt.step_and_cost(lambda v: cost(v, model, x, y), model.var(), metric_tensor_fn=metric_tensor)
            else:
                new_var, batch_loss = opt.step_and_cost(lambda v: cost(v, model, x, y), model.var())
            model.update(new_var)
            train_loss += batch_loss/len(trainloader)
        tqdm.write(f"Train loss in epoch {epoch}: {train_loss}")
        model.save()
        data_store["loss"].append(validate(valloader, model))
        data_store["accuracy"].append(accuracy(valloader, model))
        loss = data_store["loss"][-1]
        tqdm.write(f"Validation loss in epoch {epoch}: {loss}")
    model.save()
