import math
from pennylane import numpy as np



class Dataloader:
    def __init__(self, data, labels, batchsize):
        assert len(data) == len(labels), "Require the same amount of labels as data"
        self.data = data
        self.labels = labels
        self.bs = batchsize

        self.permute()
        self.current = 0

    def permute(self):
        perm = np.random.permutation(len(self.data))
        self.data_perm = self.data[perm]
        self.labels_perm = self.labels[perm]

    def __len__(self):
        return math.ceil(len(self.data)/self.bs)

    def __next__(self):
        if self.current < len(self):
            lower, upper = self.current*self.bs, (self.current+1)*self.bs
            batch = self.data_perm[lower:upper], self.labels_perm[lower:upper]
            self.current += 1
            return batch
        else:
            self.permute()
            raise StopIteration

    def __get__(self, ind):
        return data[i], labels[i]

    def __iter__(self):
        self.current = 0
        self.permute()
        return self
