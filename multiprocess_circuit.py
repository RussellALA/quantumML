import threading, queue
from functools import wraps
import math
import numpy as np


def mcirc(func):

    @wraps(func)
    def batchwise_multi(*args, x=None, **kwargs):
        q = queue.Queue()


        results = []

        def worker():
            while True:
                item = q.get()
                results.append(func(item, *args, **kwargs))
                q.task_done()

        threading.Thread(target=worker, daemon=True).start()

        for data in x:
            q.put(data)

        q.join()

        return np.array(results)

    return batchwise_multi
