import pandas as pd
import numpy as np


def saveResult(Xid, y_pred, labelEncoder, path):
    ids = []  # list of ids
    cts = []  # list of countries
    for i in range(len(Xid)):
        idx = Xid[i]
        ids += [idx] * 5
        cts += labelEncoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    # Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv(path, index=False)
