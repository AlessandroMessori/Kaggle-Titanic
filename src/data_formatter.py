import numpy as np
import pandas as pd


def weight_preds(preds):
    new_preds = list()
    for i in range(len(preds[0])):
        weighted = sum([pred[i] for pred in preds])
        new_preds.append(1 if (weighted > 4) else 0)

    return new_preds


def save_preds_tocsv(data, preds):
    preds = weight_preds(preds)

    output = {
        'PassengerID': data,
        'Survived': np.array(preds, dtype='int64')
    }

    output = pd.DataFrame.from_dict(output)
    output.set_index(output.columns[1])
    output = output.iloc[:, [0, 1]]

    output.to_csv(path_or_buf='./data/preds.csv', index=False)
