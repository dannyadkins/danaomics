import numpy as np
import pandas as pd
import torch
import torch.nn as nn

token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_tokenized_inputs(df, cols=['sequence']):
    data = np.array(df[cols]
                    .applymap(lambda seq: np.array([token2int[x] for x in seq]).tolist())
                    .values
                    .tolist()
                    )

    return data


def get_labels(df):
    labels = np.array(df[pred_cols].applymap(
        lambda seq: [x for x in seq]).values.tolist())
    return labels[:, 0, :]


def get_token2int():
    return token2int


def get_raw_data():
    return pd.read_json('./train.json', lines=True), pd.read_json('./test.json', lines=True)
