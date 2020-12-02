import os
from torch.utils.data import DataLoader
import json
import glob
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
import argparse

from preprocess import get_raw_data, get_labels, get_tokenized_inputs, get_token2int
from models.AttnModel import AttnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token2int = get_token2int()
in_features = ['sequence', 'structure', "predicted_loop_type"]
hyperparams = {
    "batch_size": 100,
    "num_epochs": 100,
    "learning_rate": 0.002,
    "model_dim": 128,
    "embedding_size": 128,
    "num_heads": 1,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "seq_len": 107,
    "dropout": 0.3,
    "num_in_features": len(in_features),
    "only_encoder": True,
    "vocab_size": len(token2int.items())
}


def prepare_model():
    raw_train, raw_test = get_raw_data()
    tokenized_inputs = get_tokenized_inputs(raw_train, cols=in_features)

    labels = get_labels(raw_train)

    split_index_inputs = int(0.9*tokenized_inputs.shape[0])
    split_index_labels = int(0.9*labels.shape[0])
    train_inputs = tokenized_inputs[:split_index_inputs]
    train_labels = labels[:split_index_labels]

    test_inputs = tokenized_inputs[split_index_inputs:]
    test_labels = labels[split_index_labels:]

    train_loader = DataLoader(
        list(zip(train_inputs, train_labels)),  batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(list(zip(test_inputs, test_labels)),
                             batch_size=hyperparams["batch_size"])

    model = AttnModel(hyperparams).to(device)

    return model, train_loader, test_loader


def train(model, loader, hyperparams):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["learning_rate"])
    loss = nn.MSELoss()

    with tqdm(total=hyperparams["num_epochs"]) as epochbar:
        for epoch in range(0, hyperparams["num_epochs"]):
            total_loss = 0
            i = 0
            for (inputs, labels) in loader:

                target = torch.zeros(inputs[:, 0, :].shape)
                target[::, :labels.size(1)] = labels
                inputs = inputs.to(device)
                target = target.to(device)
#                 predictions = torch.FloatTensor(inputs.shape).uniform_(0, 1).to(device)
                predictions = model(inputs)
                l = loss(predictions[::, :68].reshape(-1).float(),
                         target[::, :68].reshape(-1).float())
                total_loss += l.detach().cpu().numpy()
                i += 1
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                desc = f'Epoch {epoch}, loss {total_loss/i}, batch {i}'
                epochbar.set_description(desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")

    args = parser.parse_args()

    model, train_loader, test_loader = prepare_model()

    if args.load:
        print("Loading saved model...")
        model.load_state_dict(torch.load("./model.pt"))
    if args.train:
        print("Running training loop...")
        train(model, train_loader, hyperparams)
    if args.test:
        print("Running testing loop...")
        test(model, test_loader, hyperparams)
    if args.save:
        print("Saving model...")
        torch.save(model.state_dict(), "./model.pt")
