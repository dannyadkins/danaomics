from comet_ml import Experiment

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
from sklearn import metrics

from preprocess import get_raw_data, get_labels, get_tokenized_inputs, get_token2int
from models.AttnModel import AttnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token2int = get_token2int()
in_features = ['sequence', 'structure', "predicted_loop_type"]
hyperparams = {
    "batch_size": 400,
    "num_epochs": 10,
    "learning_rate": 0.0005,
    "model_dim": 128,
    "embedding_size": 128,
    "num_heads": 1,
    "num_encoder_layers": 3,
    "num_decoder_layers": 2,
    "seq_len": 107,
    "dropout": 0.1,
    "num_in_features": len(in_features),
    "only_encoder": True,
    "vocab_size": len(token2int.items())
}

possible_params = {
    "learning_rate": [0.00005],
    "model_dim": [64, 512],
    "embedding_size": [64, 512],
    "num_encoder_layers": [3, 6],
    "dropout": [0.1],
}


def build_hyperparam_grid(base_hyperparams):
    grid = []
    for learning_rate in possible_params['learning_rate']:
        for model_dim in possible_params['model_dim']:
            for embedding_size in possible_params['embedding_size']:
                for num_encoder_layers in possible_params['num_encoder_layers']:
                    for dropout in possible_params['dropout']:

                        modified_hyperparams = base_hyperparams.copy()
                        modified_hyperparams['learning_rate'] = learning_rate
                        modified_hyperparams['model_dim'] = model_dim
                        modified_hyperparams['embedding_size'] = embedding_size
                        modified_hyperparams['num_encoder_layers'] = num_encoder_layers
                        modified_hyperparams['dropout'] = dropout

                        grid.append(modified_hyperparams)

    return grid


def prepare_model():
    print("Preparing model...")
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


def train(model, loader, hyperparams, experiment):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["learning_rate"])
    loss = nn.MSELoss()

    with experiment.train():
        with tqdm(total=hyperparams["num_epochs"]) as epochbar:
            for epoch in range(0, hyperparams["num_epochs"]):
                total_loss = 0
                i = 0
                for (inputs, labels) in loader:
                    target = torch.zeros(inputs[:, 0, :].shape)
                    target[::, :labels.size(1)] = labels
                    inputs = inputs.to(device)
                    target = target.to(device)
                    predictions = model(inputs)
                    l = loss(predictions[::, :68].reshape(-1).float(),
                             target[::, :68].reshape(-1).float())
                    total_loss += l.detach().cpu().numpy()
                    i += 1
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    desc = f'Epoch {epoch}, loss {total_loss/i}, batch {i}/{len(loader)}'
                    epochbar.set_description(desc)
                epochbar.update(1)


def test_metrics(seq1, seq2, cutoff=0.7):
    true_labels = np.where(seq2 > cutoff, 1, 0)
    rocauc = metrics.roc_auc_score(true_labels, seq1)
    pr = metrics.precision_recall_curve(true_labels, seq1)
    arg1 = pr[0].argsort()
    prauc = metrics.auc(pr[0][arg1], pr[1][arg1])
    return prauc, rocauc


def test(model, loader, hyperparams, experiment):
    loss = nn.MSELoss()
    i = 0
    total_loss = 0
    total_dist = 0
    total_rocauc = 0
    total_prauc = 0
    with experiment.test():
        for j in range(0, 1):
            for (inputs, labels) in loader:
                target = torch.zeros(inputs[:, 0, :].shape)
                target[::, :labels.size(1)] = labels
                inputs = inputs.to(device)
                target = target.to(device)

                predictions = model(inputs)

                l = loss(predictions[::, :68].reshape(-1).float(),
                         target[::, :68].reshape(-1).float())
                total_loss += l.detach().cpu().numpy()
                i += 1
                prauc, rocauc = test_metrics(predictions[::, :68].reshape(-1).detach(
                ).cpu().numpy(), target[::, :68].reshape(-1).detach().cpu().numpy())
                total_prauc += prauc
                total_rocauc += rocauc
    print("PRAUC: ", total_prauc/i)
    print("ROCAUC: ", total_rocauc/i)
    print("Loss: ", total_loss/i)
    experiment.log_metric("ROCAUC", total_rocauc/i)
    experiment.log_metric("test_loss", total_loss/i)


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
    parser.add_argument("-H", "--hyperparamsearch", action="store_true",
                        help="run hyperparameter search")

    args = parser.parse_args()

    if args.hyperparamsearch:
        print("Running hyperparameter search")

        grid = build_hyperparam_grid(hyperparams)
    else:
        grid = [hyperparams]

    for hyperparam_dict in grid:
        model, train_loader, test_loader = prepare_model()

        print("Running an experiment with hyperparams: ", hyperparam_dict)
        experiment = Experiment(project_name="danaomics")

        experiment.log_parameters(hyperparam_dict)

        if args.load:
            print("Loading saved model...")
            model.load_state_dict(torch.load("./model.pt"))
        if args.train:
            print("Running training loop...")
            train(model, train_loader, hyperparam_dict, experiment)
        if args.test:
            print("Running testing loop...")
            test(model, test_loader, hyperparam_dict, experiment)
        if args.save:
            print("Saving model...")
            torch.save(model.state_dict(), "./model.pt")
