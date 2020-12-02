import torch
import torch.nn as nn


class AttnModel(nn.Module):
    def __init__(self, hyperparams):
        super(AttnModel, self).__init__()
        self.vocab_size = hyperparams["vocab_size"]
        self.embeddings = nn.ModuleList([nn.Embedding(
            self.vocab_size, hyperparams["model_dim"]) for i in range(hyperparams["num_in_features"])])

        encoder_layer = nn.TransformerEncoderLayer(
            hyperparams["model_dim"] * hyperparams["num_in_features"], hyperparams["num_heads"])
        self.encoder = nn.TransformerEncoder(
            encoder_layer, hyperparams["num_encoder_layers"])
        self.dense = nn.Linear(
            hyperparams["model_dim"] * hyperparams["num_in_features"], 1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, inputs):
        embedded_inputs = []
        for i in range(0, inputs.size(1)):
            embedded_inputs.append(self.embeddings[i](inputs[:, i, :]))
        concated_embeddings = torch.cat(embedded_inputs, axis=2)
        predictions = self.encoder(concated_embeddings)
        predictions = self.dense(predictions)
        predictions = self.relu(predictions)
        return predictions
