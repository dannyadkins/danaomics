import torch.nn as nn


class AttnModel(nn.Module):
    def __init__(self, hyperparams):
        super(AttnModel, self).__init__()
        self.only_encoder = hyperparams["only_encoder"]
        self.vocab_size = hyperparams['vocab_size']
        self.embeddings = nn.ModuleList([nn.Embedding(
            self.vocab_size, hyperparams["model_dim"]) for i in range(hyperparams["num_in_features"])])
        encoder_layer = nn.TransformerEncoderLayer(
            hyperparams["model_dim"] * hyperparams["num_in_features"], hyperparams["num_heads"], dropout=hyperparams["dropout"])
        self.encoder = nn.TransformerEncoder(
            encoder_layer, hyperparams["num_encoder_layers"])
        decoder_layer = nn.TransformerDecoderLayer(
            hyperparams["model_dim"] * hyperparams["num_in_features"], hyperparams["num_heads"], dropout=hyperparams["dropout"])
        self.decoder = nn.TransformerDecoder(
            decoder_layer, hyperparams["num_decoder_layers"])

        self.dense = nn.Linear(
            hyperparams["model_dim"] * hyperparams["num_in_features"], 1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, inputs, targets=None):
        embedded_inputs = []
        for i in range(0, inputs.size(1)):
            embedded_inputs.append(self.embeddings[i](inputs[:, i, :]))
        concated_embeddings = torch.cat(embedded_inputs, axis=2)
        predictions = self.encoder(concated_embeddings)
        if (not self.only_encoder):
            predictions = self.decoder(targets, predictions)
        predictions = self.dense(predictions)
        predictions = self.relu(predictions)
        return predictions
