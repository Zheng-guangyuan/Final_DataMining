import torch
from torch import nn
from collections import OrderedDict


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(input_size,  input_size)
        self.key_layer = nn.Linear(input_size,  input_size)
        self.value_layer = nn.Linear(input_size,  input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        scores = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = self.softmax(scores)
        attended_value = torch.matmul(attn_weights, value)

        return attended_value


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, sparse_reg=1e-5):
        super(AutoEncoder, self).__init__()

        self.attention = Attention(hidden_size[0])

        # Encoder
        encoder_layers = OrderedDict()
        for i in range(len(hidden_size) - 1):
            encoder_layers[f'enc_linear{i}'] = nn.Linear(hidden_size[i], hidden_size[i + 1])
            encoder_layers[f'enc_drop{i}'] = nn.Dropout(dropout)
            encoder_layers[f'enc_relu{i}'] = nn.ReLU()
        self.encoder = nn.Sequential(encoder_layers)

        # Decoder
        decoder_layers = OrderedDict()
        for i in range(len(hidden_size) - 1, 0, -1):
            decoder_layers[f'dec_linear{i}'] = nn.Linear(hidden_size[i], hidden_size[i - 1])
            decoder_layers[f'dec_relu{i}'] = nn.Sigmoid()
        self.decoder = nn.Sequential(decoder_layers)

        # L1 regularization weight
        self.sparse_reg = sparse_reg

    def forward(self, x):
        # Normalize input to the range [0, 1]
        x = (x - 1) / 4.0

        # Apply attention
        x = self.attention(x)

        # Pass through encoder
        encoded = self.encoder(x)

        # Pass through decoder
        decoded = self.decoder(encoded)

        # Rescale to original range [1, 5]
        decoded = decoded * 4.0 + 1

        return decoded, encoded

    def l1_penalty(self, encoded):
        # L1 penalty for sparse regularization
        return self.sparse_reg * torch.sum(torch.abs(encoded))
