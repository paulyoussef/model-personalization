import torch
from torch import nn
import numpy as np


# class MLP(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim=32, output_dim=1):
#         super().__init__()
#         self.hidden1 = nn.Linear(input_dim, hidden_dim)
#         self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
#         self.hidden3 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = nn.functional.relu(self.hidden1(x))
#         x = nn.functional.relu(self.hidden2(x))
#         x = self.hidden3(x)
#         return x



class Autoencoder(nn.Module):
    def __init__(self, num_words, embedding_dim, oh_features):
        super(Autoencoder, self).__init__()
        self.oh_features = oh_features
        self.embedding = nn.Embedding(num_words, embedding_dim, padding_idx=0)
        input_dim = oh_features + embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), nn.Linear(128, input_dim), nn.Tanh())

    def embed_input(self, input):
        if len(np.array(input).shape) == 1:
            input = torch.Tensor(input).to(list(self.parameters())[0].device)
            x = input[:self.oh_features]
            x_to_embed = input[self.oh_features:]
            embedded_features = torch.mean(self.embedding(x_to_embed.long()), dim=-2)
            x = torch.cat((x, embedded_features), 0)
            return x.unsqueeze(0)

        x = input[:, :self.oh_features]
        x_to_embed = input[:, self.oh_features:]
        embedded_features = torch.mean(self.embedding(x_to_embed.long()), dim=-2)
        x = torch.cat((x, embedded_features), 1)
        return x

    def forward(self, input):
        x = self.embed_input(input)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, input):
        x = self.embed_input(input)
        x = self.encoder(x)
        return x


class OHAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(OHAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), nn.Linear(128, input_dim), nn.Tanh())

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x

    def encode(self, input):
        x = self.encoder(input)
        return x


class APGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, output_fn=None):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)
        self.output_fn = output_fn

    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.output_fn(self.hidden3(x))
        return x


def amp_fn(x):
    return torch.sigmoid(x) * 4.9 + 0.1


def phase_fn(x):
    return torch.sigmoid(x) * np.pi


class MLPAnil(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.features(x)
        return self.head(features)

class Features(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=64, output_dim=3, static_input_dim=16, static_output_dim=4):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_dim
        self.device = device
        self.static_input_dim = static_input_dim
        self.static_output_dim = static_output_dim
        self.lstm = nn.LSTM(input_dim - static_input_dim, hidden_dim)

        # self.dropout1 = nn.Dropout(p=0.1)

        # contains the hidden state, and the hidden cell
        # both have the shape: (num_layers, batch_size, ...)
        # hidden state: H_out projection size if it is > 0 otherwise hidden_size
        # hidden cell: H_cell hidden size
        self.hidden = (torch.zeros(1, 1, self.hidden_layer_size, device=device),
                       torch.zeros(1, 1, self.hidden_layer_size, device=device))


        # MLP for encoding static part
        if self.static_input_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_input_dim, 32),
                # nn.Dropout(p=0.1),
                nn.GELU(),
                nn.Linear(32, self.static_output_dim),
                # nn.Dropout(p=0.1),
                nn.GELU())

    def forward(self, input_seq):
        # inp = torch.randn(batch_size, seq_len, input_dim)
        # hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        # cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        # hidden = (hidden_state, cell_state)

        if len(input_seq.shape) <= 2:
            batch_size = 1
        else:
            batch_size = input_seq.shape[0]

        input_seq = input_seq.view(-1, batch_size, self.input_dim)

        if self.static_input_dim > 0:
            static, dynamic = input_seq[:, :, -self.static_input_dim:], input_seq[:, :,
                                                                        :self.input_dim - self.static_input_dim],
            assert (static.shape[2] == self.static_input_dim)

            # Input should have the shape: (seq_len, batch_size, input_size)
            lstm_out, self.hidden = self.lstm(dynamic, self.hidden)

            # Output has the shape (seq_len, batch_size, output_size)

            # self.lstm return a tuple x
            # x[0] is a tensor of shape (1, 1, hidden_layer_size)
            # x[1] is a tuple that contains two tensors of the same shape (1, 1, hidden_layer_size)
            # lstm_out = self.dropout1(lstm_out).view(batch_size, -1, self.hidden_layer_size)
            lstm_out = lstm_out.view(batch_size, -1, self.hidden_layer_size)

            fnn_out = self.static_mlp(static.view(batch_size, -1, self.static_input_dim))

            # combining both
            s_d = torch.cat((lstm_out, fnn_out), 2)
            return s_d
        else:
            lstm_out, self.hidden = self.lstm(input_seq, self.hidden)
            #lstm_out = self.dropout1(lstm_out).view(batch_size, -1, self.hidden_layer_size)
            lstm_out = lstm_out.view(batch_size, -1, self.hidden_layer_size)

            return lstm_out

    def reset_hidden_state(self, batch_size=1):
        self.hidden = (torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device),
                       torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device))

class LSTMGeneric(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=64, output_dim=3, static_input_dim=16, static_output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_dim
        self.device = device
        self.static_input_dim = static_input_dim
        self.static_output_dim = static_output_dim
        # self.head_dropout = nn.Dropout(p=0.1)

        self.features = Features(device = self.device, input_dim = self.input_dim, output_dim = self.output_dim, hidden_dim = self.hidden_layer_size, static_input_dim = self.static_input_dim, static_output_dim= self.static_output_dim)
        self.features.to(self.device)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.static_output_dim, hidden_dim // 2),
            nn.GELU(),
            # self.head_dropout,
            nn.Linear(hidden_dim // 2, output_dim)
            )

    def forward(self, input_seq):
        s_d = self.features(input_seq)

        return self.head(s_d)

    def reset_hidden_state(self, batch_size=1):
        self.features.reset_hidden_state(batch_size)