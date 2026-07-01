import torch
import torch.nn as nn


class ClientEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B * T, F)

        z = self.net(x)
        z = z.reshape(B, T, -1)

        return z


class SharedLSTMModel(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.rul_head = nn.Linear(hidden_dim, 1)
        self.fault_head = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        out, _ = self.lstm(z)
        h = out[:, -1, :]

        rul = self.rul_head(h)
        fault = torch.sigmoid(self.fault_head(h))

        return rul, fault