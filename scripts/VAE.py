# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class VariationalAutoencoder(nn.Module):

    def __init__(self, in_size: int = 784, lat_size: int = 20,
                 hl1_size: int = 400, hl2_size: int = 400,
                 L: int = 3):

        super(VariationalAutoencoder, self).__init__()
        self.L = L

        self.fc1 = nn.Linear(in_size, hl1_size)
        self.fc21 = nn.Linear(hl1_size, lat_size)
        self.fc22 = nn.Linear(hl1_size, lat_size)

        self.fc4 = nn.Linear(lat_size, hl2_size)
        self.fc51 = nn.Linear(hl2_size, in_size)
        self.fc52 = nn.Linear(hl2_size, in_size)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        z_mean = self.fc21(x)
        z_logvar = self.fc22(x)
        return z_mean, z_logvar

    def reparametrize(self, z_mean, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        z = torch.distributions.Normal(z_mean, z_std).rsample([self.L])
        return z

    def decode(self, z):
        z = F.relu(self.fc4(z))
        x_mean = self.fc51(z)
        x_logvar = self.fc52(z)
        return x_mean, x_logvar

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparametrize(z_mean, z_logvar)
        x_mean, x_logvar = self.decode(z)
        return z_mean, z_logvar, x_mean, x_logvar

    def fit(self, x, loss_function, batch_size, optimizer):
        self.train()
        indices = np.random.permutation(range(x.shape[0]))
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[indices[i:i + batch_size], :]
            optimizer.zero_grad()
            z_mean, z_logvar, x_mean, x_logvar = self.forward(x_batch)
            loss = loss_function(z_mean, z_logvar, x_mean, x_logvar, x_batch)
            loss.backward()
            optimizer.step()

    def predict(self, x):
        z_mean, z_logvar = self.encode(x)
        x_mean, x_logvar = self.decode(z_mean)
        return x_mean

    def score(self, x, loss_function, batch_size):
        self.eval()
        loss_total = 0
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i + batch_size, :]
            z_mean, z_logvar, x_mean, x_logvar = self.forward(x_batch)
            loss = loss_function(z_mean, z_logvar, x_mean, x_logvar, x_batch)
            loss_total += loss.item() * x_batch.shape[0]
        return loss_total / x.shape[0]


# %%
def ELBO(z_mean, z_logvar, x_mean, x_logvar, x):
    # From Auto-Encoding Variational Bayes
    KL_div = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(axis=1)
    assert all(KL_div >= 0), print(KL_div)

    x_std = torch.exp(0.5 * x_logvar)
    dist = torch.distributions.Normal(x_mean, x_std)
    exp_NLL = dist.log_prob(x).sum(axis=2).mean(axis=0)

    avg_ELBO = torch.mean(exp_NLL - KL_div)
    return -avg_ELBO


# %%
def min_max_scale(data, center=False):
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)

    denom = np.broadcast_to((max_ - min_), data.shape)
    scaled_data = np.divide((data - min_), denom, where=(~np.isclose(denom, 0)))

    if center:
        mean_ = scaled_data.mean(axis=0)
        scaled_data = scaled_data - mean_
    else:
        mean_ = None

    return scaled_data, min_, max_, mean_


# %%
def min_max_scale_inv(data, min_, max_, mean_):
    if mean_ is not None:
        orig_data = ((data + mean_) * (max_ - min_)) + min_
    else:
        orig_data = (data * (max_ - min_)) + min_

    return orig_data
