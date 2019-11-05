# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
import torch


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
    
    def predict():
        pass

    def score():
        pass
    
    def fit():
        pass

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
