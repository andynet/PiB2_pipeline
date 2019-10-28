# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):

    def __init__(self, in_size: int = 784, lat_size: int = 20,
                 hl1_size: int = 400, hl2_size: int = 400,
                 L: int = 3):
        """
        Basic Variational Autoencoder assuming uncorrelated multivariate normal
        latent space and uncorrelated multivariate normal posterior.

        Parameters
        ----------
        in_size : int, optional
            Size of the data vector. The default is 784.
        lat_size : int, optional
            Size of the latent representation. The default is 20.
        hl1_size : int, optional
            Size of the hidden layer used in the encoder. The default is 400.
        hl2_size : int, optional
            Size of the hidden layer used in the decoder. The default is 400.
        L : int, optional
            The number of MCMC samples from latent space. The default is 3.

        Returns
        -------
        None.
        """

        super(VariationalAutoencoder, self).__init__()
        self.L = L

        self.fc1 = nn.Linear(in_size, hl1_size)
        self.fc21 = nn.Linear(hl1_size, lat_size)
        self.fc22 = nn.Linear(hl1_size, lat_size)
        self.fc4 = nn.Linear(lat_size, hl2_size)
        self.fc51 = nn.Linear(hl2_size, in_size)
        self.fc52 = nn.Linear(hl2_size, in_size)

    def encode(self, x):
        """
        Encodes the datapoints to their latent distributions.

        Parameters
        ----------
        x : torch.Tensor of size [batch_size, in_size]
            Input data with one datapoint represented in one row.

        Returns
        -------
        z_mean : torch.Tensor of size [batch_size, lat_size]
            Means of the latent distributions.
        z_logvar : torch.Tensor of size [batch_size, lat_size]
            Logarithm of the variation of the latent distributions.

        """
        x = F.relu(self.fc1(x))
        z_mean = self.fc21(x)
        z_logvar = self.fc22(x)
        return z_mean, z_logvar

    def reparametrize(self, z_mean, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        z = torch.distributions.Normal(z_mean, z_std).rsample([self.L])
        return z

    def decode(self, z, x):
        """
        Decodes the latent space MC sample to distibution of x and 
        calculates the log_prob of x.

        Parameters
        ----------
        z : torch.Tensor with size [L, batch_size, lat_size]
            MC sample from latent space.
        x : torch.Tensor with size [batch_size, in_size]
            input, which we expect to reconstruct

        Returns
        -------
        x_mean : torch.Tensor with size [L, batch_size, in_size]
            mean of the posterior distribution.
        log_prob_x_given_z : torch.Tensor of size [L, batch_size]
            reconstruction log_prob of each datapoint.
        """
        z = F.relu(self.fc4(z))

        x_mean = self.fc51(z)
        x_logvar = self.fc52(z)
        x_std = torch.exp(0.5 * x_logvar)

        log_prob_x_given_z = torch.distributions.Normal(x_mean, x_std).log_prob(x).sum(axis=2)

        return x_mean, log_prob_x_given_z

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparametrize(z_mean, z_logvar)
        x_mean, log_prob_x_given_z = self.decode(z, x)
        return x_mean, z_mean, z_logvar, log_prob_x_given_z


# %%
def ELBO(z_mean, z_logvar, log_probs):
    # From Auto-Encoding Variational Bayes
    D_KL = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(axis=1)
    assert all(D_KL >= 0)

    expected_recon_error_est = torch.mean(log_probs, axis=0)

    avg_ELBO = torch.mean(expected_recon_error_est - D_KL)
    return -avg_ELBO
