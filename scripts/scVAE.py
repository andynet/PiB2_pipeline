#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:49:22 2019

@author: andy
"""

# %%
import argparse
import torch
import torch.utils.data
from scripts.VAE import (VariationalAutoencoder, ELBO,
                         min_max_scale, min_max_scale_inv)
import numpy as np
import seaborn as sns
import pandas as pd


# %%
def load_dataset(dataset_file, device):
    res = argparse.Namespace()
    dataset = np.load(dataset_file)

    features, res.min, res.max, res.mean = (min_max_scale(dataset['features'],
                                                          center=True))
    labels = dataset['labels'].astype('uint8')

    res.features = (torch.tensor(features)
                    .to(device=device, dtype=torch.float32))

    res.labels = (torch.tensor(labels)
                  .to(device=device, dtype=torch.float32))

    return res


# %% parameters
args = argparse.Namespace()
args.batch_size = 256
args.epochs = 100
args.cuda = False
args.seed = 0
args.dataset_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/Cortex.npz'
args.results = '/home/andy/Projects/pib2/PiB2_pipeline/results'

# %%
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
dataset = load_dataset(args.dataset_file, device)
model = VariationalAutoencoder(in_size=558, lat_size=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
message = 'epoch: {}\tloss: {}'

with torch.no_grad():
    loss = model.score(dataset.features, ELBO, args.batch_size)
    print(message.format(0, round(loss, 2)))

# %%
for epoch in range(1, args.epochs + 1):
    model.fit(dataset.features, ELBO, args.batch_size, optimizer)

    with torch.no_grad():
        loss = model.score(dataset.features, ELBO, args.batch_size)
        print(message.format(epoch, round(loss, 2)))

# %%
z_mean, z_logvar, _, _ = model.forward(dataset.features)
z_mean = z_mean.detach().numpy()

c = dataset.labels.numpy()

# %%
df = pd.DataFrame(z_mean, columns=['x', 'y'])
df['class'] = pd.Series(c[:, 0].astype('int8'), dtype="category")

# %%
sns.scatterplot(x="x", y="y", hue="class", data=df, palette="Set2")
