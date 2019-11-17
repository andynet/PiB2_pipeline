#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:11:27 2019

@author: andy
"""

# %%
import numpy as np
import torch
import os

from torchvision import datasets
from scvi.dataset import (BrainLargeDataset, CortexDataset, PbmcDataset,
                          RetinaDataset, HematoDataset, CbmcDataset,
                          BrainSmallDataset, SmfishDataset)

# %%
data_dir = '/home/andy/Projects/pib2/PiB2_pipeline/data'

# %% MNIST
mnist_train = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=True, download=True))
mnist_test = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=False, download=True))

mnist_features = (torch.cat([mnist_train.dataset.data,
                             mnist_test.dataset.data], axis=0)
                  .reshape((70000, -1)).numpy())

mnist_labels = (torch.cat([mnist_train.dataset.targets,
                           mnist_test.dataset.targets], axis=0)
                .reshape((70000, -1)).numpy())

np.savez(f'{data_dir}/MNIST.npz', features=mnist_features, labels=mnist_labels)

# %%
del mnist_train, mnist_test, mnist_features, mnist_labels

# %% how to load data from npz
data = np.load(f'{data_dir}/MNIST.npz')

# %% scvi dataset
os.chdir(data_dir)
brain_small_dataset = BrainSmallDataset(save_path=f'{data_dir}/')

brain_small_features = brain_small_dataset.X.toarray()
brain_small_labels = brain_small_dataset.labels

np.savez(f'{data_dir}/BrainSmall.npz', features=brain_small_features, labels=brain_small_labels)

# %%
del brain_small_dataset, brain_small_features, brain_small_labels

# %% scvi dataset
os.chdir(data_dir)
dataset = CortexDataset(save_path=f'{data_dir}/')

features = dataset.X
labels = dataset.labels

np.savez(f'{data_dir}/Cortex.npz', features=features, labels=labels)
