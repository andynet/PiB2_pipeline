# %%
import argparse
import torch
import torch.utils.data
from torchvision.utils import save_image
from scripts.VAE import VariationalAutoencoder, ELBO
import pandas as pd
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
# %%
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

# %%
def flatten_params(model):
    params = np.array([])
    for p in model.parameters():
        params = np.concatenate([params, p.flatten().detach()])

    return params


# %%
def train(epoch, model, dataset, loss_function, batch_size, optimizer):
    model.train()
    train_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(dataloader):
        # print(epoch, batch_idx)
        optimizer.zero_grad()
        x_mean, z_mean, z_logvar, log_probs = model(data)
        loss = loss_function(z_mean, z_logvar, log_probs)
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        optimizer.step()
        train_loss += loss.item()

        # assert all(np.isfinite(flatten_params(model)))
    return train_loss


# %%
def validate(epoch, model, dataset, loss_function):
    model.eval()
    validation_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            x_mean, z_mean, z_logvar, log_probs = model(data)
            loss = loss_function(z_mean, z_logvar, log_probs)
            validation_loss += loss.item()
            x_mean = torch.mean(x_mean, axis=0)

            if idx == 0:
                n = min(data.size(0), 8)
                data = data / 255
                x_mean = x_mean / 255
                comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                                        x_mean.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

                print('results/reconstruction_' + str(epoch) + '.png')

    return validation_loss


# %%
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels, device='cpu'):
        features = pd.read_feather(features)
        labels = pd.read_feather(labels)

        assert features.shape[0] == labels.shape[0]

        self.means = features.mean(axis=0).values
        self.stds = features.std(axis=0).values

        features = np.divide(features.values - self.means, self.stds, where=(self.stds != 0))
        # features = np.nan_to_num(stats.zscore(features.values, axis=0))
        self.features = torch.tensor(features).to(device=device, dtype=torch.float32)
        self.labels = torch.tensor(labels.values).to(device=device, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]


# %%
# if __name__ == "__main__":
args = argparse.Namespace()
args.batch_size = 256
args.epochs = 15
args.cuda = False
args.seed = 1
args.log_interval = 50

args.labels_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_labels.feather'
args.features_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_features.feather'

torch.manual_seed(args.seed)

# %%
data = MyDataset(args.features_file, args.labels_file)
train_set, test_set = torch.utils.data.random_split(data, [60000, 10000])

# %%
train_set.dataset

# %%
# args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# %%
model = VariationalAutoencoder().to(device)
# model.initialize_model(-0.0000001, 0.0000001)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# %%
validation_loss = validate(epoch=0, model=model, dataset=test_set, loss_function=ELBO)
print(f'Epoch: 0\tValidation_loss: {validation_loss}')
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch=epoch, model=model, dataset=train_set,
                       loss_function=ELBO, batch_size=args.batch_size,
                       optimizer=optimizer)
    print(f'Epoch: {epoch}\tTrain_loss: {train_loss}')
    validation_loss = validate(epoch=epoch, model=model,
                               dataset=test_set, loss_function=ELBO)
    print(f'Epoch: {epoch}\tValidation_loss: {validation_loss}')
    with torch.no_grad():
        z = torch.randn(3, 64, 20).to(device)
        x = torch.randn(3, 64, 784).to(device)
        x_mean, _ = model.decode(z, x)
        x_mean = torch.mean(x_mean, axis=0)
        save_image(x_mean.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
