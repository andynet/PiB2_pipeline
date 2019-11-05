# %%
import argparse
import torch
import torch.utils.data
from torchvision.utils import save_image
from scripts.VAE2 import VariationalAutoencoder, ELBO
import pandas as pd
import numpy as np


# %%
def gridwalk_z(z_size, lower=-2, upper=2):

    result = []
    for i in range(z_size):
        tmp = torch.zeros((z_size, z_size))
        step = (upper - lower) / (z_size - 1)
        tmp[:, i] = torch.arange(lower, upper + step, step)
        result.append(tmp)
    result = torch.cat(result)
    return result


# %%
def min_max_scale(data, center=False):
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)

    scaled_data = np.divide(data - min_, max_ - min_, where=(max_ - min_ != 0))

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


# %%
def train(epoch, model, dataset, loss_function, batch_size, optimizer):
    model.train()
    train_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        z_mean, z_logvar, x_mean, x_logvar = model(data)
        loss = loss_function(z_mean, z_logvar, x_mean, x_logvar, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


# %%
def validate(epoch, model, dataset, loss_function, n_epochs):
    model.eval()
    validation_loss = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    stats = vars(dataset.dataset)

    for idx, (data, target) in enumerate(dataloader):

        z_mean, z_logvar, x_mean, x_logvar = model(data)
        loss = loss_function(z_mean, z_logvar, x_mean, x_logvar, data)
        validation_loss += loss.item()
        x_mean = torch.mean(x_mean, axis=0)

        if idx == 0:

            n = min(data.size(0), n_epochs)
            orig = min_max_scale_inv(data.data.numpy(), stats['min'], stats['max'], stats['mean'])
            orig /= 255
            orig = torch.tensor(orig).view(-1, 1, 28, 28)[:n]

            pred = min_max_scale_inv(x_mean.numpy(), stats['min'], stats['max'], stats['mean'])
            pred /= 255
            pred = torch.tensor(pred).view(-1, 1, 28, 28)[:n]

    return validation_loss, orig, pred


# %%
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels, device='cpu'):
        features = pd.read_feather(features)
        labels = pd.read_feather(labels)

        assert features.shape[0] == labels.shape[0]

        features, self.min, self.max, self.mean = min_max_scale(features.values, center=True)
        self.features = torch.tensor(features).to(device=device, dtype=torch.float32)
        self.labels = torch.tensor(labels.values).to(device=device, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]


# %% parameters
args = argparse.Namespace()
args.batch_size = 256
args.epochs = 15
args.cuda = False
args.seed = 0

args.labels_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_labels.feather'
args.features_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_features.feather'

# %%
torch.manual_seed(args.seed)

data = MyDataset(args.features_file, args.labels_file)
train_set, test_set = torch.utils.data.random_split(data, [60000, 10000])

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

model = VariationalAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

comparison = []
nrow = args.epochs + 2

# %%
with torch.no_grad():
    validation_loss, orig, pred = validate(epoch=0, model=model,
                                           dataset=test_set,
                                           loss_function=ELBO,
                                           n_epochs=nrow)
comparison.append(orig)
comparison.append(pred)
print(f'Epoch: 0\tValidation_loss: {round(validation_loss)}')

# %%
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch=epoch, model=model, dataset=train_set,
                       loss_function=ELBO, batch_size=args.batch_size,
                       optimizer=optimizer)

    with torch.no_grad():
        validation_loss, orig, pred = validate(epoch=epoch, model=model,
                                               dataset=test_set, loss_function=ELBO,
                                               n_epochs=nrow)
        comparison.append(pred)

        stats = vars(test_set.dataset)
        z = gridwalk_z(20).to(device)
        x_mean, _ = model.decode(z)
        # x_mean = torch.mean(x_mean, axis=0)
        x_mean = min_max_scale_inv(x_mean.numpy(), stats['min'], stats['max'], stats['mean'])
        x_mean /= 255
        x_mean = torch.tensor(x_mean).view(-1, 1, 28, 28)
        # save_image(x_mean, 'results/sample_' + str(epoch) + '.png')
        filename = f'results/sample_{epoch}.png'
        save_image(tensor=x_mean, filename=filename, nrow=20)

    print(f'Epoch: {epoch}\tTrain_loss: {round(train_loss, 2)}\tValidation_loss: {round(validation_loss, 2)}')

# %%
comparison = torch.cat(comparison)

save_image(comparison.cpu(), 'results/reconstruction.png', nrow=nrow)
print('results/reconstruction.png')
