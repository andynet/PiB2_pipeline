# %%
import argparse
import torch
import torch.utils.data
from torchvision.utils import save_image
from scripts.VAE3 import VariationalAutoencoder, ELBO
import numpy as np


# %%
def gridwalk_z(z_mean, z_std):
    result = []
    for i in range(z_mean.shape[0]):
        tmp = z_mean.repeat(z_mean.shape[0], 1)
        upper = z_mean[i] + (z_std[i] * 2)
        lower = z_mean[i] - (z_std[i] * 2)
        step = (upper - lower) / (z_mean.shape[0] - 1)
        tmp[:, i] = torch.arange(lower, upper + step / 2, step)
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
def split_dataset(dataset_file, train_size, device):
    res = argparse.Namespace()
    dataset = np.load(dataset_file)

    features, res.min, res.max, res.mean = (min_max_scale(dataset['features'],
                                                          center=True))
    labels = dataset['labels']

    indices = np.random.permutation(range(features.shape[0]))

    res.features_train = features[indices[0:train_size], :]
    res.features_train = (torch.tensor(res.features_train)
                          .to(device=device, dtype=torch.float32))

    res.labels_train = labels[indices[0:train_size], :]
    res.labels_train = (torch.tensor(res.labels_train)
                        .to(device=device, dtype=torch.float32))

    res.features_test = features[indices[train_size:], :]
    res.features_test = (torch.tensor(res.features_test)
                         .to(device=device, dtype=torch.float32))

    res.labels_test = labels[indices[train_size:], :]
    res.labels_test = (torch.tensor(res.labels_test)
                       .to(device=device, dtype=torch.float32))

    return res


# %% parameters
args = argparse.Namespace()
args.batch_size = 256
args.epochs = 15
args.cuda = False
args.seed = 1
args.dataset_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST.npz'

# %%
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
dataset = split_dataset(args.dataset_file, 40000, device)
model = VariationalAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
comparison = []
nrow = args.epochs + 2
message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}'

orig = dataset.features_test.numpy()[:nrow]
orig = min_max_scale_inv(orig, dataset.min, dataset.max, dataset.mean) / 255
orig = torch.tensor(orig).view(-1, 1, 28, 28)
comparison.append(orig)

with torch.no_grad():
    train_loss = model.score(dataset.features_train, ELBO, args.batch_size)
    validation_loss = model.score(dataset.features_test, ELBO, args.batch_size)
    print(message.format(0, round(train_loss, 2), round(validation_loss, 2)))

    pred = model.predict(dataset.features_test).numpy()[:nrow]
    pred = min_max_scale_inv(pred, dataset.min, dataset.max, dataset.mean) / 255
    pred = torch.tensor(pred).view(-1, 1, 28, 28)
    comparison.append(pred)

# %%
for epoch in range(1, args.epochs + 1):
    model.fit(dataset.features_train, ELBO, args.batch_size, optimizer)

    with torch.no_grad():
        train_loss = model.score(dataset.features_train, ELBO, args.batch_size)
        validation_loss = model.score(dataset.features_test, ELBO, args.batch_size)
        print(message.format(epoch, round(train_loss, 2), round(validation_loss, 2)))

        pred = model.predict(dataset.features_test).numpy()[:nrow]
        pred = min_max_scale_inv(pred, dataset.min, dataset.max, dataset.mean) / 255
        pred = torch.tensor(pred).view(-1, 1, 28, 28)
        comparison.append(pred)

# %%
comparison = torch.cat(comparison)

save_image(comparison.cpu(), 'results/reconstruction.png', nrow=nrow)
print('results/reconstruction.png')

# %%
labels = dataset.labels_test.numpy()

z_mean, z_logvar, _, _ = model.forward(dataset.features_test)
z_mean = z_mean.detach().numpy()
z_mean = z_mean[(labels == 3).squeeze()]
z_mean = np.mean(z_mean, axis=0)
z_mean = torch.tensor(z_mean)

z_logvar = z_logvar.detach().numpy()
z_logvar = z_logvar[(labels == 3).squeeze()]
z_var = np.exp(z_logvar)
z_var = np.sum(z_var, axis=0)
z_std = np.sqrt(z_var)
z_std = torch.tensor(z_std)

# %%
z = gridwalk_z(z_mean, z_std).to(device)
x_mean, _ = model.decode(z)
x_mean = min_max_scale_inv(x_mean.detach().numpy(), dataset.min, dataset.max, dataset.mean)
x_mean /= 255
x_mean = torch.tensor(x_mean).view(-1, 1, 28, 28)
filename = f'results/sample.png'
save_image(tensor=x_mean, filename=filename, nrow=20)
