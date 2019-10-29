# %%
import argparse
import torch
import torch.utils.data
from torchvision.utils import save_image
from scripts.VAE import VariationalAutoencoder, ELBO
import pandas as pd
from scipy import stats
import numpy as np


# %%
def flatten_params(model):
    params = []
    for p in model.parameters():
        params = params + p.flatten().tolist()

    return params


# %%
def train(epoch, model, dataset, loss_function, batch_size, optimizer):
    model.train()
    train_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(dataloader):
        print(epoch, batch_idx)
        optimizer.zero_grad()
        x_mean, z_mean, z_logvar, log_probs = model(data)
        loss = loss_function(z_mean, z_logvar, log_probs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

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

        features = np.nan_to_num(stats.zscore(features.values, axis=0))
        self.features = torch.tensor(features).to(device=device, dtype=torch.float32)
        self.labels = torch.tensor(labels.values).to(device=device, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]


# %%
# if __name__ == "__main__":
args = argparse.Namespace()
args.batch_size = 64
args.epochs = 15
args.cuda = False
args.seed = 0
args.log_interval = 50

args.labels_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_labels.feather'
args.features_file = '/home/andy/Projects/pib2/PiB2_pipeline/data/MNIST_features.feather'

torch.manual_seed(args.seed)

# %%
data = MyDataset(args.features_file, args.labels_file)
train_set, test_set = torch.utils.data.random_split(data, [60000, 10000])

# %%
# args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# %%
model = VariationalAutoencoder().to(device)
# model.initialize_model(-0.0000001, 0.0000001)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
