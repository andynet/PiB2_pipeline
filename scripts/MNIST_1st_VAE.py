# %%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scripts.VAE import VariationalAutoencoder, ELBO


# %%
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# %%
model = VariationalAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# %%
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        x_mean, z_mean, z_logvar, log_probs = model(data)
        loss = ELBO(z_mean, z_logvar, log_probs)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# %%
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, 784)
            x_mean, z_mean, z_logvar, log_probs = model(data)
            test_loss += ELBO(z_mean, z_logvar, log_probs).item()
            data = data.view(-1, 1, 28, 28)
            x_mean = torch.mean(x_mean, axis=0)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        x_mean.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# %%
if __name__ == "__main__":
    test(0)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            z = torch.randn(3, 64, 20).to(device)
            x = torch.randn(3, 64, 784).to(device)
            x_mean, _ = model.decode(z, x)
            x_mean = torch.mean(x_mean, axis=0)
            save_image(x_mean.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
