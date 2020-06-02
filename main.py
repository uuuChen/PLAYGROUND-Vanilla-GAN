import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-bs', type=int, default=64)
parser.add_argument('--epochs', '-ep', type=int, default=1024)
parser.add_argument('--loss-log-freq', '-llf', type=int, default=100)
parser.add_argument('--save-dir-path', '-sdp', type=str, default='./results')
parser.add_argument('--save-fig-freq', '-sff', type=int, default=1)

parser.add_argument('--generator-learning-rate', '-glr', type=float, default=5e-4)
parser.add_argument('--generator-embedded-size', '-ges', type=int, default=128)
parser.add_argument('--generator-hidden-size', '-ghs', type=int, default=1024)
parser.add_argument('--generator-output-size', '-gos', type=int, default=784)

parser.add_argument('--discriminator-hidden-size', '-dhs', type=int, default=256)
parser.add_argument('--discriminator-learning-rate', '-dlr', type=float, default=5e-4)

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get data loader
transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_mnist = MNIST("data", download=True, train=True, transform=transforms_)
test_mnist = MNIST("data", download=True, train=False, transform=transforms_)
train_loader = DataLoader(train_mnist, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(test_mnist, batch_size=args.batch_size, shuffle=False)


# Get Generator model and Discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.sigmoid(self.fc3(x))
        return x


generator = Generator(args.generator_embedded_size, args.generator_hidden_size, args.generator_output_size).to(args.device)
discriminator = Discriminator(args.generator_output_size, args.discriminator_hidden_size).to(args.device)


# Get optimizers, note that "betas" is an important setting
g_optimizer = optim.Adam(generator.parameters(), lr=args.generator_learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate, betas=(0.5, 0.999))


# Get criterion
criterion = nn.BCELoss()


# Check directories exist
os.makedirs(args.save_dir_path, exist_ok=True)


def train():
    for epoch in range(1, args.epochs+1):
        d_losses = AverageMeter()
        g_losses = AverageMeter()
        for batch_index, (input, _) in enumerate(train_loader, start=1):
            # train two models
            d_loss = train_discriminator(input)
            g_loss = train_generator(input)

            # print statistics
            d_losses.update(d_loss, count=input.shape[0])
            g_losses.update(g_loss, count=input.shape[0])

            if batch_index % args.loss_log_freq == 0:
                print(f'epoch: [{epoch}/{args.epochs}][{batch_index}/{len(train_loader)}]\td_loss: {d_losses.mean:.3f}\tg_loss: {g_losses.mean:.3f}')

        if epoch % args.save_fig_freq == 0:
            show_fake_images(epoch, 16)


def train_discriminator(input):
    d_real_input = Variable(input.view(input.shape[0], -1)).to(args.device)
    ones_var = Variable(torch.ones(input.shape[0], 1)).to(args.device)
    zeros_var = Variable(torch.zeros(input.shape[0], 1)).to(args.device)

    generator.eval()
    discriminator.train()
    d_optimizer.zero_grad()

    # train on real data
    d_real_decision = discriminator(d_real_input)
    d_real_loss = criterion(d_real_decision, ones_var)
    d_real_loss.backward()

    # train on fake data
    d_fake_seed = get_gaussian_noise(input.shape[0], args.generator_embedded_size)
    d_fake_input = generator(d_fake_seed).detach()
    d_fake_decision = discriminator(d_fake_input)
    d_fake_loss = criterion(d_fake_decision, zeros_var)
    d_fake_loss.backward()

    # optimize weights
    d_optimizer.step()

    return (d_real_loss.item() + d_fake_loss.item()) / 2


def train_generator(input):
    ones_var = Variable(torch.ones(input.shape[0], 1)).to(args.device)

    generator.train()
    discriminator.eval()
    g_optimizer.zero_grad()

    # train on fake data
    d_fake_seed = get_gaussian_noise(input.shape[0], args.generator_embedded_size)
    d_fake_input = generator(d_fake_seed)
    d_fake_decision = discriminator(d_fake_input)
    g_loss = criterion(d_fake_decision, ones_var)
    g_loss.backward()

    # optimize weights
    g_optimizer.step()

    return g_loss


def show_fake_images(epoch, show_count=1):
    g_images_tensor = generator(get_gaussian_noise(show_count, args.generator_embedded_size)).detach().cpu().numpy()
    g_images_tensor = g_images_tensor.reshape(-1, 28, 28)
    sqrtn = int(np.ceil(np.sqrt(g_images_tensor.shape[0])))
    for index, image in enumerate(g_images_tensor):
        # transforms.ToPILImage()(image.cpu()).save(f'{args.save_dir_path}/{epoch}_{i}.png')
        plt.subplot(sqrtn, sqrtn, index + 1)
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def get_gaussian_noise(batch_size, hidden_size):
    return Variable((torch.randn(batch_size, hidden_size))).to(args.device)


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.mean = 0

    def update(self, value, count=1):
        self.sum += value * count
        self.count += count
        self.mean = self.sum / self.count


def main():
    train()


if __name__ == '__main__':
    main()
