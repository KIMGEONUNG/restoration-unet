from models import UNet, UNetSmall
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from skimage import color
from torch.utils.tensorboard import SummaryWriter


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str,
            default='./datasets/')
    parser.add_argument('--size_batch', type=int,
            default=8)
    parser.add_argument('--num_epoch', type=int,
            default=2)
    parser.add_argument('--dev', type=str,
            default='cuda')
    parser.add_argument('--interval_loss', type=int,
            default=5)
    parser.add_argument('--interval_img', type=int,
            default=5)

    return parser.parse_args()


def lab2rgb(x):
    # x = post_lab(x)
    x = x.permute(0, 2, 3, 1)
    x = color.lab2rgb(x.cpu().detach())
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    return x


def rgb2lab(x):
    x = x.permute(0, 2, 3, 1)
    x = color.rgb2lab(x)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    # x = prep_lab(x)
    return x


def prep_lab(x):
    x[:, :1, :, :] /= 100
    x[:, 1:, :, :] /= 128
    return x


def post_lab(x):
    x[:, :1, :, :] *= 100
    x[:, 1:, :, :] *= 128
    return x


def loss_hist(x1, x2):
    pass


def main(args):
    print(args)

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        ])

    dataset = ImageFolder(args.path_dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.size_batch,
            shuffle=True, num_workers=8)

    # Model
    model = UNet(1, 2).to(args.dev)

    # Loss
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Logger
    writer = SummaryWriter('runs/colorize')

    for epoch in range(args.num_epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):

            x = rgb2lab(x)

            y = x[:, 1:, :, :].to(args.dev)
            x = x[:, :1, :, :].to(args.dev)

            y_hat = model(x)
            loss = loss_fn(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.interval_loss == 0:
                writer.add_scalar('loss', loss.item(), i)

            if i % args.interval_img == 0:
                gt = lab2rgb(torch.cat([x, y], dim=1))
                hat = lab2rgb(torch.cat([x, y_hat], dim=1))

                grid = torch.cat([gt[:4], hat[:4]], dim=0)
                grid = make_grid(grid, nrow=4)
                writer.add_image('qualitative', grid, i)
                writer.flush()


if __name__ == '__main__':
    args = parse()
    main(args)
