import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

wandb.int(project="mnist.pytorch", entity="jto")

print("Loading MNIST dataset without normalization...")
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=1000, shuffle=False)
