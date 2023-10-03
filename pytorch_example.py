from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

print("Loading MNIST dataset without normalization...")
trainset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
