from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=64):
    transform = transforms.ToTensor()

    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader
