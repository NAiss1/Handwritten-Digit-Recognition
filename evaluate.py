import torch
from model import DigitClassifier
from data_loader import get_loaders

_, test_loader = get_loaders()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitClassifier().to(device)
model.load_state_dict(torch.load('saved_model/mnist_model.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
