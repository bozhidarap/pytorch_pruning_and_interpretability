# Installation of needed libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
import random
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])
normalizer = transforms.Normalize((0.49139968, 0.48215827, 0.44653124), 
                                   (0.24703233, 0.24348505, 0.26158768))
# Load CIFAR10 dataset
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
test_adv_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Initialize the model
model = resnet18()
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
model.fc = nn.Linear(512, 10)

# Set up optimizer, loss function and device
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def pqd_attacks(model, X, y, epsilon=8/255, alpha=0.01, num_iter=7, restarts=1, normalizer=normalizer):
    """
    Perform Projected Gradient Descent (PGD) attacks.
    
    Args:
        model: The neural network model.
        X: Input data.
        y: True labels.
        epsilon: Maximum perturbation.
        alpha: Step size for each iteration.
        num_iter: Number of iterations.
        restarts: Number of restarts for the attack.
        normalizer: Normalization function.
        
    Returns:
        max_delta: Maximum perturbation found.
    """
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_delta = torch.zeros_like(X).to(X.device)

    for _ in range(restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon

        for _ in range(num_iter):
            loss = nn.CrossEntropyLoss(reduction='none')(model(normalizer(torch.clamp(X + delta, 0, 1))), y)
            loss_b = loss.mean()
            loss_b.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)

            delta.grad.zero_()
            max_delta[loss >= max_loss] = delta.detach()[loss >= max_loss]
            max_loss = torch.max(max_loss, loss)

    return max_delta

def train(model, device, train_loader, optimizer, epoch, normalizer=normalizer):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model.
        device: Device to use for training (CPU or GPU).
        train_loader: DataLoader for training data.
        optimizer: Optimizer for the model.
        epoch: Current epoch number.
        normalizer: Normalization function.
        
    Returns:
        Average loss over the epoch.
    """
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = normalizer(data)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.2f}%)]\tLoss: {running_loss/(batch_idx+1):.6f}')

    return running_loss / (batch_idx + 1)

def test(model, device, test_loader, normalizer=normalizer):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The neural network model.
        device: Device to use for evaluation (CPU or GPU).
        test_loader: DataLoader for test data.
        normalizer: Normalization function.
        
    Returns:
        Test accuracy.
    """
    model.eval()
    test_loss = 0
    acc_mean = 0
    k = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(normalizer(data))
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            acc_mean += (pred == target).float().mean().item()
            k += 1

    test_loss /= k
    test_accuracy = (100. * acc_mean) / k
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n')
    return test_accuracy

def test_adv(model, device, test_adv_loader, normalizer=normalizer, epsilon=8/255, alpha=0.01, num_iter=7, restarts=1):
    """
    Evaluate the model on adversarially perturbed test data.
    
    Args:
        model: The neural network model.
        device: Device to use for evaluation (CPU or GPU).
        test_adv_loader: DataLoader for adversarial test data.
        normalizer: Normalization function.
        epsilon: Maximum perturbation.
        alpha: Step size for each iteration.
        num_iter: Number of iterations.
        restarts: Number of restarts for the attack.
        
    Returns:
        Adversarial test accuracy.
    """
    model.eval()
    test_loss = 0
    acc_mean = 0
    k = 0

    for data, target in test_adv_loader:
        data, target = data.to(device), target.to(device)
        delta = pqd_attacks(model, data, target, epsilon, alpha, num_iter, restarts)
        adv_data = normalizer(torch.clamp(data + delta, 0, 1))

        with torch.no_grad():
            adv_output = model(adv_data)

        test_loss += criterion(adv_output, target).item()
        pred = adv_output.argmax(dim=1)
        acc_mean += (pred == target).float().mean().item()
        k += 1

    test_loss /= k
    test_adv_accuracy = (100. * acc_mean) / k
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_adv_accuracy:.2f}%\n')
    return test_adv_accuracy

# Training and testing loop
for epoch in tqdm_notebook(range(1, 2)):
    run_loss = train(model, device, train_loader, optimizer, epoch)
    test_acc = test(model, device, test_loader)
    test_acc_adv = test_adv(model, device, test_adv_loader)

    # Save the model state
    PATH = "/content/drive/MyDrive/model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': run_loss
    }, PATH)

    print(f'\nAttack success rate: {test_acc - test_acc_adv:.2f}%\n')
