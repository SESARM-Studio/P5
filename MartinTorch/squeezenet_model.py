import torch
import torch.nn.functional as F
from torchvision.models import squeezenet1_1
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets
from torch import nn
import time

torch.manual_seed(15)

model = squeezenet1_1(weights=None)

model.classifier[1] = nn.Conv2d(
    in_channels=512,
    out_channels=10,
    kernel_size=1
)

training_data = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

test_data = datasets.CIFAR10(
    root = "data",
    train = False,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

train_dataloader = DataLoader(training_data, batch_size=64) # Can add shuffle=True as parameter for code on line 51.
test_dataloader = DataLoader(test_data, batch_size=64) # Can add shuffle=True as parameter for code on line 51.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Using:", device)

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function. CrossEntropyLoss combines LogSoftMax and NLLLoss (Negative Log Likelihood Loss)
loss_fn = nn.CrossEntropyLoss()

# Optimizer algorithm. It uses the SGD (Stochastic Gradient Descent) optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    return 100*correct

training_time = dict()
testing_time = dict()
accuracies = dict()
for t in range(epochs):
    start_time = time.time()
    train_loop(train_dataloader, model, loss_fn, optimizer)
    end_time = time.time()
    training_time[t+1] = round(end_time - start_time, 2)

    start_time = time.time()
    accuracies[t+1] = round(test_loop(test_dataloader, model), 2)
    end_time = time.time()
    testing_time[t+1] = round(end_time - start_time, 2)

print(f"Training times: {training_time}")
print(f"Testing times: {testing_time}")
print(f"Accuracies: {accuracies}")

