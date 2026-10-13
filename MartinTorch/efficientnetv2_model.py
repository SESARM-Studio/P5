import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets
from torch import nn
import time
import random
import numpy as np
import math

# These lines of code ensure that a seed gets the same accuracy on different GPU's on AI-lab. 
# It comes from https://docs.pytorch.org/docs/2.14/notes/randomness.html and implements until and including headline "CUDA convolution determinism"
# All lines were tested at once but not if there is a minimum needed e.g. maybe last 3 lines are not needed, idk I havent tested without.
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# Until this

# This is the initialisation of the model. weights is none at default but this makes it clear that we want random initialisation of weights.
model = efficientnet_v2_s(weights=None)

# Changes the last classification layer to output 10 classes as base architechture is based on ImageNet1k and outputs to 1k classes.
model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=10,
    bias=True
)

# Loading of the training data and the test data. download = False because the data is already on AI-lab. Transform is basic pytorch documentation implementation.
training_data = datasets.CIFAR10(
    root = "data",
    train = True,
    download = False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

test_data = datasets.CIFAR10(
    root = "data",
    train = False,
    download = False,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

# batch_size is the amount of images the model should train at once, num_workers is how many sub_processes it should use for loading and pin_memory is if it should pin the memory in your device.
# Do not run this code on your own device. This, including some other stuff, makes it so the AI-Lab GPU uses 100% memory.
train_dataloader = DataLoader(training_data, batch_size=512, num_workers=4, pin_memory=True) # Can add shuffle=True as parameter for code on line 51.
test_dataloader = DataLoader(test_data, batch_size=512, num_workers=4, pin_memory=True) # Can add shuffle=True as parameter for code on line 51.

# Makes sure the code uses the GPU. Without this the fair use agreement of AI-lab is violated.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Hyperparameters
learning_rate = 1e-3
epochs = 600

# Initialize the loss function. CrossEntropyLoss combines LogSoftMax and NLLLoss (Negative Log Likelihood Loss)
loss_fn = nn.CrossEntropyLoss()

# Optimizer algorithm. It uses the SGD (Stochastic Gradient Descent) optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for (X, y) in dataloader: # X consists of a tensor with every image in the batchsize aka. 512 tensors of images. y consists of their respective labels.
        X = X.to(device, non_blocking = True) # Pushes it to the GPU
        y = y.to(device, non_blocking = True) # Pushes it to the GPU
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
            X = X.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Top 1 accuracy

    correct /= size
    return 100*correct


# THIS is just because AI-labs .out file is filled with print() statements so instead of having 1 long line of all the epochs I just split it into 100 epochs.
# its completely useless for the final file. You can see how it looks in the efficient_test.out
training_sections = []
testing_sections = []
accuracy_sections = []

program_start_time = time.time()

for segment in range(math.ceil(epochs / 100)):
    training_sections.append(dict())
    testing_sections.append(dict())
    accuracy_sections.append(dict())
# UNTIL THIS


for t in range(epochs): # For loop is important to go through every epoch BUT only the commented parts is relevant for training and testing
    train_start_time = time.time()
    train_loop(train_dataloader, model, loss_fn, optimizer) # THIS IS THE ONLY IMPORTANT PART FOR TRAINING
    train_end_time = time.time()
    if t == 0: # DUMB IF-ELSE JUST FOR MY SPECIFIC efficient_test.out
        training_sections[t][t+1] = round(train_end_time - train_start_time, 2)
        test_start_time = time.time()
        accuracy_sections[t][t+1] = round(test_loop(test_dataloader, model), 2) #ONLY test_loop(test_dataloader, model) IS IMPORTANT FOR TESTING, RIGHT NOW IT JUST RETURNS THE ACCURACY FOR THIS EPOCH.
        test_end_time = time.time()
        testing_sections[t][t+1] = round(test_end_time - test_start_time, 2)
    else:
        segment = math.ceil(t/100)-1
        training_sections[segment][t+1] = round(train_end_time - train_start_time, 2)
        test_start_time = time.time()
        accuracy_sections[segment][t+1] = round(test_loop(test_dataloader, model), 2)
        test_end_time = time.time()
        testing_sections[segment][t+1] = round(test_end_time - test_start_time, 2)

# PRINT STATEMENTS. USELESS FOR REAL CODE.
for x in range(len(training_sections)):
    print(f"Training times segment {x}: {training_sections[x]}")
for x in range(len(testing_sections)):
    print(f"Testing times segment {x}: {testing_sections[x]}")
for x in range(len(accuracy_sections)):
    print(f"Accuracies segment {x}: {accuracy_sections[x]}")

# JUST HOW LONG THE CODE TAKES TO RUN TO COMPLETION.
program_end_time = time.time()

program_seconds = program_end_time - program_start_time
hours = int(program_seconds // 3600)
minutes = int((program_seconds % 3600) // 60)
seconds = program_seconds % 60

print(f"It takes {hours}h {minutes}m and {seconds:.2f}s to run {epochs} epochs")
