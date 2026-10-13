import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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

labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

# Show a 3x3 grid of the images in the dataset
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.permute(1, 2, 0)) # ChatGPT code in this line because PyTorch docu is BW and Cifar is RGB. Original code is plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_dataloader = DataLoader(training_data, batch_size=64) # Can add shuffle=True as parameter for code on line 51.
test_dataloader = DataLoader(test_data, batch_size=64) # Can add shuffle=True as parameter for code on line 51.

# Show a random img from the training dataset to show shuffle is working
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img.permute(1, 2, 0)) # ChatGPT code in this line because PyTorch docu is BW and Cifar is RGB. Original code is plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
# print(f"Label: {label}")

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv_pool_stack(x)
        x = torch.flatten(x, 1)
        logits = self.linear_relu_stack(x)
        return logits

# Code to show the structure of the NN. Uses device from line 61.
# model = NeuralNetwork().to(device)
# print(model)

model = NeuralNetwork()

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function. CrossEntropyLoss combines LogSoftMax and NLLLoss (Negative Log Likelihood Loss)
loss_fn = nn.CrossEntropyLoss()

# Optimizer algorithm. It uses the SGD (Stochastic Gradient Descent) optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")