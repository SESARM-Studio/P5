import argparse
import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import efficientnet_v2_s
import time
import csv

training_data = datasets.CIFAR10(
    root="/data",
    train=True,
    download=False,
    transform=v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

test_data = datasets.CIFAR10(
    root="/data",
    train=False,
    download=False,
    transform=v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

batch_size = 64

# Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)


"""
TODO: Revise index dataset wrapper
"""
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """Returnerer (billede, label, CIFAR-10-indeks)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


train_dataloader = DataLoader(IndexedDataset(training_data), batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(IndexedDataset(test_data), batch_size=batch_size,
                             shuffle=True, num_workers=8, pin_memory=True)

for X, y, _ in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = efficientnet_v2_s(weights=None, num_classes=10)
model.features[0][0].stride = (1, 1)  # stem
model.features[2][0].block[0][0].stride = (1, 1)  # første lag i blok 2
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, required=True)
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("--model-output", default="efficientnet_model.pth")
parser.add_argument("--csv-output", default="training_results.csv")
parser.add_argument("--dist-output", default="distribution_results.csv")
args = parser.parse_args()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)


def test(dataloader, model, loss_fn, args, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y, idx in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            probs = pred.softmax(dim=1)
            torch.set_printoptions(precision=3, sci_mode=False)
            with open(f"{args.dist_output}", "a", newline='') as s:
                for i, label, p in zip(idx.tolist(), y.tolist(), pred.softmax(1).tolist()):  #
                    csv.writer(s).writerow([args.seed, epoch, i, label] + [f"{x:.4f}" for x in p])  #
            # print(f"Prediction: {probs}");
    test_loss /= num_batches
    correct /= size
    # print(f"y: {y}");
    # with open(f"{csv_output}", "a", newline='') as f:
    #    csv.writer(f).writerow([seed, epoch, round(100*correct, 1)])
    # print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return round(100 * correct, 1), round(test_loss, 8)


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

t_start = time.time();
for t in range(args.epochs):
    epoch_time = time.time()
    print(f"\nEpoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    t_acc, t_loss = test(test_dataloader, model, loss_fn, args, t)

    print(f"Accuracy: {t_acc}, Avg loss: {t_loss}")
    print(f"Time: {time.time() - epoch_time}")
    with open(f"{args.csv_output}", "a", newline='') as f:
        csv.writer(f).writerow([args.seed, t, t_acc, time.time() - epoch_time])

print("Done!")
print(f"Total time: {time.time() - t_start}")


torch.save(model.state_dict(), args.model_output)
print(f"Saved PyTorch Model State to {args.model_output}")