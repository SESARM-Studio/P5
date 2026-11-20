import argparse
import csv
import random
import time
from db_scripts.database_api import *
from db_scripts.database_login import *

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import efficientnet_v2_s




"""
Main structure is inspired by PyTorch docs:
https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

class IndexedDataset(Dataset):
    """
    Wrapper based on docs from:
    https://github.com/pytorch/pytorch/blob/282ec02fd44e135fba816d1e97a92848ac23a117/torch/utils/data/dataset.py#L39
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, img_id: list) -> list:
        x, y = self.dataset[img_id]
        return x, y, img_id

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True)
parser.add_argument("--training-id", required=True)
parser.add_argument("-s", "--seed", type=int, required=True)
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("--csv-output", default=None)
parser.add_argument("--dist-output", default="distribution_results.csv")
args = parser.parse_args()

# These lines of code ensure that a seed gets the same accuracy on different GPU's on AI-lab.
# It comes from https://docs.pytorch.org/docs/2.14/notes/randomness.html and implements until
# and including headline "CUDA convolution determinism"
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


TENSOR_MEAN_CHANNELS = (0.5, 0.5, 0.5)
TENSOR_STD_CHANNELS = (0.5, 0.5, 0.5)
BATCH_SIZE = 64
NUM_WORKERS = 8

training_data = datasets.CIFAR10(
    root="/data",
    train=True,
    download=False,
    transform=v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(TENSOR_MEAN_CHANNELS, TENSOR_STD_CHANNELS)])
)

test_data = datasets.CIFAR10(
    root="/data",
    train=False,
    download=False,
    transform=v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(TENSOR_MEAN_CHANNELS, TENSOR_STD_CHANNELS)])
)

seed = torch.Generator().manual_seed(args.seed)

if args.exp == 3:
    training_set_size = .8
    validation_set_size = .2
    train_set, validation_set = random_split(IndexedDataset(training_data),
                                             [training_set_size, validation_set_size], generator=seed)
else:
    train_set = IndexedDataset(training_data)
    validation_set = None

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

validation_dataloader = DataLoader(validation_set, batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) \
    if validation_set is not None else None

test_dataloader = DataLoader(IndexedDataset(test_data), batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

""" 
Some  default strides of EfficientNetV2-S is overriden, as CIFAR-10 resolution is only 32x32 
Model structure of EfficientNetV2-S can be seen on https://hfviewer.com/torchvision/efficientnet_v2_s 
"""

STRIDE = (1,1) # Horizontal and vertical
FEATURE_0_0 = "features.0.0"
FEATURE_2_0_BLOCK_0_0 = "features.2.0.block.0.0"

model = efficientnet_v2_s(weights=None, num_classes=10)
model.get_submodule(FEATURE_0_0).stride = STRIDE
model.get_submodule(FEATURE_2_0_BLOCK_0_0).stride = STRIDE
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss, correct = 0, 0
    for X, labels_in_batch, _ in dataloader:
        X, labels_in_batch = X.to(device), labels_in_batch.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, labels_in_batch)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save correct and loss for avg accuracy and loss
        test_loss += loss_fn(pred, labels_in_batch).item()
        correct += (pred.argmax(1) == labels_in_batch).type(torch.float).sum().item()

        avg_loss = test_loss / num_batches
        accuracy = correct / size
        return accuracy, avg_loss

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, labels_in_batch, image_ids_in_batch in dataloader:
            X, labels_in_batch = X.to(device), labels_in_batch.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, labels_in_batch).item()
            correct += (pred.argmax(1) == labels_in_batch).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    return accuracy, avg_loss

def test(dataloader, model, loss_fn, args, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, labels_in_batch, image_ids_in_batch in dataloader:
            X, labels_in_batch = X.to(device), labels_in_batch.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, labels_in_batch).item()
            correct += (pred.argmax(1) == labels_in_batch).type(torch.float).sum().item()

            # Iterates over each image of the batch, and writes class predictions,
            # along with seed, epoch, image id and correct label to CSV file
            with open(f"{args.dist_output}", "a", newline='') as s:
                batch_info = zip(image_ids_in_batch.tolist(), labels_in_batch.tolist(), pred.softmax(1).tolist())
                for image_id, label, image_prediction in batch_info:
                    class_predictions = [f"{class_prediction:.4f}" for class_prediction in image_prediction]
                    csv.writer(s).writerow([args.seed, epoch, image_id, label] + class_predictions)

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    return accuracy, avg_loss

epoch_start = time.time();
list_of_epochs_accuracy = []
for epoch in range(args.epochs):
    epoch_time = time.time()
    print(f"\nEpoch {epoch}\n-------------------------------")
    train_avg_acc, train_avg_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_avg_acc, val_avg_loss = validation(validation_dataloader, model, loss_fn)
    test_avg_acc, test_avg_loss = test(test_dataloader, model, loss_fn, args, epoch)


    print(f"Training accuracy: {train_avg_acc}, Train avg loss: {train_avg_loss}")
    print(f"Validation accuracy: {val_avg_acc}, Validation avg loss: {val_avg_loss}")
    print(f"Testing accuracy: {test_avg_acc}, Testing Avg loss: {test_avg_loss}")


    print(f"Time: {time.time() - epoch_time}")

    if args.csv_output is None:
        list_of_epochs_accuracy.append([epoch, train_avg_acc])
    else:
        with open(f"{args.csv_output}", "a", newline='') as f:
            csv.writer(f).writerow([args.seed, epoch, train_avg_acc, train_avg_loss, val_avg_acc, val_avg_loss, test_avg_acc, test_avg_loss, time.time() - epoch_time])

end_time = datetime.now().isoformat()

if args.csv_output is None:
    client = setup_supabase_client()
    finalize_training(client, args.training_id, list_of_epochs_accuracy, end_time)

print("Done!")
print(f"Total time: {time.time() - epoch_start}")
