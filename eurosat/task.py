"""eurosat: A Flower / PyTorch app."""

import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"The currently tested dataset are",
    category=UserWarning,
)

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, bit_width=32):
        super(Net, self).__init__()
        self.bit_width = bit_width

        # Import quantized layers if needed
        if bit_width < 32:
            from eurosat.quantization import QuantizedConv2d, QuantizedLinear
            self.conv1 = QuantizedConv2d(3, 32, 5, bit_width=bit_width)
            self.conv2 = QuantizedConv2d(32, 64, 5, bit_width=bit_width)
            self.conv3 = QuantizedConv2d(64, 96, 3, bit_width=bit_width)
            self.fc1 = QuantizedLinear(96 * 5 * 5, 128, bit_width=bit_width)
            self.fc2 = QuantizedLinear(128, 128, bit_width=bit_width)
            self.fc3 = QuantizedLinear(128, 10, bit_width=bit_width)
        else:
            # Standard FP32 layers
            self.conv1 = nn.Conv2d(3, 32, 5)
            self.conv2 = nn.Conv2d(32, 64, 5)
            self.conv3 = nn.Conv2d(64, 96, 3)
            self.fc1 = nn.Linear(96 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 10)

        # BatchNorm and pooling stay in FP32 (standard practice)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 96 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


dataset_cache = None  # Cache dataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Load partition EuroSAT data using standard datasets library."""
    # Only load dataset once
    global dataset_cache
    if dataset_cache is None:
        # Load the full dataset
        dataset_cache = load_dataset("tanganke/eurosat", split="train")

    # Create IID partitions manually
    total_samples = len(dataset_cache)
    samples_per_partition = total_samples // num_partitions

    # Calculate start and end indices for this partition
    start_idx = partition_id * samples_per_partition
    if partition_id == num_partitions - 1:
        # Last partition gets remaining samples
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_partition

    # Create indices for this partition
    partition_indices = list(range(start_idx, end_idx))

    # Split into train/test (80/20)
    np.random.seed(42)
    np.random.shuffle(partition_indices)
    split_point = int(len(partition_indices) * 0.8)
    train_indices = partition_indices[:split_point]
    test_indices = partition_indices[split_point:]

    # Create subsets
    train_dataset = Subset(dataset_cache, train_indices)
    test_dataset = Subset(dataset_cache, test_indices)

    # Apply transforms and create dataloaders
    train_dataset_transformed = dataset_cache.select(train_indices).with_transform(apply_transforms)
    test_dataset_transformed = dataset_cache.select(test_indices).with_transform(apply_transforms)

    # Use configurable batch size with num_workers for faster data loading
    trainloader = DataLoader(
        train_dataset_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    testloader = DataLoader(
        test_dataset_transformed,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    train_accuracy = correct / total
    return avg_trainloss, train_accuracy


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def create_run_dir(bit_width: int = 32) -> tuple[Path, str]:
    """Create a directory where to save results from this run, organized by bit-width."""
    import os

    # Get bit-width from environment if not provided
    if bit_width == 32:
        bit_width = int(os.getenv("QUANTIZATION_BITS", "32"))

    # Create output directory organized by bit-width
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # New structure: outputs/{bit_width}bit/run_{timestamp}/
    save_path = Path.cwd() / f"outputs/{bit_width}bit/run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path, str(save_path)