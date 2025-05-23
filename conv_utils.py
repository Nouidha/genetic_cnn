import math
import os
import random
from enum import Enum
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


transform  = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5),  # Mean for each channel
                         (0.5, 0.5, 0.5))  # Std for each channel
])

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using GPU

class DatasetName(Enum):
    CIFAR10 = 0
    CIFAR100 = 1

def load_train_test_dataset(dataset:DatasetName, train_size, test_size, random_state=42, force_download=False):
    """

    :param dataset: enum value designing the name of the dataset
    :param train_size: should be between 0 and 1, representing the percentage of data to be used for training
    :param test_size: should be between 0 and 1, representing the percentage of data to be used for testing
    :param random_state: random seed
    :param force_download: force the download the dataset
    :return:
    """
    if dataset == DatasetName.CIFAR100:
        download = force_download or not os.path.exists('./data/cifar-100-batches-py')
        train_dataset = tv.datasets.CIFAR10(root='./data', train=True, download=download,
                                            transform=transform)
    elif dataset == DatasetName.CIFAR10:
        download = force_download or not os.path.exists('./data/cifar-10-batches-py')
        train_dataset = tv.datasets.CIFAR10(root='./data', train=True, download=download,
                                            transform=transform)

    targets = train_dataset.targets

    # Stratified split (10% of the data, balanced across classes)
    indices = list(range(len(targets)))
    train_indices, test_indices = train_test_split(indices, train_size=train_size, test_size=test_size, stratify=targets, random_state=random_state)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(train_dataset, test_indices)
    return train_subset, test_subset, len(train_dataset.classes), train_dataset.data.shape[1:]


def train_model(train_data, test_data, model, optimizer, device, n_epochs=10, verbose=False):
    """

    :param train_data: training dataset
    :param test_data: test dataset
    :param model: nn.Model
    :param optimizer: optimizer
    :param device: device
    :param n_epochs:
    :return:
    """
    test_acc = 0
    for epoch in range(0, n_epochs):

        train_loss, train_acc = run_epoch(train_data, model.train(), optimizer, device)
        if verbose:
            print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(train_loss, train_acc))

        # Run **validation**
        test_loss, test_acc = run_epoch(test_data, model.eval(), optimizer, device)
        if verbose:
            print('Test loss:   {:.6f} | Test accuracy:   {:.6f}'.format(test_loss, test_acc))

    return test_acc

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def run_epoch(data, model, optimizer, device):
    """

    :param data: data loader
    :param model: nn.Module
    :param optimizer:
    :param device:
    :return: average loss, average accuracy
    """
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    for images, labels in tqdm(data):
        images, labels = images.to(device), labels.to(device)

        # Get output predictions
        out = model(images)

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, labels))

        # Compute loss
        loss = F.cross_entropy(out, labels)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy

