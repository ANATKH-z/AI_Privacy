"""
Utilities for loading and distributing MNIST data across clients.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def load_mnist_data():
    """
    Load MNIST training and test datasets.
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def distribute_data_iid(dataset, num_clients, seed=42):
    """
    Distribute dataset in IID (Independent and Identically Distributed) manner.
    Each client gets an equal number of samples.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        seed: Random seed for reproducibility
    
    Returns:
        client_datasets: List of Subset datasets, one per client
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    
    indices = np.random.permutation(num_samples)
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test dataset.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        accuracy: Test accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

