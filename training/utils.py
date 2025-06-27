from torch import nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from typing import Tuple
import torch
import seaborn as sns
from torchvision import datasets
import torchmetrics
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import ConfusionMatrixDisplay

class ChannelRepeat(nn.Module):
    def __init__(self):
        super(ChannelRepeat, self).__init__()

    def forward(self, x):
        return x.repeat(3, 1, 1) # Repeat the channel dimension


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420, test=False
):
    """
    Split a dataset into training, validation, and optionally test sets with PyTorch DataLoader.

    Args:
        dataset: PyTorch Dataset object
        batch_train: Batch size for training
        batch_eval: Batch size for validation/test
        val_size: Size of validation (and test if test=True) set as fraction of total dataset
        shuffle: Whether to shuffle the indices before splitting
        seed: Random seed for reproducibility
        test: If True, creates three splits (train/val/test) instead of two (train/val)

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set if test=True, else None
    """
    # Calculate dataset size and create indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    if test:
        # For test mode, we want three equal splits
        # If val_size is 0.2, we want:
        # - test_size = 0.1 (half of val_size)
        # - val_size = 0.1 (half of val_size)
        # - train_size = 0.8 (remaining portion)
        split_size = int(np.floor(val_size * dataset_size / 2))

        # Create the splits
        test_indices = indices[:split_size]  # First portion for test
        val_indices = indices[split_size:split_size * 2]  # Second portion for validation
        train_indices = indices[split_size * 2:]  # Remainder for training
    else:
        # For validation-only mode, we want two splits
        val_split = int(np.floor(val_size * dataset_size))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]
        test_indices = None

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices) if test else None

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_eval, sampler=test_sampler) if test else None

    return train_loader, val_loader, test_loader

def choose_random_balanced_subset(y, n_samples=500):
    unique_classes, counts = np.unique(y, return_counts=True)
    # print(counts)
    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = min(n_samples, len(cls_indices))
        selected_cls_indices = np.random.choice(cls_indices, size=n_cls, replace=False)
        selected_indices.extend(selected_cls_indices.tolist())
    return selected_indices


def prepare_data(transform_tr, transform_infer, batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares the training, validation, and test data loaders.
    
    Args:
        transform_tr: Transformation for training data.
        transform_infer: Transformation for inference data.
    
    Returns:
        Tuple of DataLoader objects for training, validation, and test datasets.
    """
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_tr)
    
    train_loader, val_loader, _ = torch_train_val_split(
        dataset,
        batch_train=batch_size,
        batch_eval=batch_size
    )       

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_infer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_results(model: torch.nn.Module, test_loader: DataLoader) -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torchmetrics.classification.Precision(task="multiclass", num_classes=10).to(device)
    recall = torchmetrics.classification.Recall(task="multiclass", num_classes=10).to(device)
    confmat = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=10).to(device)
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=False)
            acc = metric(preds, target)
            prec = precision(preds, target)
            rec = recall(preds, target)
            conf = confmat(preds, target)

    acc = metric.compute()
    prec = precision.compute()
    rec = recall.compute()
    conf = confmat.compute()

    results = {
        "precision": [prec.item()],
        "recall": [rec.item()],
        "accuracy": [acc.item()]
        }
    return results, conf


def save_confusion_matrix_plot(conf_metric: torch.Tensor, filename: str) -> None:
    """
    Save confusion matrix plot to a file.
    
    Args:
        conf_metric (torch.Tensor): Confusion matrix tensor.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_metric.cpu().numpy(), annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def load_precomputed_features(n_s_per_layer, layers_export, subset='train'):
    precompute_x_l = sorted(glob(f'./data/precompute/{subset}/x{layers_export}_*.npy'))
    precompute_y_l = sorted(glob(f'./data/precompute/{subset}/y{layers_export}_*.npy'))
    y = np.array([np.load(f) for f in precompute_y_l])
    idxs = choose_random_balanced_subset(y, n_samples=n_s_per_layer) 
    x = np.array([np.load(precompute_x_l[idx]) for idx in idxs])
    y = np.array([y[idx] for idx in idxs]).reshape(-1)
    
    return x, y


def save_confusion_matrix_sklearn(cm, filename: str) -> None:
    """
    Save confusion matrix plot to a file using sklearn's ConfusionMatrixDisplay.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        filename (str): Path to save the plot.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(filename)
    plt.show()