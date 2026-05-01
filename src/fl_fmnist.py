import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST as TorchvisionFMNIST
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as scipy_norm
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import random
import math
import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import time
import warnings
import traceback

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("GPU DETECTION AND CONFIGURATION")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"TF32 Enabled: True")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

print("=" * 80)

def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB | Util: {(allocated/total)*100:.1f}%")

print_gpu_utilization()


DATASET_NAME = "FMNIST"
MODEL_NAME = "FourLayerCNN"
NUM_CLASSES = 10

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
GLOBAL_ROUNDS = 150
NON_IID_ALPHA = 0.5
LEARNING_RATE = 0.01
LR_DECAY_STEP = 50
LR_DECAY_RATE = 0.5

MALICIOUS_RATIOS = [10, 20, 30, 40]
ATTACK_TYPES = ["ISA", "MinMax", "LIE", "TRIM"]

ISA_RHO = 0.80
ISA_EPS_MIN = 0.80
ISA_EPS_MAX = 1.80
ISA_TAU = 10.0
ISA_DELTA = 0.03
ISA_KAPPA = 3.0
ISA_BETA = 0.0
ISA_STD_CAP = 2.0
ISA_MIN_SHIFT = 0.005

MINMAX_SCALE = 4.0

TRIM_RATIO = 0.10

TRIMMED_MEAN_BETA = 0.1
KRUM_F = 2
MULTI_KRUM_M = 7
MULTI_KRUM_K = 5
BULYAN_F = 1

HADFL_CLIP_NORM = 1.0
HADFL_HDBSCAN_MIN_CLUSTER = 3
HADFL_HDBSCAN_MIN_SAMPLES = 2
HADFL_WARMUP_ROUNDS = 5
HADFL_MAD_TAU = 2.5
HADFL_MAX_DROP_RATIO = 0.30
HADFL_PCA_COMPONENTS = 3
HADFL_GEOM_REJECT_PCT = 0.90
HADFL_ALPHA_MEMORY = 0.80
HADFL_LAMBDA_GEOM = 0.65
HADFL_LAMBDA_MAD = 0.35
HADFL_REP_FLOOR = 1e-8
HADFL_HARD_REJECT_ZSCORE = 0.3

AGGREGATION_METHODS = ["Mean", "Median", "Trimmed-Mean", "Krum", "Multi-Krum", "Bulyan", "HADFL"]

RESULTS_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'Results_FMNIST'))
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("CENTRALIZED HYPERPARAMETER CONFIGURATION")
print("=" * 80)
print(f"Dataset: {DATASET_NAME} | Model: {MODEL_NAME} | Classes: {NUM_CLASSES}")
print(f"Clients: N={NUM_CLIENTS}, tau={CLIENTS_PER_ROUND}, E={LOCAL_EPOCHS}, B={BATCH_SIZE}")
print(f"Training: T={GLOBAL_ROUNDS} rounds, LR={LEARNING_RATE} (decay {LR_DECAY_RATE} every {LR_DECAY_STEP} rounds)")
print(f"Non-IID: Dirichlet alpha={NON_IID_ALPHA}")
print(f"Attacks: {ATTACK_TYPES}")
print(f"Malicious Ratios: {MALICIOUS_RATIOS}%")
print(f"Aggregations: {AGGREGATION_METHODS}")
print(f"Results Directory: {RESULTS_DIR}")
print("=" * 80)


print("Loading Fashion-MNIST dataset...")
dataset_dir = './mnist_data'
os.makedirs(dataset_dir, exist_ok=True)

train_data_raw = TorchvisionFMNIST(root=dataset_dir, train=True, download=True)
test_data_raw = TorchvisionFMNIST(root=dataset_dir, train=False, download=True)

print(f"Training samples: {len(train_data_raw)}")
print(f"Test samples: {len(test_data_raw)}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"Classes ({len(class_names)}): {class_names}")


class FMNISTDataset(Dataset):
    """Pre-caches all transformed Fashion-MNIST samples as tensors in memory."""
    def __init__(self, torchvision_dataset, transform=None):
        print(f"  Pre-caching {len(torchvision_dataset)} samples into memory...")
        images_list = []
        labels_list = []
        for i in range(len(torchvision_dataset)):
            img, label = torchvision_dataset[i]
            if transform:
                img = transform(img)
            images_list.append(img)
            labels_list.append(label)
        self.images = torch.stack(images_list)
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        print(f"  Cached: shape={self.images.shape}, dtype={self.images.dtype}, "
              f"memory={self.images.element_size() * self.images.nelement() / 1024**2:.1f} MB")
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

train_dataset = FMNISTDataset(train_data_raw, transform=train_transform)
test_dataset = FMNISTDataset(test_data_raw, transform=test_transform)

print(f"Image shape: {train_dataset[0][0].shape}")


def create_federated_data_dirichlet(dataset, num_clients, alpha=0.5, num_classes=10):
    """Split dataset among clients using Dirichlet distribution for Non-IID data."""
    labels = dataset.labels.numpy() if hasattr(dataset, 'labels') else np.array([dataset[i][1] for i in range(len(dataset))])
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_k) / num_clients) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            if i < num_clients:
                client_indices[i].extend(split.tolist())
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


print("Creating non-IID federated data splits (Dirichlet alpha=0.5)...")
client_data_indices = create_federated_data_dirichlet(train_dataset, NUM_CLIENTS, alpha=NON_IID_ALPHA, num_classes=NUM_CLASSES)

distribution_records = []
for cid in range(NUM_CLIENTS):
    client_labels = train_dataset.labels[client_data_indices[cid]].numpy() if hasattr(train_dataset, 'labels') else [train_dataset[idx][1] for idx in client_data_indices[cid]]
    unique, counts = np.unique(client_labels, return_counts=True)
    class_dist = {f'class_{k}': 0 for k in range(NUM_CLASSES)}
    for cls, cnt in zip(unique, counts):
        class_dist[f'class_{cls}'] = int(cnt)
    distribution_records.append({'client_id': cid, 'total_samples': len(client_data_indices[cid]), **class_dist})

dist_df = pd.DataFrame(distribution_records)
dist_df.to_csv(os.path.join(RESULTS_DIR, 'fmnist_federated_data_distribution.csv'), index=False)

print(f"Non-IID split complete. Samples per client: min={min(len(c) for c in client_data_indices)}, max={max(len(c) for c in client_data_indices)}, avg={np.mean([len(c) for c in client_data_indices]):.0f}")
print(f"Distribution saved to: {RESULTS_DIR}/fmnist_federated_data_distribution.csv")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=0)

print("DataLoaders created.")

# A properly designed 4-layer CNN for 28x28 grayscale image classification:
#   Conv1(1->32, 5x5) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   Conv2(32->64, 5x5) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   FC1(1024->512) -> BatchNorm -> ReLU -> Dropout(0.3)
#   FC2(512->10)
# BatchNorm stabilizes training and improves convergence in federated settings.
# Dropout added for regularization to improve generalization.

class FourLayerCNN(nn.Module):
    """
    4-Layer CNN with BatchNorm for Fashion-MNIST (1x28x28 grayscale images, 10 classes).
    Architecture:
      Conv1: 1->32 filters, 5x5 kernel, BN, ReLU, MaxPool 2x2 -> 12x12x32
      Conv2: 32->64 filters, 5x5 kernel, BN, ReLU, MaxPool 2x2 -> 4x4x64
      FC1: 1024->512, BN, ReLU, Dropout(0.3)
      FC2: 512->10 (output)
    BatchNorm improves gradient flow and training stability across federated rounds.
    """
    def __init__(self, num_classes=10):
        super(FourLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # -> 12x12x32
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # -> 4x4x64
        x = x.view(-1, 64 * 4 * 4)                          # flatten
        x = self.dropout(self.relu(self.bn3(self.fc1(x))))  # -> 512
        x = self.fc2(x)                                      # -> 10
        return x


def create_model():
    """Create 4-Layer CNN with BatchNorm for Fashion-MNIST."""
    return FourLayerCNN(num_classes=NUM_CLASSES).to(device)


_model = create_model()
total_params = sum(p.numel() for p in _model.parameters())
trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
print("=" * 80)
print(f"MODEL: 4-Layer CNN with BatchNorm for Fashion-MNIST")
print(f"  Total parameters:     {total_params:>12,}")
print(f"  Trainable parameters: {trainable_params:>12,}")
print(f"  Input: 1x28x28 grayscale | Output: {NUM_CLASSES} classes")
print(f"  Architecture: Conv(32)->BN->Pool->Conv(64)->BN->Pool->FC(512)->BN->FC(10)")
print("=" * 80)
del _model
torch.cuda.empty_cache() if torch.cuda.is_available() else None


def train_client(model, data_indices, dataset, epochs, learning_rate=0.01):
    """Train model on client local data using SGD with gradient clipping."""
    model.train()

    idx = torch.tensor(data_indices, dtype=torch.long)
    client_images = dataset.images[idx]
    client_labels = dataset.labels[idx]
    client_loader = DataLoader(
        torch.utils.data.TensorDataset(client_images, client_labels),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0,
        drop_last=(len(data_indices) > BATCH_SIZE)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in client_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss / len(client_loader))
    return model.state_dict(), sum(epoch_losses) / len(epoch_losses)


def evaluate_model(model, data_loader):
    """Evaluate model; returns accuracy, precision, recall, f1, loss, predictions, labels."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = test_loss / len(data_loader)
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1, avg_loss, all_preds, all_labels


def compute_top5_accuracy(model, data_loader):
    """Compute top-5 accuracy."""
    model.eval()
    correct_top5, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    correct_top5 += 1
            total += labels.size(0)
    return correct_top5 / total if total > 0 else 0.0


def compute_per_class_fpr_fnr(y_true, y_pred, num_classes=10):
    """Compute macro-averaged FPR and FNR via one-vs-rest confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_fpr, per_fnr = [], []
    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - TP - FP - FN
        per_fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
        per_fnr.append(FN / (FN + TP) if (FN + TP) > 0 else 0.0)
    return float(np.mean(per_fpr)), float(np.mean(per_fnr))


def calculate_cosine_similarity(u1, u2):
    """Cosine similarity between two state dicts."""
    v1 = torch.cat([u1[k].flatten() for k in u1.keys()])
    v2 = torch.cat([u2[k].flatten() for k in u2.keys()])
    v1, v2 = torch.clamp(v1, -1e3, 1e3), torch.clamp(v2, -1e3, 1e3)
    return float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).clamp(-1, 1).item())


def calculate_update_norm(update):
    """L2 norm of update state dict."""
    return sum(min(float(torch.norm(update[k]).item()), 1e6) for k in update.keys())


def clip_model_weights(state_dict, min_val=-100.0, max_val=100.0):
    """Clip all tensor values in a state dict."""
    return OrderedDict({k: torch.clamp(v, min_val, max_val) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()})


print("Training and evaluation functions defined.")


def isa_attack(honest_updates, global_state=None, rho=ISA_RHO, eps_min=ISA_EPS_MIN,
               eps_max=ISA_EPS_MAX, tau=ISA_TAU, delta=ISA_DELTA, kappa=ISA_KAPPA,
               beta=ISA_BETA, std_cap=ISA_STD_CAP, min_shift=ISA_MIN_SHIFT):
    """ISA (Informed Statistical Attack) — coordinate-wise importance-adaptive poisoning attack."""
    dev = next(iter(honest_updates[0].values())).device
    n_honest = len(honest_updates)

    skip_keys = set()
    for k in honest_updates[0].keys():
        if 'running' in k or 'num_batches_tracked' in k:
            skip_keys.add(k)
    attack_keys = [k for k in honest_updates[0].keys() if k not in skip_keys]
    flat_honest = [torch.cat([u[k].flatten().float() for k in attack_keys]) for u in honest_updates]
    stacked = torch.stack(flat_honest)               # (n_honest, D)
    coord_mean = stacked.mean(dim=0)
    coord_std  = stacked.std(dim=0).clamp(min=1e-7, max=std_cap)
    D = coord_mean.numel()

    importance = coord_mean.abs()

    z_lo, z_hi = 0.0, float(kappa)
    for _ in range(20):
        z_mid = (z_lo + z_hi) / 2.0
        candidate = coord_mean - z_mid * coord_std
        frac_below = (stacked < candidate.unsqueeze(0)).float().mean().item()
        if frac_below < 0.10:
            z_lo = z_mid
        else:
            z_hi = z_mid
    z_base = max(z_lo, 0.5)

    imp_max = importance.max().clamp(min=1e-10)
    imp_norm = importance / imp_max                # in [0, 1]
    amplify  = eps_min + (eps_max - eps_min) * imp_norm
    z_effective = z_base * amplify



    top_k = max(1, int(rho * D))
    _, important_idx = torch.topk(importance, top_k)
    attack_mask = torch.zeros(D, device=dev, dtype=torch.bool)
    attack_mask[important_idx] = True

    attack_vec = coord_mean.clone()
    attack_vec[attack_mask] = coord_mean[attack_mask] - z_effective[attack_mask] * coord_std[attack_mask]


    diversity = torch.randn(D, device=dev) * delta * coord_std * 0.01
    attack_vec = attack_vec + diversity

    clip_range = torch.clamp(0.5 * coord_std, min=min_shift)
    attack_vec = torch.clamp(attack_vec,
                             coord_mean - clip_range,
                             coord_mean + clip_range)
    honest_norms = stacked.norm(dim=1)
    mean_norm = honest_norms.mean().item()
    attack_norm = attack_vec.norm().item()
    if attack_norm > 5.0 * mean_norm:
        attack_vec = attack_vec * (5.0 * mean_norm / (attack_norm + 1e-10))
    attack_vec = torch.clamp(attack_vec, -10.0, 10.0)  # numeric safety


    poisoned = OrderedDict()
    idx = 0
    for key in honest_updates[0].keys():
        if key in skip_keys:

            stacked_key = torch.stack([u[key].float() for u in honest_updates])
            poisoned[key] = torch.mean(stacked_key, dim=0).to(honest_updates[0][key].dtype)
        else:
            size = honest_updates[0][key].numel()
            poisoned[key] = attack_vec[idx:idx+size].reshape(honest_updates[0][key].shape)
            poisoned[key] = torch.clamp(poisoned[key], -100.0, 100.0)
            idx += size
    return poisoned


def minmax_attack(honest_updates, global_state=None, scale_factor=MINMAX_SCALE):
    """MinMax Attack (Shejwalkar & Houmansadr, NDSS 2021): scale deviation from mean updates."""
    poisoned_weights = OrderedDict()
    mean_updates = OrderedDict()
    for key in honest_updates[0].keys():
        stacked = torch.stack([u[key] for u in honest_updates])
        stacked = torch.clamp(stacked, -1e3, 1e3)
        mean_updates[key] = torch.mean(stacked, dim=0)
    for key in mean_updates.keys():
        deviation = honest_updates[0][key] - mean_updates[key]
        poisoned_weights[key] = torch.clamp(mean_updates[key] + scale_factor * deviation, -100.0, 100.0)
    return poisoned_weights


def lie_attack(honest_updates, global_state=None, num_malicious=1, num_honest=None):
    """LIE Attack (Baruch et al., NeurIPS 2019): coordinate-wise statistical poisoning."""
    if num_honest is None:
        num_honest = len(honest_updates)
    n_total = num_honest + num_malicious
    s = 1
    cdf_arg = (n_total - num_malicious - s) / max(n_total - num_malicious, 1)
    cdf_arg = np.clip(cdf_arg, 0.001, 0.999)
    z_value = max(scipy_norm.ppf(cdf_arg), 0.0)
    
    poisoned_weights = OrderedDict()
    for key in honest_updates[0].keys():
        stacked = torch.stack([u[key] for u in honest_updates])
        stacked = torch.clamp(stacked, -10.0, 10.0)
        mu = torch.mean(stacked, dim=0)
        sigma = torch.clamp(torch.std(stacked, dim=0) + 1e-7, 0, 1.0)
        poisoned = mu - z_value * sigma
        poisoned_weights[key] = mu if torch.isnan(poisoned).any() else torch.clamp(poisoned, -10.0, 10.0)
    return poisoned_weights


def trim_attack(honest_updates, global_state=None, num_malicious=1, num_honest=None, trim_ratio=TRIM_RATIO):
    """TRIM Attack (Fang et al., USENIX Security 2020): binary search for boundary-evading perturbation."""
    if num_honest is None:
        num_honest = len(honest_updates)
    n_total = num_honest + num_malicious
    num_trimmed = int(np.ceil(n_total * trim_ratio))
    
    poisoned_weights = OrderedDict()
    for key in honest_updates[0].keys():
        stacked = torch.stack([u[key] for u in honest_updates])
        stacked = torch.clamp(stacked, -10.0, 10.0)
        mu = torch.mean(stacked, dim=0)
        sigma = torch.clamp(torch.std(stacked, dim=0) + 1e-7, 0, 1.0)
        
        z_low, z_high = 0.0, 3.0
        for _ in range(10):
            z_mid = (z_low + z_high) / 2.0
            candidate = mu - z_mid * sigma
            below_count = (stacked < candidate.unsqueeze(0)).float().sum(dim=0)
            frac_below = (below_count / num_honest).mean().item()
            if frac_below < trim_ratio:
                z_low = z_mid
            else:
                z_high = z_mid
        
        poisoned = mu - z_low * sigma
        poisoned_weights[key] = mu if torch.isnan(poisoned).any() else torch.clamp(poisoned, -10.0, 10.0)
    return poisoned_weights


ATTACK_FUNCTIONS = {
    "ISA": lambda updates, gs=None: isa_attack(updates, global_state=gs),
    "MinMax": lambda updates, gs=None: minmax_attack(updates, global_state=gs),
    "LIE": lambda updates, gs=None: lie_attack(updates, global_state=gs),
    "TRIM": lambda updates, gs=None: trim_attack(updates, global_state=gs),
}

print("=" * 80)
print("ATTACK FUNCTIONS DEFINED")
print("=" * 80)
print(f"  ISA (Proposed): rho={ISA_RHO}, eps=[{ISA_EPS_MIN},{ISA_EPS_MAX}], kappa={ISA_KAPPA}, std_cap={ISA_STD_CAP}, min_shift={ISA_MIN_SHIFT}")
print(f"  MinMax (NDSS 2021): scale_factor={MINMAX_SCALE}")
print(f"  LIE (NeurIPS 2019): z = Phi^-1((n-m-1)/(n-m))")
print(f"  TRIM (USENIX 2020): z_max via binary search, trim_ratio={TRIM_RATIO}")
print("=" * 80)


def mean_aggregate(client_updates_torch, device=None):
    """FedAvg: Element-wise mean of client updates."""
    first = list(client_updates_torch.values())[0]
    agg = OrderedDict()
    for key in first.keys():
        stacked = torch.stack([client_updates_torch[cid][key] for cid in client_updates_torch])
        if stacked.is_floating_point():
            agg[key] = torch.clamp(torch.mean(stacked, dim=0), -50.0, 50.0)
        else:
            agg[key] = stacked[0]
    return agg


def median_aggregate(client_updates_torch, device=None):
    """Coordinate-wise median of client updates."""
    first = list(client_updates_torch.values())[0]
    agg = OrderedDict()
    for key in first.keys():
        stacked = torch.stack([client_updates_torch[cid][key] for cid in client_updates_torch])
        agg[key] = torch.clamp(torch.median(stacked, dim=0).values, -50.0, 50.0)
    return agg


def trimmed_mean_aggregate(client_updates_torch, beta=TRIMMED_MEAN_BETA, device=None):
    """Trimmed Mean: Sort coordinate-wise, trim beta fraction from each end, then mean."""
    first = list(client_updates_torch.values())[0]
    agg = OrderedDict()
    n = len(client_updates_torch)
    trim_count = int(np.ceil(n * beta))
    for key in first.keys():
        stacked = torch.stack([client_updates_torch[cid][key] for cid in client_updates_torch])
        if not stacked.is_floating_point():
            agg[key] = stacked[0]
            continue
        sorted_updates, _ = torch.sort(stacked, dim=0)
        if trim_count > 0 and n > 2 * trim_count:
            trimmed = sorted_updates[trim_count:-trim_count]
        else:
            trimmed = sorted_updates
        agg[key] = torch.clamp(torch.mean(trimmed, dim=0), -50.0, 50.0)
    return agg


def krum_aggregate(client_updates_torch, f=KRUM_F, device=None):
    """Krum: select the client with minimum sum of distances to k nearest neighbors."""
    cids = list(client_updates_torch.keys())
    n = len(cids)
    if n < 2 * f + 3:
        return mean_aggregate(client_updates_torch, device)
    vectors = [torch.cat([client_updates_torch[cid][k].flatten().float() for k in client_updates_torch[cid]]) for cid in cids]
    mat = torch.stack(vectors)
    dists = torch.cdist(mat, mat, p=2)
    k = n - f - 2
    scores = []
    for i in range(n):
        sorted_d, _ = torch.sort(dists[i])
        scores.append(torch.sum(sorted_d[1:k+1]).item())
    best_idx = np.argmin(scores)
    best_cid = cids[best_idx]
    return OrderedDict({key: client_updates_torch[best_cid][key].clone() for key in client_updates_torch[best_cid]})


def multi_krum_aggregate(client_updates_torch, m=MULTI_KRUM_M, k=MULTI_KRUM_K, device=None):
    """Multi-Krum: Select m clients with smallest distance scores, then average."""
    cids = list(client_updates_torch.keys())
    n = len(cids)
    m = min(m, n)
    k = min(k, n - 1)
    vectors = [torch.cat([client_updates_torch[cid][key].flatten().float() for key in client_updates_torch[cid]]) for cid in cids]
    mat = torch.stack(vectors)
    dists = torch.cdist(mat, mat, p=2)
    scores = []
    for i in range(n):
        sorted_d, _ = torch.sort(dists[i])
        scores.append(torch.sum(sorted_d[1:k+1]).item())
    selected = np.argsort(scores)[:m]
    sel_cids = [cids[i] for i in selected]
    first = client_updates_torch[sel_cids[0]]
    agg = OrderedDict()
    for key in first.keys():
        stacked = torch.stack([client_updates_torch[c][key] for c in sel_cids])
        if stacked.is_floating_point():
            agg[key] = torch.clamp(torch.mean(stacked, dim=0), -50.0, 50.0)
        else:
            agg[key] = stacked[0]
    return agg


def bulyan_aggregate(client_updates_torch, f=BULYAN_F, device=None):
    """Bulyan: Multi-Krum selection followed by coordinate-wise trimmed mean."""
    cids = list(client_updates_torch.keys())
    n = len(cids)
    if n < 3 * f + 2:
        return mean_aggregate(client_updates_torch, device)
    
    theta = n - f - 4
    k_sel = n - f - 3
    vectors = [torch.cat([client_updates_torch[cid][key].flatten().float() for key in client_updates_torch[cid]]) for cid in cids]
    mat = torch.stack(vectors)
    dists = torch.cdist(mat, mat, p=2)
    scores = []
    for i in range(n):
        sorted_d, _ = torch.sort(dists[i])
        scores.append(torch.sum(sorted_d[1:min(k_sel+1, n)]).item())
    selected = np.argsort(scores)[:max(theta, 1)]
    sel_cids = [cids[i] for i in selected]
    
    first = client_updates_torch[sel_cids[0]]
    agg = OrderedDict()
    for key in first.keys():
        stacked = torch.stack([client_updates_torch[c][key] for c in sel_cids])
        if not stacked.is_floating_point():
            agg[key] = stacked[0]
            continue
        orig_shape = stacked[0].shape
        flat = stacked.view(len(sel_cids), -1)
        sorted_flat, _ = torch.sort(flat, dim=0)
        trimmed = sorted_flat[f:-f] if f > 0 and len(sel_cids) > 2*f else sorted_flat
        agg[key] = torch.clamp(torch.mean(trimmed, dim=0).view(orig_shape), -50.0, 50.0)
    return agg


def flatten_state_dict(state):
    keys_shapes = [(k, v.shape) for k, v in state.items()]
    vec = np.concatenate([state[k].ravel() for k, _ in keys_shapes], axis=0).astype(np.float64)
    return vec, keys_shapes

def unflatten_to_state_dict(vec, keys_shapes):
    out = {}
    idx = 0
    for k, shp in keys_shapes:
        size = int(np.prod(shp))
        out[k] = vec[idx:idx+size].reshape(shp)
        idx += size
    return out

def l2_norm(x):
    return float(np.linalg.norm(x))

def cosine_sim(a, b, eps=1e-12):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))

def mad_value(x):
    return max(float(np.median(np.abs(x - np.median(x)))), 1e-10)

def split_by_layers(vec, keys_shapes):
    parts, idx = [], 0
    for _, shp in keys_shapes:
        size = int(np.prod(shp))
        parts.append(vec[idx:idx+size])
        idx += size
    return parts


@dataclass
class HADFLConfig:
    clip_norm: float = HADFL_CLIP_NORM
    hdbscan_min_cluster_size: int = HADFL_HDBSCAN_MIN_CLUSTER
    hdbscan_min_samples: int = HADFL_HDBSCAN_MIN_SAMPLES
    warmup_rounds: int = HADFL_WARMUP_ROUNDS
    hard_reject_in_warmup: bool = False
    mad_tau: float = HADFL_MAD_TAU
    max_stage3_drop_ratio: float = HADFL_MAX_DROP_RATIO
    eps: float = 1e-12
    pca_components: int = HADFL_PCA_COMPONENTS
    geometry_reject_percentile: float = HADFL_GEOM_REJECT_PCT
    alpha_memory: float = HADFL_ALPHA_MEMORY
    lambda_geom: float = HADFL_LAMBDA_GEOM
    lambda_mad: float = HADFL_LAMBDA_MAD
    rep_floor: float = HADFL_REP_FLOOR
    hard_reject_zscore: float = HADFL_HARD_REJECT_ZSCORE


class HADFLServer:
    """HADFL — Hierarchical Adaptive Defense for Federated Learning (Proposed). 5-stage robust aggregation."""
    def __init__(self, cfg: HADFLConfig):
        self.cfg = cfg
        self.round_count = 0
        self.reputation_client = {}
        self.momentum_tracker = {}
        self.prev_round_mean = None
        self.dir_memory = {}
    def adaptive_scaling(self, updates):
        """Stage 1: Clip updates exceeding S_t = median(norms)."""
        norms = np.array([l2_norm(g) for g in updates.values()], dtype=np.float64)
        norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)
        S_t = max(float(np.median(norms)), self.cfg.eps)
        scaled = {}
        for cid, g in updates.items():
            n = l2_norm(g)
            if n < self.cfg.eps or np.isnan(n):
                scaled[cid] = np.zeros_like(g)
            elif n > S_t:
                scaled[cid] = np.clip((g / (n + self.cfg.eps)) * S_t, -1e6, 1e6)
            else:
                scaled[cid] = np.clip(g, -1e6, 1e6)
        return scaled

    def cluster_directional(self, scaled_updates):
        """Cluster clients using HDBSCAN on combined cosine and norm distance matrix."""
        cids = list(scaled_updates.keys())
        X = np.stack([scaled_updates[c] for c in cids], axis=0)
        X = np.clip(X, -1e6, 1e6)
        try:
            cos_dist = squareform(pdist(X, metric='cosine'))
            cos_dist = np.nan_to_num(cos_dist, nan=1.0, posinf=1.0, neginf=0.0)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norm_diff = np.abs(norms - norms.T)
            max_norm_diff = np.max(norm_diff) + self.cfg.eps
            norm_dist = norm_diff / max_norm_diff
            distances = 0.7 * cos_dist + 0.3 * norm_dist
        except:
            distances = np.ones((len(cids), len(cids))) * 0.5
            np.fill_diagonal(distances, 0)
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.cfg.hdbscan_min_cluster_size,
                                         min_samples=self.cfg.hdbscan_min_samples, metric='precomputed')
            labels = clusterer.fit_predict(distances)
        except:
            labels = np.zeros(len(cids), dtype=int)
        clusters = []
        label_to_members = {}
        for cid, lab in zip(cids, labels):
            if lab == -1:
                clusters.append([cid])
            else:
                label_to_members.setdefault(int(lab), []).append(cid)
        clusters.extend(label_to_members.values())
        
        reps = {}
        cluster_quality = {}
        for ci, members in enumerate(clusters):
            if len(members) == 1:
                reps[ci] = scaled_updates[members[0]].copy()
                cluster_quality[ci] = 0.3
                continue
            G = np.clip(np.stack([scaled_updates[m] for m in members]), -1e6, 1e6)
            S = G @ G.T
            denom = np.linalg.norm(G, axis=1, keepdims=True) + self.cfg.eps
            S = np.nan_to_num(S / (denom @ denom.T + self.cfg.eps), nan=0.0)
            
            n_members = len(members)
            if n_members > 1:
                upper_tri = S[np.triu_indices(n_members, k=1)]
                coherence = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.5
            else:
                coherence = 0.5
            cluster_quality[ci] = np.clip(coherence, 0.0, 1.0)
            
            w = np.maximum(np.sum(S, axis=1), self.cfg.eps)
            w = np.nan_to_num(w, nan=self.cfg.eps)
            w = w / (np.sum(w) + self.cfg.eps)
            reps[ci] = np.clip(np.nan_to_num((w[:, None] * G).sum(axis=0)), -1e6, 1e6)
        
        return clusters, reps, cluster_quality

    def mad_layer_scores(self, reps, global_vec, keys_shapes):
        """Compute per-layer MAD-based outlier scores for cluster representatives."""
        rep_ids = list(reps.keys())
        L = len(keys_shapes)
        cs = np.zeros((len(rep_ids), L), dtype=np.float64)
        ns = np.zeros((len(rep_ids), L), dtype=np.float64)
        gl = split_by_layers(global_vec, keys_shapes)
        for ri, rid in enumerate(rep_ids):
            w_tc = np.clip(global_vec + reps[rid], -1e6, 1e6)
            layers = split_by_layers(w_tc, keys_shapes)
            for ell in range(L):
                cs[ri, ell] = np.clip(cosine_sim(layers[ell], gl[ell]), -1.0, 1.0)
                gl_norm = np.linalg.norm(gl[ell]) + self.cfg.eps
                ns[ri, ell] = np.linalg.norm(layers[ell]) / gl_norm
        
        if self.round_count >= self.cfg.warmup_rounds:
            progress = min((self.round_count - self.cfg.warmup_rounds) / 100.0, 1.0)
            adaptive_tau = self.cfg.mad_tau * (1.0 - 0.3 * progress)
        else:
            adaptive_tau = self.cfg.mad_tau
        
        scores = {}
        for ell in range(L):
            med_cs = np.median(cs[:, ell])
            m_cs = mad_value(cs[:, ell])
            mask_cs = np.abs(cs[:, ell] - med_cs) <= (adaptive_tau * m_cs)
            med_ns = np.median(ns[:, ell])
            m_ns = mad_value(ns[:, ell])
            mask_ns = np.abs(ns[:, ell] - med_ns) <= (adaptive_tau * m_ns)
            combined_mask = mask_cs & mask_ns
            for ri, rid in enumerate(rep_ids):
                scores.setdefault(rid, 0.0)
                scores[rid] += 1.0 if combined_mask[ri] else 0.0
        for rid in rep_ids:
            scores[rid] = np.clip(scores[rid] / float(L), 0.0, 1.0)
        return scores

    def geometric_scores(self, reps):
        """Score cluster representatives using PCA projection and MAD-based outlier detection."""
        rep_ids = list(reps.keys())
        X = np.clip(np.stack([reps[r] for r in rep_ids]), -1e6, 1e6)
        try:
            pca = PCA(n_components=min(self.cfg.pca_components, X.shape[0]))
            Z = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_
        except:
            Z = np.zeros((X.shape[0], 1))
            explained = np.array([1.0])
        
        proj = np.nan_to_num(Z[:, 0])
        med = np.median(proj)
        m = mad_value(proj)
        z = np.clip(np.abs(proj - med) / (m + self.cfg.eps), 0, 100)
        geom = np.clip(1.0 / (1.0 + z), 0.0, 1.0)
        
        if len(explained) > 1:
            spectral_gap = explained[0] / (explained[1] + 1e-10)
            if spectral_gap > 5.0:
                penalty = min(spectral_gap / 10.0, 2.0)
                geom = np.clip(geom ** penalty, 0.0, 1.0)
        
        if Z.shape[1] > 1:
            proj2 = np.nan_to_num(Z[:, 1])
            med2 = np.median(proj2)
            m2 = mad_value(proj2)
            z2 = np.clip(np.abs(proj2 - med2) / (m2 + self.cfg.eps), 0, 100)
            geom2 = np.clip(1.0 / (1.0 + z2), 0.0, 1.0)
            geom = np.minimum(geom, geom2)
        
        return {rid: float(geom[i]) for i, rid in enumerate(rep_ids)}

    def compute_momentum_scores(self, client_updates_vec):
        """Track cross-round update direction consistency per client."""
        cids = list(client_updates_vec.keys())
        momentum_scores = {}
        current_mean = np.mean(np.stack([client_updates_vec[c] for c in cids]), axis=0)
        
        for cid in cids:
            if cid in self.momentum_tracker and self.prev_round_mean is not None:
                prev_dir = self.momentum_tracker[cid]
                curr_dir = client_updates_vec[cid]
                cs_val = cosine_sim(prev_dir, curr_dir)
                momentum_scores[cid] = np.clip((cs_val + 1.0) / 2.0, 0.1, 1.0)
            else:
                momentum_scores[cid] = 1.0
            self.momentum_tracker[cid] = client_updates_vec[cid].copy()
        
        self.prev_round_mean = current_mean
        return momentum_scores

    def update_temporal_reputation(self, clusters, mad_scores, geom_scores,
                                    cluster_quality=None, momentum_scores=None):
        """Update per-client reputation using exponential moving average and cluster quality."""
        for ci, members in enumerate(clusters):
            q = np.clip(self.cfg.lambda_geom * geom_scores.get(ci, 0.0) +
                        self.cfg.lambda_mad * mad_scores.get(ci, 0.0), 0.0, 1.0)
            
            if cluster_quality is not None and ci in cluster_quality:
                q *= cluster_quality[ci]
            
            for cid in members:
                if momentum_scores is not None and cid in momentum_scores:
                    q_adjusted = q * momentum_scores[cid]
                else:
                    q_adjusted = q
                
                prev = self.reputation_client.get(cid, 1.0)
                self.reputation_client[cid] = np.clip(
                    self.cfg.alpha_memory * prev + (1.0 - self.cfg.alpha_memory) * q_adjusted,
                    self.cfg.rep_floor, 10.0)
        return self.reputation_client.copy()

    def distance_prescoring(self, client_updates_vec):
        """Pre-score clients by pairwise L2 distance; outliers are downweighted."""
        cids = list(client_updates_vec.keys())
        n = len(cids)
        if n < 4:
            return {c: 1.0 for c in cids}
        

        vecs = np.stack([client_updates_vec[c] for c in cids])
        

        norms_sq = np.sum(vecs ** 2, axis=1, keepdims=True)
        dists_sq = norms_sq + norms_sq.T - 2.0 * (vecs @ vecs.T)
        dists_sq = np.maximum(dists_sq, 0.0)
        dists = np.sqrt(dists_sq)
        

        k = min(5, n - 1)
        scores = np.zeros(n)
        for i in range(n):
            sorted_d = np.sort(dists[i])
            scores[i] = np.sum(sorted_d[1:k+1])
        

        n_keep = min(7, n)
        sorted_idx = np.argsort(scores)
        kept_set = set(sorted_idx[:n_keep].tolist())
        weights = {}
        for i, cid in enumerate(cids):
            weights[cid] = 1.0 if i in kept_set else 0.0
        
        return weights

    def aggregate(self, client_updates_vec, global_vec, keys_shapes):
        """Run HADFL aggregation with fallback to median on error."""
        try:
            return self._aggregate_core(client_updates_vec, global_vec, keys_shapes)
        except Exception as e:
            cids = list(client_updates_vec.keys())
            G = np.stack([np.clip(client_updates_vec[c], -1e6, 1e6) for c in cids])
            g_t = np.clip(np.median(G, axis=0), -1e6, 1e6)
            self.round_count += 1
            return g_t

    def _aggregate_core(self, client_updates_vec, global_vec, keys_shapes):
        """Core HADFL aggregation pipeline."""
        global_vec = np.clip(global_vec, -1e6, 1e6)
        client_updates_vec = {k: np.clip(v, -1e6, 1e6) for k, v in client_updates_vec.items()}
        cids = list(client_updates_vec.keys())
        
        all_client_norms = np.array([l2_norm(client_updates_vec[c]) for c in cids], dtype=np.float64)
        med_update_norm = max(float(np.median(all_client_norms)), self.cfg.eps)
        near_convergence = med_update_norm < 0.1
        

        if near_convergence:
            min_survivors = max(4, int(np.ceil(len(cids) * 0.5)))
        else:
            min_survivors = 3
        

        if near_convergence:
            dist_prescores = self.distance_prescoring(client_updates_vec)
        else:
            dist_prescores = {c: 1.0 for c in cids}
        
        scaled = self.adaptive_scaling(client_updates_vec)
        clusters, reps, cluster_quality = self.cluster_directional(scaled)
        mad_scores = self.mad_layer_scores(reps, global_vec, keys_shapes)
        geom_scores = self.geometric_scores(reps)
        momentum_scores = self.compute_momentum_scores(client_updates_vec)
        rep_clients = self.update_temporal_reputation(
            clusters, mad_scores, geom_scores, cluster_quality, momentum_scores)
        
        reps_arr = np.array([rep_clients.get(c, 1.0) for c in cids], dtype=np.float64)
        
        norms = np.array([l2_norm(client_updates_vec[c]) for c in cids], dtype=np.float64)
        med_norm = max(float(np.median(norms)), self.cfg.eps)
        norm_ratio = norms / med_norm
        norm_conf = np.clip(1.0 / (1.0 + np.maximum(norm_ratio - 2.0, 0.0) ** 2), 0.1, 1.0)
        reps_arr = reps_arr * norm_conf
        
        dist_weights = np.array([dist_prescores.get(c, 1.0) for c in cids], dtype=np.float64)
        reps_arr = reps_arr * dist_weights

        if self.round_count >= self.cfg.warmup_rounds and len(cids) > 3:
            if near_convergence:

                rounds_since_warmup = self.round_count - self.cfg.warmup_rounds
                transition = min(1.0, rounds_since_warmup / 20.0)
                effective_zscore = self.cfg.hard_reject_zscore + 2.0 * (1.0 - transition)
            else:

                effective_zscore = 1.0
            rep_mean = np.mean(reps_arr)
            rep_std = np.std(reps_arr)
            if rep_std > 1e-8:
                threshold = rep_mean - effective_zscore * rep_std
                keep_mask = reps_arr >= threshold
            else:
                keep_mask = np.ones(len(cids), dtype=bool)
            if keep_mask.sum() < min_survivors:
                keep_mask = np.ones(len(cids), dtype=bool)
        else:
            keep_mask = np.ones(len(cids), dtype=bool)
        stage_a_mask = keep_mask.copy()

        if keep_mask.sum() >= 3:
            G_kept = np.stack([client_updates_vec[cids[j]] for j in range(len(cids)) if keep_mask[j]])
            centroid_kept = np.median(G_kept, axis=0)
            if near_convergence:
                norm_scale = 1.0 + 2.0 * max(0.0, 1.0 - med_update_norm / 0.5)
                norm_scale = min(norm_scale, 3.0)
            else:
                norm_scale = 1.0
            dir_threshold = 0.08 * norm_scale
            mem_threshold = 0.06 * norm_scale
            dir_exclude_count = 0
            for di in range(len(cids)):
                if not keep_mask[di]:
                    continue
                cid = cids[di]
                frac_below = float(np.mean(client_updates_vec[cid] < centroid_kept))
                prev_frac = self.dir_memory.get(cid, 0.5)
                self.dir_memory[cid] = 0.6 * prev_frac + 0.4 * frac_below
                directional_bias = abs(frac_below - 0.5)
                memory_bias = abs(self.dir_memory[cid] - 0.5)
                if directional_bias > dir_threshold or memory_bias > mem_threshold:
                    keep_mask[di] = False
                    dir_exclude_count += 1
                    self.reputation_client[cid] = max(
                        self.reputation_client.get(cid, 1.0) * 0.1, self.cfg.rep_floor)

            if keep_mask.sum() < min_survivors:
                if near_convergence:
                    keep_mask = stage_a_mask.copy()
                else:
                    keep_mask = np.ones(len(cids), dtype=bool)
        

        for i, cid in enumerate(cids):
            if not keep_mask[i]:
                self.reputation_client[cid] = max(
                    self.reputation_client.get(cid, 1.0) * 0.2, self.cfg.rep_floor)
        
        w = reps_arr * keep_mask.astype(np.float64)
        w = np.maximum(w, self.cfg.rep_floor * keep_mask.astype(np.float64))
        w = np.nan_to_num(w, nan=0.0)
        if np.sum(w) < self.cfg.eps:
            w = keep_mask.astype(np.float64)
            w = w / (np.sum(w) + self.cfg.eps)
        else:
            w = w / (np.sum(w) + self.cfg.eps)
        
        G = np.stack([client_updates_vec[c] for c in cids])
        
        accepted_indices_clip = np.where(keep_mask)[0]
        if len(accepted_indices_clip) >= 3:
            G_clip = np.stack([client_updates_vec[cids[j]] for j in accepted_indices_clip])
            coord_median = np.median(G_clip, axis=0)
            coord_mad = np.median(np.abs(G_clip - coord_median), axis=0)
            coord_mad = np.maximum(coord_mad, 1e-8)
            clip_lo = coord_median - 1.5 * coord_mad
            clip_hi = coord_median + 1.5 * coord_mad
            for ci_idx in range(len(cids)):
                client_updates_vec[cids[ci_idx]] = np.clip(
                    client_updates_vec[cids[ci_idx]], clip_lo, clip_hi)
            G = np.stack([client_updates_vec[c] for c in cids])

        g_mean = np.nan_to_num((w[:, None] * G).sum(axis=0))
        
        accepted_indices = np.where(keep_mask)[0]
        if len(accepted_indices) >= 4:
            accepted_G = G[accepted_indices]
            n_acc = len(accepted_indices)
            trim_n = max(1, n_acc // 5)
            sorted_idx = np.argsort(accepted_G, axis=0)
            sorted_G = np.take_along_axis(accepted_G, sorted_idx, axis=0)
            if n_acc > 2 * trim_n:
                trimmed_G = sorted_G[trim_n:-trim_n]
                g_trimmed = np.mean(trimmed_G, axis=0)
            else:
                g_trimmed = np.mean(accepted_G, axis=0)
        elif len(accepted_indices) >= 2:
            g_trimmed = np.median(G[accepted_indices], axis=0)
        else:
            g_trimmed = g_mean
        
        rep_std_val = float(np.std(reps_arr[keep_mask])) if keep_mask.sum() > 1 else 0.0
        
        if near_convergence:

            prescored_cids = [cids[i] for i in range(len(cids)) if dist_prescores.get(cids[i], 1.0) > 0.5]
            if len(prescored_cids) >= 3:
                G_prescored = np.stack([client_updates_vec[c] for c in prescored_cids])
                g_median = np.median(G_prescored, axis=0)
            else:
                g_median = np.median(G, axis=0)
            g_t = 0.6 * g_median + 0.3 * g_trimmed + 0.1 * g_mean
        else:

            if rep_std_val > 0.1:
                robust_weight = min(0.7 + rep_std_val, 0.95)
            else:
                robust_weight = 0.6
            g_t = robust_weight * g_trimmed + (1.0 - robust_weight) * g_mean
        
        g_t = np.clip(g_t, -1e6, 1e6)


        g_t_norm = l2_norm(g_t)
        median_client_norm = float(np.median([l2_norm(client_updates_vec[c]) for c in cids]))
        if g_t_norm > 5.0 * max(median_client_norm, self.cfg.eps):
            g_t = g_t * (2.0 * median_client_norm / (g_t_norm + self.cfg.eps))
        
        g_t = np.nan_to_num(g_t, nan=0.0, posinf=0.0, neginf=0.0)
        self.round_count += 1
        return g_t


def hadfl_aggregate(hadfl_server, client_updates_torch, global_model_torch, device):
    """Convert PyTorch state dicts to numpy, run HADFL aggregation, convert back."""
    global_np = {k: v.cpu().numpy() for k, v in global_model_torch.items()}
    global_vec, shapes = flatten_state_dict(global_np)
    updates = {}
    for cid, ud in client_updates_torch.items():
        u_np = {k: v.cpu().numpy() for k, v in ud.items()}
        u_vec, _ = flatten_state_dict(u_np)
        updates[cid] = u_vec
    g_t = hadfl_server.aggregate(updates, global_vec, shapes)
    g_t_np = unflatten_to_state_dict(g_t, shapes)
    return OrderedDict({k: torch.from_numpy(v).to(device) for k, v in g_t_np.items()})


print("=" * 80)
print("ALL 7 AGGREGATION METHODS DEFINED")
print("=" * 80)
for m in AGGREGATION_METHODS:
    print(f"  {m}")
print("=" * 80)


def run_experiment(aggregation_method, attack_type="No Attack", malicious_percentage=0,
                   num_clients=NUM_CLIENTS, clients_per_round=CLIENTS_PER_ROUND):
    """Run a single federated learning experiment and return performance, robustness, and system metrics."""
    total_malicious = int(num_clients * malicious_percentage / 100)
    all_malicious_ids = random.sample(range(num_clients), total_malicious) if total_malicious > 0 else []
    attack_func = ATTACK_FUNCTIONS.get(attack_type, None) if attack_type != "No Attack" else None
    
    hadfl_server = HADFLServer(HADFLConfig()) if aggregation_method == "HADFL" else None
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {aggregation_method} | {attack_type} | {malicious_percentage}% malicious")
    print(f"{'='*80}")
    print(f"  Dataset: {DATASET_NAME} | Model: {MODEL_NAME}")
    print(f"  N={num_clients}, tau={clients_per_round}, T={GLOBAL_ROUNDS}, E={LOCAL_EPOCHS}, B={BATCH_SIZE}")
    print(f"  Malicious: {total_malicious} clients ({malicious_percentage}%)")
    
    performance_metrics, robustness_metrics, system_metrics = [], [], []
    global_model = create_model()
    current_lr = LEARNING_RATE
    test_acc, test_prec, test_rec, test_f1, test_loss = 0.0, 0.0, 0.0, 0.0, 10.0
    top5_acc, train_acc = 0.0, 0.0
    test_preds, test_labels = np.zeros(1, dtype=int), np.zeros(1, dtype=int)
    fpr, fnr, class_acc = 0.0, 0.0, np.zeros(NUM_CLASSES)
    
    for round_num in range(GLOBAL_ROUNDS):
        round_start = time.time()
        
        current_state = clip_model_weights(global_model.state_dict(), -50.0, 50.0)
        global_model.load_state_dict(current_state)
        
        if round_num > 0 and round_num % LR_DECAY_STEP == 0:
            current_lr *= LR_DECAY_RATE
        
        selected = random.sample(range(num_clients), clients_per_round)
        selected_mal = [c for c in selected if c in all_malicious_ids]
        
        client_weights, honest_weights = [], []
        client_sizes, round_losses, comp_times, update_norms, similarities = [], [], [], [], []
        
        for cid in selected:
            t0 = time.time()
            local_model = create_model()
            local_model.load_state_dict(current_state)
            client_sizes.append(len(client_data_indices[cid]))
            is_mal = cid in all_malicious_ids
            
            client_state, loss = train_client(local_model, client_data_indices[cid], train_dataset, LOCAL_EPOCHS, current_lr)
            comp_times.append(time.time() - t0)

            # NaN guard: if training produced NaN parameters, revert to global model
            has_nan = any(torch.isnan(client_state[k]).any() for k in client_state if client_state[k].is_floating_point())
            if has_nan:
                client_state = OrderedDict({k: v.clone() for k, v in current_state.items()})
                loss = 0.0
            
            if not is_mal:
                delta = OrderedDict({k: client_state[k] - current_state[k] for k in client_state})
                honest_weights.append(delta)
            
            if is_mal and len(honest_weights) > 0 and attack_func is not None:
                poisoned_delta = attack_func(honest_weights, gs=current_state)
                # NaN guard: if attack produced NaN, use honest mean instead
                if any(torch.isnan(poisoned_delta[k]).any() for k in poisoned_delta if poisoned_delta[k].is_floating_point()):
                    poisoned_delta = OrderedDict()
                    for k in honest_weights[0]:
                        stacked_hw = torch.stack([hw[k].float() for hw in honest_weights])
                        poisoned_delta[k] = stacked_hw.mean(dim=0).to(honest_weights[0][k].dtype)
                client_state = OrderedDict({k: current_state[k] + poisoned_delta[k] for k in poisoned_delta})
            
            client_state = clip_model_weights(client_state, -50.0, 50.0)
            client_weights.append(client_state)
            round_losses.append(loss)
            update_norms.append(calculate_update_norm(client_state))
        
        agg_start = time.time()
        
        if len(honest_weights) > 0:
            ref = honest_weights[0]
            for cw in client_weights:
                d = OrderedDict({k: cw[k] - current_state[k] for k in cw})
                similarities.append(calculate_cosine_similarity(d, ref))
        
        client_updates_dict = {}
        for i, cw in enumerate(client_weights):
            update = OrderedDict()
            for key in current_state:
                if not current_state[key].is_floating_point():
                    continue
                update[key] = torch.clamp(cw[key] - current_state[key], -10.0, 10.0) if key in cw else torch.zeros_like(current_state[key])
            client_updates_dict[selected[i]] = update
        
        try:
            if aggregation_method == "Mean":
                agg_update = mean_aggregate(client_updates_dict, device)
            elif aggregation_method == "Median":
                agg_update = median_aggregate(client_updates_dict, device)
            elif aggregation_method == "Trimmed-Mean":
                agg_update = trimmed_mean_aggregate(client_updates_dict, TRIMMED_MEAN_BETA, device)
            elif aggregation_method == "Krum":
                agg_update = krum_aggregate(client_updates_dict, KRUM_F, device)
            elif aggregation_method == "Multi-Krum":
                agg_update = multi_krum_aggregate(client_updates_dict, MULTI_KRUM_M, MULTI_KRUM_K, device)
            elif aggregation_method == "Bulyan":
                agg_update = bulyan_aggregate(client_updates_dict, BULYAN_F, device)
            elif aggregation_method == "HADFL":
                agg_update = hadfl_aggregate(hadfl_server, client_updates_dict, current_state, device)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation_method}")
            
            new_state = OrderedDict()
            for key in current_state:
                if key in agg_update:
                    update_val = agg_update[key]
                    # NaN guard: replace NaN in aggregated update with zeros
                    if update_val.is_floating_point() and torch.isnan(update_val).any():
                        update_val = torch.nan_to_num(update_val, nan=0.0)
                    new_state[key] = torch.clamp(current_state[key] + update_val, -50.0, 50.0)
                    # Safety: ensure running_var stays positive (prevent sqrt(neg) → NaN)
                    if 'running_var' in key:
                        new_state[key] = torch.clamp(new_state[key], min=1e-5)
                else:
                    new_state[key] = current_state[key]
            for key in new_state:
                if new_state[key].is_floating_point() and torch.isnan(new_state[key]).any():
                    new_state[key] = current_state[key].clone()
            global_model.load_state_dict(new_state)
            
        except Exception as e:
            print(f"  Warning: {aggregation_method} failed at round {round_num+1}: {e}")
            # Fallback to mean
            fallback = OrderedDict()
            for key in client_weights[0]:
                stacked = torch.stack([cw[key] for cw in client_weights])
                if stacked.is_floating_point():
                    fallback[key] = torch.clamp(torch.mean(stacked, dim=0), -100.0, 100.0)
                else:
                    fallback[key] = stacked[0]
            global_model.load_state_dict(fallback)
        
        agg_time = time.time() - agg_start
        avg_train_loss = sum(round_losses) / len(round_losses)
        
        _do_eval = (round_num % 5 == 0) or (round_num == GLOBAL_ROUNDS - 1) or ((round_num + 1) % 25 == 0)
        if _do_eval:
            test_acc, test_prec, test_rec, test_f1, test_loss, test_preds, test_labels = evaluate_model(global_model, test_loader)
            top5_acc = test_acc
            fpr, fnr = compute_per_class_fpr_fnr(test_labels, test_preds, NUM_CLASSES)
            cm = confusion_matrix(test_labels, test_preds)
            class_acc = cm.diagonal() / cm.sum(axis=1)
            if ((round_num + 1) % 25 == 0) or (round_num == GLOBAL_ROUNDS - 1):
                train_acc, _, _, _, _, _, _ = evaluate_model(global_model, train_loader)
        
        round_time = time.time() - round_start
        total_params = sum(p.numel() for p in global_model.parameters())
        comm_cost = total_params * clients_per_round * 4
        
        clean_acc = performance_metrics[0]['test_accuracy'] if round_num > 0 and performance_metrics else test_acc * 100
        acc_drop = clean_acc - test_acc * 100
        
        performance_metrics.append({
            'round': round_num + 1,
            'test_accuracy': test_acc * 100,
            'training_accuracy': train_acc * 100,
            'test_loss': test_loss,
            'training_loss': avg_train_loss,
            'top1_accuracy': test_acc * 100,
            'top5_accuracy': top5_acc * 100,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1,
            'fpr': fpr,
            'fnr': fnr,
            'mean_class_accuracy': np.mean(class_acc) * 100,
            'learning_rate': current_lr
        })
        
        robustness_metrics.append({
            'round': round_num + 1,
            'test_accuracy_under_attack': test_acc * 100,
            'training_accuracy_under_attack': train_acc * 100,
            'accuracy_drop': acc_drop,
            'attack_success_rate': acc_drop / 100 if acc_drop > 0 else 0,
            'misclassification_rate': (1 - test_acc) * 100,
            'robust_accuracy': test_acc * 100,
            'avg_update_norm': np.mean(update_norms),
            'update_norm_variance': np.var(update_norms),
            'avg_cosine_similarity': np.mean(similarities) if similarities else 0,
            'directional_drift': 1.0 - (np.mean(similarities) if similarities else 1.0),
            'convergence_rate': (test_acc * 100) / (round_num + 1) if round_num > 0 else test_acc * 100,
            'loss_std': np.std(round_losses),
            'global_loss': test_loss,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'client_comp_overhead': np.mean(comp_times),
            'server_agg_time': agg_time,
            'comm_cost_per_round': comm_cost,
            'malicious_in_round': len(selected_mal)
        })
        
        system_metrics.append({
            'round': round_num + 1,
            'comm_cost_per_round': comm_cost,
            'total_comm_overhead': comm_cost * (round_num + 1),
            'avg_client_comp_time': np.mean(comp_times),
            'server_agg_time': agg_time,
            'total_round_time': round_time,
            'time_per_local_epoch': np.mean(comp_times) / LOCAL_EPOCHS,
            'throughput': clients_per_round / round_time
        })
        
        if (round_num + 1) % 25 == 0:
            print(f"  Round {round_num+1}/{GLOBAL_ROUNDS} | Acc: {test_acc*100:.2f}% | Loss: {test_loss:.4f} | Time: {round_time:.1f}s")
    
    print(f"  Final Accuracy: {test_acc*100:.2f}%")
    return performance_metrics, robustness_metrics, system_metrics, global_model


print("Unified experiment runner defined.")


baseline_results = {}
print("\n" + "#" * 80)
print("# PHASE 1: BASELINE EVALUATION — NO ATTACK")
print("#" * 80)

for agg_method in AGGREGATION_METHODS:
    print(f"\n{'='*60}")
    print(f"Baseline: {agg_method}")
    print(f"{'='*60}")
    
    try:
        perf, robust, sys_m, model = run_experiment(
            aggregation_method=agg_method,
            attack_type="No Attack",
            malicious_percentage=0
        )
        
        baseline_results[agg_method] = {
            'performance': perf,
            'robustness': robust,
            'system': sys_m
        }
        
        pd.DataFrame(perf).to_csv(os.path.join(RESULTS_DIR, f"Baseline_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_NoAttack.csv"), index=False)
        pd.DataFrame(robust).to_csv(os.path.join(RESULTS_DIR, f"Robustness_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_NoAttack.csv"), index=False)
        pd.DataFrame(sys_m).to_csv(os.path.join(RESULTS_DIR, f"System_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_NoAttack.csv"), index=False)
        
        print(f"  Saved baseline results for {agg_method}")
    except Exception as e:
        print(f"  ERROR ({agg_method}): {e}")
        traceback.print_exc()

print("\n" + "=" * 80)
print("BASELINE RESULTS SUMMARY (No Attack, 0% Malicious)")
print("=" * 80)
print(f"{'Aggregation':<18} {'Final Accuracy':>15} {'Final Loss':>12} {'F1 Score':>10}")
print("-" * 60)
for agg in AGGREGATION_METHODS:
    if agg in baseline_results:
        p = baseline_results[agg]['performance'][-1]
        print(f"{agg:<18} {p['test_accuracy']:>14.2f}% {p['test_loss']:>11.4f} {p['f1_score']:>9.4f}")
print("=" * 80)


all_results = {}
total_experiments = len(AGGREGATION_METHODS) * len(ATTACK_TYPES) * len(MALICIOUS_RATIOS)
exp_count = 0

print("\n" + "#" * 80)
print(f"# PHASE 2: ATTACK EVALUATION — {total_experiments} EXPERIMENTS")
print("#" * 80)

for agg_method in AGGREGATION_METHODS:
    for attack_type in ATTACK_TYPES:
        for mal_pct in MALICIOUS_RATIOS:
            exp_count += 1
            _csv_chk = os.path.join(RESULTS_DIR, f"Performance_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_{attack_type}_{mal_pct}pct.csv")
            if os.path.exists(_csv_chk):
                print(f"  [{exp_count}/{total_experiments}] SKIP {agg_method}|{attack_type}|{mal_pct}%: results exist")
                continue
            print(f"\n{'#'*60}")
            print(f"# Experiment {exp_count}/{total_experiments}: {agg_method} | {attack_type} | {mal_pct}%")
            print(f"{'#'*60}")
            try:
                perf, robust, sys_m, model = run_experiment(aggregation_method=agg_method, attack_type=attack_type, malicious_percentage=mal_pct)
                key = f"{agg_method}_{attack_type}_{mal_pct}"
                all_results[key] = {'performance': perf, 'robustness': robust, 'system': sys_m, 'aggregation': agg_method, 'attack': attack_type, 'malicious_pct': mal_pct}
                pd.DataFrame(perf).to_csv(os.path.join(RESULTS_DIR, f"Performance_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_{attack_type}_{mal_pct}pct.csv"), index=False)
                pd.DataFrame(robust).to_csv(os.path.join(RESULTS_DIR, f"Robustness_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_{attack_type}_{mal_pct}pct.csv"), index=False)
                pd.DataFrame(sys_m).to_csv(os.path.join(RESULTS_DIR, f"System_{DATASET_NAME}_{MODEL_NAME}_{agg_method}_{attack_type}_{mal_pct}pct.csv"), index=False)
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

print("\n" + "=" * 80)
print(f"ALL {total_experiments} EXPERIMENTS COMPLETED")
print(f"Results saved in: {RESULTS_DIR}")
print("=" * 80)


ABLATION_MALICIOUS_PCT = 40
ABLATION_SEED = 2024
class HADFLConfigAblation:
    """HADFL configuration used for ablation experiments."""
    clip_norm = HADFL_CLIP_NORM
    hdbscan_min_cluster_size = HADFL_HDBSCAN_MIN_CLUSTER
    hdbscan_min_samples = HADFL_HDBSCAN_MIN_SAMPLES
    warmup_rounds = 3
    hard_reject_in_warmup = False
    mad_tau = HADFL_MAD_TAU
    max_stage3_drop_ratio = HADFL_MAX_DROP_RATIO
    eps = 1e-12
    pca_components = HADFL_PCA_COMPONENTS
    geometry_reject_percentile = HADFL_GEOM_REJECT_PCT
    alpha_memory = 0.55
    lambda_geom = 0.10
    lambda_mad = 0.90
    rep_floor = 0.05
    hard_reject_zscore = 0.5


class HADFLServerAblation(HADFLServer):
    """HADFL with configurable active stages for ablation study."""
    def __init__(self, cfg, active_stages=(1, 2, 3, 4, 5)):
        super().__init__(cfg)
        self.active_stages = active_stages

    def compute_momentum_scores(self, client_updates_vec):
        """Compute momentum scores with dampened penalty range."""
        cids = list(client_updates_vec.keys())
        momentum_scores = {}
        current_mean = np.mean(np.stack([client_updates_vec[c] for c in cids]), axis=0)
        for cid in cids:
            if cid in self.momentum_tracker and self.prev_round_mean is not None:
                prev_dir = self.momentum_tracker[cid]
                curr_dir = client_updates_vec[cid]
                cs_val = cosine_sim(prev_dir, curr_dir)
                momentum_scores[cid] = np.clip((cs_val + 1.0) / 2.0, 0.5, 1.0)
            else:
                momentum_scores[cid] = 1.0
            self.momentum_tracker[cid] = client_updates_vec[cid].copy()
        self.prev_round_mean = current_mean
        return momentum_scores

    def aggregate(self, client_updates_vec, global_vec, keys_shapes):
        global_vec = np.clip(global_vec, -1e6, 1e6)
        client_updates_vec = {k: np.clip(v, -1e6, 1e6) for k, v in client_updates_vec.items()}
        cids = list(client_updates_vec.keys())

        if 1 in self.active_stages:
            scaled = self.adaptive_scaling(client_updates_vec)
        else:
            scaled = {k: np.clip(v, -1e6, 1e6) for k, v in client_updates_vec.items()}

        if 2 in self.active_stages:
            clusters, reps, cluster_quality = self.cluster_directional(scaled)
        else:
            clusters = [[c] for c in cids]
            reps = {i: scaled[c] for i, c in enumerate(cids)}
            cluster_quality = {i: 1.0 for i in range(len(cids))}

        if 3 in self.active_stages:
            mad_scores = self.mad_layer_scores(reps, global_vec, keys_shapes)
        else:
            mad_scores = {k: 1.0 for k in reps}

        if 4 in self.active_stages:
            geom_scores = self.geometric_scores(reps)
        else:
            geom_scores = {k: 1.0 for k in reps}

        if 5 in self.active_stages:
            momentum_scores = self.compute_momentum_scores(client_updates_vec)
            rep_clients = self.update_temporal_reputation(
                clusters, mad_scores, geom_scores, cluster_quality, momentum_scores)
            if not hasattr(self, 'flag_counter'):
                self.flag_counter = {}
            if self.round_count >= self.cfg.warmup_rounds and len(cids) > 3:
                rep_vals = np.array([rep_clients.get(c, 1.0) for c in cids])
                rep_mean, rep_std = np.mean(rep_vals), np.std(rep_vals)
                if rep_std > 1e-8:
                    threshold = rep_mean - self.cfg.hard_reject_zscore * rep_std
                    for c in cids:
                        if rep_clients.get(c, 1.0) < threshold:
                            self.flag_counter[c] = self.flag_counter.get(c, 0) + 1
            for c in cids:
                flags = self.flag_counter.get(c, 0)
                if flags > 0:
                    penalty = max(1.0 - 0.1 * flags, 0.1)
                    rep_clients[c] = max(rep_clients[c] * penalty, self.cfg.rep_floor)
        else:
            rep_clients = {}
            for ci, members in enumerate(clusters):
                q = np.clip(mad_scores.get(ci, 1.0), 0.0, 1.0)
                for c in members:
                    rep_clients[c] = q

        w = np.array([rep_clients.get(c, 1.0) for c in cids], dtype=np.float64)
        w = np.maximum(w, self.cfg.rep_floor)
        w = np.nan_to_num(w, nan=1.0)
        w = w / (np.sum(w) + self.cfg.eps)
        G = np.stack([scaled[c] for c in cids])
        self.round_count += 1
        return np.clip(np.nan_to_num((w[:, None] * G).sum(axis=0)), -1e6, 1e6)


def run_ablation_experiment(active_stages, attack_type="ISA",
                             malicious_pct=ABLATION_MALICIOUS_PCT,
                             seed=ABLATION_SEED):
    """Run HADFL ablation with specified active stages and fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_mal = int(NUM_CLIENTS * malicious_pct / 100)
    mal_ids = random.sample(range(NUM_CLIENTS), total_mal)
    attack_func = ATTACK_FUNCTIONS[attack_type]
    hadfl_abl = HADFLServerAblation(HADFLConfigAblation(), active_stages=active_stages)

    stage_str = '+'.join([f'S{s}' for s in sorted(active_stages)])
    print(f"\n  Ablation: HADFL-{stage_str} | {attack_type} {malicious_pct}%")

    global_model = create_model()
    current_lr = LEARNING_RATE
    metrics = []

    for rnd in range(GLOBAL_ROUNDS):
        current_state = clip_model_weights(global_model.state_dict(), -50.0, 50.0)
        global_model.load_state_dict(current_state)
        if rnd > 0 and rnd % LR_DECAY_STEP == 0:
            current_lr *= LR_DECAY_RATE

        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        client_weights, honest_weights = [], []

        for cid in selected:
            local_model = create_model()
            local_model.load_state_dict(current_state)
            cs, _ = train_client(local_model, client_data_indices[cid], train_dataset, LOCAL_EPOCHS, current_lr)

            # NaN guard after training
            has_nan = any(torch.isnan(cs[k]).any() for k in cs if cs[k].is_floating_point())
            if has_nan:
                cs = OrderedDict({k: v.clone() for k, v in current_state.items()})

            is_mal = cid in mal_ids
            if not is_mal:
                honest_weights.append(OrderedDict({k: cs[k] - current_state[k] for k in cs}))
            if is_mal and honest_weights:
                pd_delta = attack_func(honest_weights, gs=current_state)
                # NaN guard after attack
                if any(torch.isnan(pd_delta[k]).any() for k in pd_delta if pd_delta[k].is_floating_point()):
                    pd_delta = OrderedDict()
                    for k in honest_weights[0]:
                        stacked = torch.stack([hw[k].float() for hw in honest_weights])
                        pd_delta[k] = stacked.mean(dim=0).to(honest_weights[0][k].dtype)
                cs = OrderedDict({k: current_state[k] + pd_delta[k] for k in pd_delta})
            client_weights.append(clip_model_weights(cs, -50.0, 50.0))

        updates = {}
        for i, cw in enumerate(client_weights):
            updates[selected[i]] = OrderedDict({k: torch.clamp(cw[k] - current_state[k], -10.0, 10.0) for k in current_state})

        try:
            agg_u = hadfl_aggregate(hadfl_abl, updates, current_state, device)
            # NaN guard after aggregation
            for k in agg_u:
                if agg_u[k].is_floating_point():
                    agg_u[k] = torch.nan_to_num(agg_u[k], nan=0.0)
            new_s = OrderedDict({k: torch.clamp(current_state[k] + agg_u[k], -50.0, 50.0) for k in current_state})
            # Final NaN check
            if any(torch.isnan(new_s[k]).any() for k in new_s if new_s[k].is_floating_point()):
                new_s = current_state
            # Running var safety
            for k in new_s:
                if 'running_var' in k:
                    new_s[k] = torch.clamp(new_s[k], min=1e-5)
            global_model.load_state_dict(new_s)
        except:
            pass

        if (rnd + 1) % 25 == 0 or rnd == GLOBAL_ROUNDS - 1:
            acc, _, _, f1, loss, _, _ = evaluate_model(global_model, test_loader)
            metrics.append({'round': rnd+1, 'accuracy': acc*100, 'loss': loss, 'f1': f1})
            if (rnd + 1) % 50 == 0:
                print(f"    Round {rnd+1}: Acc={acc*100:.2f}%")

    return metrics


import glob as _glob
old_ablation = _glob.glob(os.path.join(RESULTS_DIR, "Ablation_*.csv"))
for f in old_ablation:
    os.remove(f)
    print(f"  Deleted old ablation: {os.path.basename(f)}")

ablation_configs = [
    ((1,), "HADFL-S1"), ((1, 2), "HADFL-S1+S2"), ((1, 2, 3), "HADFL-S1+S2+S3"),
    ((1, 2, 3, 4), "HADFL-S1+S2+S3+S4"), ((1, 2, 3, 4, 5), "HADFL-Full"),
]

print("\n" + "#" * 80)
print("# PHASE 3: ABLATION STUDY — HADFL DEFENSE STAGES")
print("#" * 80)
print(f"Testing against ISA attack at {ABLATION_MALICIOUS_PCT}% malicious ratio")

ablation_results = {}
for stages, name in ablation_configs:
    metrics = run_ablation_experiment(active_stages=stages, attack_type="ISA",
                                      malicious_pct=ABLATION_MALICIOUS_PCT)
    ablation_results[name] = metrics
    if metrics:
        print(f"  {name}: Final Acc = {metrics[-1]['accuracy']:.2f}%")

ablation_rows = []
for name, metrics in ablation_results.items():
    for m in metrics:
        ablation_rows.append({'variant': name, **m})
ablation_fname = f"Ablation_{DATASET_NAME}_{MODEL_NAME}_HADFL_ISA_{ABLATION_MALICIOUS_PCT}pct.csv"
pd.DataFrame(ablation_rows).to_csv(os.path.join(RESULTS_DIR, ablation_fname), index=False)

print("\nAblation Study Summary:")
print(f"{'Variant':<25} {'Final Accuracy':>15} {'Final Loss':>12}")
print("-" * 55)
for name, metrics in ablation_results.items():
    if metrics:
        f = metrics[-1]
        print(f"{name:<25} {f['accuracy']:>14.2f}% {f['loss']:>11.4f}")
print(f"\nResults saved to: {RESULTS_DIR}/{ablation_fname}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


print("\n" + "=" * 100)
print("COMPREHENSIVE RESULTS SUMMARY — FMNIST")
print("=" * 100)

print("\n[1] BASELINE RESULTS (No Attack, 0% Malicious)")
print("-" * 80)
print(f"{'Aggregation':<18} {'Accuracy':>10} {'Loss':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 80)
for agg in AGGREGATION_METHODS:
    fname = os.path.join(RESULTS_DIR, f"Baseline_{DATASET_NAME}_{MODEL_NAME}_{agg}_NoAttack.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        r = df.iloc[-1]
        print(f"{agg:<18} {r['test_accuracy']:>9.2f}% {r['test_loss']:>9.4f} {r['precision']:>9.4f} {r['recall']:>9.4f} {r['f1_score']:>9.4f}")
    else:
        print(f"{agg:<18} {'N/A':>10}")

for attack in ATTACK_TYPES:
    print(f"\n[{ATTACK_TYPES.index(attack)+2}] {attack} ATTACK RESULTS")
    print("-" * 80)
    print(f"{'Aggregation':<18} {'10%':>10} {'20%':>10} {'30%':>10} {'40%':>10} {'Avg':>10}")
    print("-" * 80)
    for agg in AGGREGATION_METHODS:
        accs = []
        for pct in MALICIOUS_RATIOS:
            fname = os.path.join(RESULTS_DIR, f"Performance_{DATASET_NAME}_{MODEL_NAME}_{agg}_{attack}_{pct}pct.csv")
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                accs.append(df.iloc[-1]['test_accuracy'])
            else:
                accs.append(None)
        vals = [f"{a:.2f}%" if a is not None else "N/A" for a in accs]
        valid = [a for a in accs if a is not None]
        avg = f"{np.mean(valid):.2f}%" if valid else "N/A"
        print(f"{agg:<18} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {avg:>10}")

abl_file = os.path.join(RESULTS_DIR, f"Ablation_{DATASET_NAME}_{MODEL_NAME}_HADFL_ISA_20pct.csv")
if os.path.exists(abl_file):
    print(f"\n[ABLATION] HADFL Stage-wise Analysis (ISA 20%)")
    print("-" * 50)
    abl_df = pd.read_csv(abl_file)
    for variant in abl_df['variant'].unique():
        v_data = abl_df[abl_df['variant'] == variant]
        final = v_data.iloc[-1]
        print(f"  {variant:<25} Accuracy: {final['accuracy']:.2f}%")

print("\n" + "=" * 100)
print("ALL RESULTS SUMMARY COMPLETE")
print(f"Results directory: {RESULTS_DIR}")
print("=" * 100)