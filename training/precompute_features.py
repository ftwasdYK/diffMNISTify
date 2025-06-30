from models import CustomResNet50, FeatureExtractor
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from utils import ChannelRepeat, torch_train_val_split, prepare_data
from models import CustomResNet50, FeatureExtractor
import numpy as np

####
BATCH_SIZE = 1
NN_MODEL = CustomResNet50()
NUM_LAYERS_EXPORT = 9
####

dirs = ['train', 'val', 'test']
for dir in dirs:
     if not os.path.exists(f"./data/precompute/{dir}"):
         os.makedirs(f"./data/precompute/{dir}")

transform = transforms.Compose([
        transforms.ToTensor(),
        ChannelRepeat(), 
        NN_MODEL.get_transformation(),
    ])

train_loader, val_loader, test_loader = prepare_data(
    transform_tr=transform,
    transform_infer=transform,
    batch_size=BATCH_SIZE
)

NN_MODEL = FeatureExtractor(model=resnet50(weights=ResNet50_Weights.DEFAULT), num_layers=NUM_LAYERS_EXPORT)

print('Started')
for i, (x, y) in enumerate(train_loader):
    feature_x = NN_MODEL(x)
    y = y.numpy()
    np.save(f'data/precompute/train/x{NUM_LAYERS_EXPORT}_{i}.npy', feature_x)
    np.save(f'data/precompute/train/y{NUM_LAYERS_EXPORT}_{i}.npy', y)
print('Train features computed.')
for i, (x, y) in enumerate(val_loader):
    feature_x = NN_MODEL(x)
    y = y.numpy()
    np.save(f'data/precompute/val/x{NUM_LAYERS_EXPORT}_{i}.npy', feature_x)
    np.save(f'data/precompute/val/y{NUM_LAYERS_EXPORT}_{i}.npy', y)
print('Val features computed.')
for i, (x, y) in enumerate(test_loader):
    if (i % 100)==0:
        print(i)
    feature_x = NN_MODEL(x)
    y = y.numpy()
    np.save(f'data/precompute/test/x{NUM_LAYERS_EXPORT}_{i}.npy', feature_x)
    np.save(f'data/precompute/test/y{NUM_LAYERS_EXPORT}_{i}.npy', y)
print('Test features computed.')