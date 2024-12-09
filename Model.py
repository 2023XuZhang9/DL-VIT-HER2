# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Early_Stopping import EarlyStopping
from Dataloader import gen_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

class CustomDataset(Dataset):
    def __init__(self, img_tensors, labels, ids, transform=None):
        self.img_tensors = img_tensors
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.img_tensors)

    def __getitem__(self, idx):
        image = self.img_tensors[idx]
        label = self.labels[idx]
        patient_id = self.ids[idx]
        # Return a single label instead of one-hot encoding
        return image, label, patient_id

# Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Data augmentation
data_transforms = transforms.Compose([
    transforms.RandomRotation(15),  # Rotation
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.RandomVerticalFlip(),  # Vertical flip
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1), ratio=(0.9, 1.1)),  # Random crop
    AddGaussianNoise(0., 0.01),  # Add Gaussian noise
    transforms.Normalize([0.3281186, 0.28937867, 0.20702125], [0.09407319, 0.09732835, 0.106712654])
])

test_transforms = transforms.Compose([
    transforms.Normalize([0.3281186, 0.28937867, 0.20702125], [0.09407319, 0.09732835, 0.106712654])
])

bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
def loss_function(outputs, targets): 
    targets = targets.float()
    return bce_with_logits_loss_fn(outputs, targets)

# Read clinical data file using Pandas
clini_df = pd.read_excel('/home1/HWGroup/zhangxu/Transformer/dl-mri-main/HER_LOW/HER_LOW_OVER_471.xlsx')
name_list = clini_df.id.tolist()
labels = clini_df['HER2_subtype'].values

# 8:2 stratified split of training and validation sets
X_train, X_test, y_train, y_test = train_test_split(name_list, labels, test_size=0.2, stratify=labels, random_state=42)

batch_size = 16
lr = 0.0001
epochs = 50

# Generate datasets and loaders based on training and validation set IDs
file_dir = '/home1/HWGroup/zhangxu/Resnet/Data_Aurora'
train_dataset, train_HER2_subtype_list = gen_dataset(file_dir, X_train, clini_df, dataset_type="train")
test_dataset, test_HER2_subtype_list = gen_dataset(file_dir, X_test, clini_df, dataset_type="test")

train_labels = np.array(train_HER2_subtype_list).astype('int')
train_labels = np.expand_dims(train_labels, axis=-1)

train_custom_dataset = CustomDataset(train_dataset, train_labels, X_train, transform=data_transforms)
test_labels = np.array(test_HER2_subtype_list).astype('float32')
test_labels = np.expand_dims(test_labels, axis=-1)
test_custom_dataset = CustomDataset(test_dataset, test_labels, X_test, transform=test_transforms)

train_loader = DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_custom_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Reset model and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_tiny_patch16_224', 
                            num_classes=1,
                            drop_rate=0.,
                            drop_path_rate=0.,
                            attn_drop_rate=0.,
                            pos_drop_rate=0.,
                            patch_drop_rate=0.,
                            proj_drop_rate=0.)
model.to(device)

pretrained_weights = torch.load('/home1/HWGroup/zhangxu/Transformer/dl-mri-main/vit_tiny_patch16_224.pth')
model.load_state_dict(pretrained_weights, strict=False)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# Training and validation loop
model_save_path = '/home1/HWGroup/zhangxu/Transformer/dl-mri-main/HER_LOW/HER_vit_tiny_best.pth'
early_stopping = EarlyStopping(model_path=model_save_path, patience=8, verbose=True)

for epoch in range(epochs):
    model.train()
    print("\nStart of epoch %d" % (epoch,))
    train_proba = []
    train_true_labels = []
    train_loss = 0.0

    for x_batch_train, y_batch_train, _ in train_loader:
        x_batch_train, y_batch_train = x_batch_train.to(device), y_batch_train.to(device)
        
        # Forward pass
        y_pred = model(x_batch_train)
        y_probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        loss = loss_function(y_pred, y_batch_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_proba.extend(y_probs)
        train_true_labels.extend(y_batch_train.cpu().numpy())

    scheduler.step()
    train_auc = roc_auc_score(np.array(train_true_labels), np.array(train_proba))
    print(f"Epoch {epoch + 1}, train_auc: {train_auc}")
    
    # Validation
    model.eval()
    test_proba = []
    test_true_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for x_batch_test, y_batch_test, _ in test_loader:
            x_batch_test, y_batch_test = x_batch_test.to(device), y_batch_test.to(device)
            test_pred = model(x_batch_test)
            probs = torch.sigmoid(test_pred).detach().cpu().numpy()
            loss = loss_function(test_pred, y_batch_test)
            test_loss += loss.item() * x_batch_test.size(0)
            test_proba.extend(probs)
            test_true_labels.extend(y_batch_test.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_auc = roc_auc_score(np.array(test_true_labels), np.array(test_proba))
    print(f"Epoch {epoch + 1}, test_auc: {test_auc}")

    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
best_model_weights_path = model_save_path
model.load_state_dict(torch.load(best_model_weights_path))
