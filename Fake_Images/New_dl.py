# -*- coding: utf-8 -*-

import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import random


# ============================================================
# 1. CLEAN CORRUPTED IMAGES
# ============================================================

def clean_corrupted_images(root_folder):
    removed = 0
    total = 0

    for root, _, files in os.walk(root_folder):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            total += 1
            path = os.path.join(root, fname)

            try:
                img = Image.open(path)
                img.verify()   # Validate file
            except Exception:
                removed += 1
                os.remove(path)

    print(f"✓ Clean completed: removed {removed} corrupted images out of {total}")


# ============================================================
# 2. DATASET CLASS
# ============================================================

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        class_names = sorted(os.listdir(root))

        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        for cls in class_names:
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(folder, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# 3. CNN MODEL
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 4. DIFFUSION-STYLE NOISE FUNCTION
# ============================================================

def make_diffusion_noise_fn(train=True):
    def add_noise(x):
        if train:
            noise = torch.randn_like(x) * 0.15
            return torch.clamp(x + noise, 0, 1)
        return x
    return add_noise


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device, noise_fn):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        imgs = noise_fn(imgs)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total += len(labels)
        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / total


def eval_model(model, loader, loss_fn, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            total += len(labels)
            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / total


# ============================================================
# 6. FULL TRAINING LOOP
# ============================================================

def run_training(model, train_loader, test_loader, device, num_epochs=1, lr=1e-3, add_noise_fn_train=None, tag="Model"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"====== Starting Training: {tag} ======")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, add_noise_fn_train
        )

        test_loss, test_acc = eval_model(
            model, test_loader, loss_fn, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    return model


# ============================================================
# 7. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    data_root = "ddata/train"   # CHANGE THIS
    clean_corrupted_images(data_root)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolderDataset("ddata/train", transform)
    test_dataset = ImageFolderDataset("ddata/test", transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    num_classes = len(train_dataset.class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes)

    model = run_training(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=10,
        lr=1e-3,
        add_noise_fn_train=make_diffusion_noise_fn(train=True),
        tag="Diffusion-Style Noisy CNN"
    )
