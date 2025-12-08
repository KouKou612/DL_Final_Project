
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ================================================================
# Clean corrupted images
# ================================================================
def clean_corrupted_images(root_folder):
    removed, total = 0, 0
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
                total += 1
                fp = os.path.join(root, f)
                try:
                    img = Image.open(fp)
                    img.verify()
                except Exception:
                    print(f"❌ Removed corrupted: {fp}")
                    os.remove(fp)
                    removed += 1

    print("====================================")
    print(f"Checked: {total} images")
    print(f"Removed: {removed} corrupted")
    print("====================================")


# ================================================================
# Dataset / Dataloaders
# ================================================================
def create_loaders(base_path):
    train_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = datasets.ImageFolder(os.path.join(base_path, "train"), transform=train_trans)
    test_ds = datasets.ImageFolder(os.path.join(base_path, "test"), transform=test_trans)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    return train_loader, test_loader, train_ds.classes


# ================================================================
# Training functions
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total


# ================================================================
# Training + metrics recording + visualization
# ================================================================
def plot_metrics(history, tag="MODEL"):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["test_acc"], label="Test Acc")
    plt.title(f"{tag} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title(f"{tag} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_training(model, train_loader, test_loader, device,
                 num_epochs=10, lr=1e-4, tag="MODEL",
                 add_noise_fn_train=None, add_noise_fn_test=None):

    print("===================================================")
    print(f"Training: {tag}")
    print(f"Device: {device}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Test Batches : {len(test_loader)}")
    print("===================================================\n")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    best_acc = 0.0

    for ep in range(num_epochs):
        start = time.time()

        # === Train ===
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if add_noise_fn_train is not None:
                x = add_noise_fn_train(x)  # add noise here

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        tr_loss, tr_acc = running_loss / total, correct / total

        # === Evaluate ===
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if add_noise_fn_test is not None:
                    x = add_noise_fn_test(x)  # add test noise if provided

                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        te_loss, te_acc = running_loss / total, correct / total

        # record history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc

        print(f"[Epoch {ep+1}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} || "
              f"Test Loss: {te_loss:.4f} | Acc: {te_acc:.4f} || "
              f"Time: {time.time()-start:.1f}s | Best Acc: {best_acc:.4f}")

    print("\nTraining completed.\n")
    plot_metrics(history, tag)

    return model
