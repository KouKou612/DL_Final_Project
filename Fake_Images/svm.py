# svm_rbf.py

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from trainer import create_loaders
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# ============================================================
# Feature extractor using pretrained ResNet18
# ============================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # remove fc

    def forward(self, x):
        x = self.features(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        return x

# ============================================================
# Extract features batch-wise
# ============================================================
def extract_features_batch(loader, model, device):
    model.eval()
    X_batches, y_batches = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            X_batches.append(feats.cpu().numpy())
            y_batches.append(labels.numpy())
    return X_batches, y_batches

# ============================================================
# Main
# ============================================================
def main():
    base_path = "ddata"
    train_loader, test_loader, classes = create_loaders(base_path)
    num_classes = len(classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("====================================")
    print(f"Device (feature extraction): {device}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Test Batches : {len(test_loader)}")
    print(f"Number of Classes: {num_classes}")
    print("Dataset Size (Approx): 80000 images")
    print("====================================\n")

    # Feature extractor
    extractor = FeatureExtractor().to(device)

    # Extract train and test features batch-wise
    print("Extracting train features...")
    X_train_batches, y_train_batches = extract_features_batch(train_loader, extractor, device)
    print("Extracting test features...")
    X_test_batches, y_test_batches = extract_features_batch(test_loader, extractor, device)

    # Stack batches
    X_train = np.vstack(X_train_batches)
    y_train = np.concatenate(y_train_batches)
    X_test = np.vstack(X_test_batches)
    y_test = np.concatenate(y_test_batches)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train RBF kernel SVM
    print("Training SVM with RBF kernel...")
    start_time = time.time()
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f}s\n")

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("SVM (RBF Kernel) Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()

'''
Linear

Train Accuracy: 0.6745
Test Accuracy : 0.7456

RBF
Train Accuracy: 0.8991
Test Accuracy : 0.8934
'''