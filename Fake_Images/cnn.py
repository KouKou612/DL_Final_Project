# cnn.py

import torch
import torch.nn as nn
from trainer import create_loaders, run_training  # assuming your previous image_dl renamed trainer

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    base_path = "ddata"
    train_loader, test_loader, classes = create_loaders(base_path)
    num_classes = len(classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("====================================")
    print("Device Status:")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    print("Train Batches:", len(train_loader))
    print("Test Batches :", len(test_loader))
    print("Number of Classes:", num_classes)
    print("====================================\n")

    model = SimpleCNN(num_classes=num_classes).to(device)
    print(model)

    run_training(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=10,
        lr=1e-4,
        tag="CNN"
    )


if __name__ == "__main__":
    main()


'''
===================================================
Training: CNN
Device: cuda:0
Learning Rate: 0.0001
Epochs: 10
Train Batches: 2826
Test Batches : 681
===================================================

[Epoch 1/10] Train Loss: 0.5321 | Acc: 0.7344 || Test Loss: 0.3600 | Acc: 0.8218 || Time: 360.9s | Best Acc: 0.8218
[Epoch 2/10] Train Loss: 0.4282 | Acc: 0.8070 || Test Loss: 0.2554 | Acc: 0.9068 || Time: 303.5s | Best Acc: 0.9068
[Epoch 3/10] Train Loss: 0.3686 | Acc: 0.8408 || Test Loss: 0.2266 | Acc: 0.9203 || Time: 362.9s | Best Acc: 0.9203
[Epoch 4/10] Train Loss: 0.3228 | Acc: 0.8627 || Test Loss: 0.1815 | Acc: 0.9536 || Time: 364.8s | Best Acc: 0.9536
[Epoch 5/10] Train Loss: 0.2925 | Acc: 0.8780 || Test Loss: 0.2120 | Acc: 0.9364 || Time: 363.4s | Best Acc: 0.9536
[Epoch 6/10] Train Loss: 0.2675 | Acc: 0.8894 || Test Loss: 0.2269 | Acc: 0.9279 || Time: 363.3s | Best Acc: 0.9536
[Epoch 7/10] Train Loss: 0.2450 | Acc: 0.8995 || Test Loss: 0.2439 | Acc: 0.9157 || Time: 335.4s | Best Acc: 0.9536
[Epoch 8/10] Train Loss: 0.2261 | Acc: 0.9084 || Test Loss: 0.2328 | Acc: 0.9305 || Time: 356.3s | Best Acc: 0.9536
[Epoch 9/10] Train Loss: 0.2149 | Acc: 0.9132 || Test Loss: 0.2124 | Acc: 0.9479 || Time: 343.7s | Best Acc: 0.9536
[Epoch 10/10] Train Loss: 0.2042 | Acc: 0.9183 || Test Loss: 0.2371 | Acc: 0.9440 || Time: 342.8s | Best Acc: 0.9536

Training completed.

'''