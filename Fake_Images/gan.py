# gan.py

import torch
import torch.nn as nn
from trainer import create_loaders, run_training


class DCGANDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    base_path = "ddata"
    train_loader, test_loader, classes = create_loaders(base_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("====================================")
    print("Device Status:")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    print("====================================\n")

    model = DCGANDiscriminator(len(classes))

    run_training(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=10,
        lr=1e-4,
        tag="GAN-style Discriminator"
    )


if __name__ == "__main__":
    main()


'''
====================================
Device Status:
CUDA Available: True
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
====================================

===================================================
Training: GAN-style Discriminator
Device: cuda:0
Learning Rate: 0.0001
Epochs: 10
Train Batches: 2826
Test Batches : 681
===================================================

[Epoch 1/10] Train Loss: 0.5165 | Acc: 0.7428 || Test Loss: 0.3620 | Acc: 0.8434 || Time: 328.0s | Best Acc: 0.8434
[Epoch 2/10] Train Loss: 0.3952 | Acc: 0.8237 || Test Loss: 0.2862 | Acc: 0.8905 || Time: 369.6s | Best Acc: 0.8905
[Epoch 3/10] Train Loss: 0.3245 | Acc: 0.8611 || Test Loss: 0.2322 | Acc: 0.9167 || Time: 371.4s | Best Acc: 0.9167
[Epoch 4/10] Train Loss: 0.2748 | Acc: 0.8853 || Test Loss: 0.1908 | Acc: 0.9321 || Time: 370.9s | Best Acc: 0.9321
[Epoch 5/10] Train Loss: 0.2349 | Acc: 0.9043 || Test Loss: 0.1758 | Acc: 0.9407 || Time: 383.0s | Best Acc: 0.9407
[Epoch 6/10] Train Loss: 0.2086 | Acc: 0.9168 || Test Loss: 0.1983 | Acc: 0.9302 || Time: 395.5s | Best Acc: 0.9407
[Epoch 7/10] Train Loss: 0.1831 | Acc: 0.9279 || Test Loss: 0.1917 | Acc: 0.9351 || Time: 393.2s | Best Acc: 0.9407
[Epoch 8/10] Train Loss: 0.1666 | Acc: 0.9345 || Test Loss: 0.2199 | Acc: 0.9180 || Time: 393.7s | Best Acc: 0.9407
[Epoch 9/10] Train Loss: 0.1508 | Acc: 0.9405 || Test Loss: 0.1784 | Acc: 0.9374 || Time: 392.7s | Best Acc: 0.9407
[Epoch 10/10] Train Loss: 0.1389 | Acc: 0.9462 || Test Loss: 0.1580 | Acc: 0.9496 || Time: 391.7s | Best Acc: 0.9496

Training completed.


'''