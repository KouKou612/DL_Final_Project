# diffusion.py

import torch
import torch.nn as nn
from trainer import create_loaders, run_training  # assuming previous trainer.py

# ================================================================
# Diffusion-style CNN model
# ================================================================
class DiffusionStyleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ================================================================
# Diffusion noise function
# ================================================================
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# no decreasing
'''
def make_diffusion_noise_fn(train=True, device='cpu'):
    alpha_cumprod_device = alpha_cumprod.to(device)
    def add_noise(x):
        b = x.size(0)
        t = torch.randint(0, T, (b,), device=x.device)
        a_bar = alpha_cumprod_device[t].view(-1,1,1,1)
        noise = torch.randn_like(x)
        return torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise
    return add_noise

'''

# decreasing noise
def make_diffusion_noise_fn(train=True, device='cpu', noise_scale=1.0):
    alpha_cumprod_device = alpha_cumprod.to(device)

    def add_noise(x, epoch=None, max_epoch=None):
        b = x.size(0)
        t = torch.randint(0, T, (b,), device=x.device)
        a_bar = alpha_cumprod_device[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)

        scale = noise_scale
        if epoch is not None and max_epoch is not None:
            # Linearly decrease noise from full scale to 10% over training
            scale *= max(0.1, 1 - epoch / max_epoch)

        noisy = torch.sqrt(a_bar) * x + scale * torch.sqrt(1 - a_bar) * noise
        return noisy

    return add_noise




# ================================================================
# Main training
# ================================================================
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

    model = DiffusionStyleCNN(num_classes=num_classes).to(device)

    # diffusion noise for training
    noise_fn_train = make_diffusion_noise_fn(train=True, device=device)

    run_training(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=10,
        lr=1e-4,
        add_noise_fn_train=noise_fn_train,
        add_noise_fn_test=None,
        tag="Diffusion-style Noisy CNN"
    )


if __name__ == "__main__":
    main()



'''
No decrease
===================================================
Training: Diffusion-style Noisy CNN
Device: cuda:0
Learning Rate: 0.0001
Epochs: 10
Train Batches: 354
Test Batches : 86
===================================================

[Epoch 1/10] Train Loss: 0.6876 | Acc: 0.5467 || Test Loss: 0.6294 | Acc: 0.6841 || Time: 262.7s | Best Acc: 0.6841
[Epoch 2/10] Train Loss: 0.6737 | Acc: 0.5774 || Test Loss: 0.6040 | Acc: 0.6290 || Time: 271.5s | Best Acc: 0.6841
[Epoch 3/10] Train Loss: 0.6608 | Acc: 0.5965 || Test Loss: 0.4963 | Acc: 0.8660 || Time: 226.6s | Best Acc: 0.8660
[Epoch 4/10] Train Loss: 0.6528 | Acc: 0.6082 || Test Loss: 0.5208 | Acc: 0.7323 || Time: 272.0s | Best Acc: 0.8660
[Epoch 5/10] Train Loss: 0.6475 | Acc: 0.6132 || Test Loss: 0.5359 | Acc: 0.7102 || Time: 296.5s | Best Acc: 0.8660
[Epoch 6/10] Train Loss: 0.6434 | Acc: 0.6156 || Test Loss: 0.4432 | Acc: 0.8522 || Time: 257.9s | Best Acc: 0.8660
[Epoch 7/10] Train Loss: 0.6405 | Acc: 0.6192 || Test Loss: 0.4251 | Acc: 0.8902 || Time: 198.5s | Best Acc: 0.8902
[Epoch 8/10] Train Loss: 0.6362 | Acc: 0.6243 || Test Loss: 0.5102 | Acc: 0.7354 || Time: 199.2s | Best Acc: 0.8902
[Epoch 9/10] Train Loss: 0.6349 | Acc: 0.6250 || Test Loss: 0.4231 | Acc: 0.8639 || Time: 221.6s | Best Acc: 0.8902
[Epoch 10/10] Train Loss: 0.6338 | Acc: 0.6249 || Test Loss: 0.4491 | Acc: 0.8057 || Time: 202.1s | Best Acc: 0.8902

Training completed.

'''

'''
decreasing
====================================
Device Status:
CUDA Available: True
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
Train Batches: 354
Test Batches : 86
Number of Classes: 2
====================================

===================================================
Training: Diffusion-style Noisy CNN
Device: cuda:0
Learning Rate: 0.0001
Epochs: 10
Train Batches: 354
Test Batches : 86
===================================================

[Epoch 1/10] Train Loss: 0.6856 | Acc: 0.5508 || Test Loss: 0.6605 | Acc: 0.6001 || Time: 200.2s | Best Acc: 0.6001
[Epoch 2/10] Train Loss: 0.6687 | Acc: 0.5852 || Test Loss: 0.5522 | Acc: 0.7303 || Time: 208.8s | Best Acc: 0.7303
[Epoch 3/10] Train Loss: 0.6565 | Acc: 0.6034 || Test Loss: 0.6331 | Acc: 0.6017 || Time: 208.0s | Best Acc: 0.7303
[Epoch 4/10] Train Loss: 0.6512 | Acc: 0.6078 || Test Loss: 0.4875 | Acc: 0.8076 || Time: 210.0s | Best Acc: 0.8076
[Epoch 5/10] Train Loss: 0.6461 | Acc: 0.6159 || Test Loss: 0.4780 | Acc: 0.7967 || Time: 211.3s | Best Acc: 0.8076
[Epoch 6/10] Train Loss: 0.6431 | Acc: 0.6155 || Test Loss: 0.4816 | Acc: 0.7864 || Time: 213.8s | Best Acc: 0.8076
[Epoch 7/10] Train Loss: 0.6392 | Acc: 0.6196 || Test Loss: 0.4501 | Acc: 0.8236 || Time: 213.1s | Best Acc: 0.8236
[Epoch 8/10] Train Loss: 0.6369 | Acc: 0.6223 || Test Loss: 0.6230 | Acc: 0.6306 || Time: 209.6s | Best Acc: 0.8236
[Epoch 9/10] Train Loss: 0.6341 | Acc: 0.6235 || Test Loss: 0.4961 | Acc: 0.7464 || Time: 210.5s | Best Acc: 0.8236
[Epoch 10/10] Train Loss: 0.6312 | Acc: 0.6259 || Test Loss: 0.4054 | Acc: 0.8916 || Time: 213.9s | Best Acc: 0.8916

Training completed.

'''