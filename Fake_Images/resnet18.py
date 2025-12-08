# resnet18.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from trainer import create_loaders, run_training

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

    # Load pretrained ResNet18
    model = models.resnet18(weights=None)  # set weights='IMAGENET1K_V1' if pretrained
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Use weight decay for regularization
    lr = 1e-3
    weight_decay = 1e-4  # L2 regularization
    optimizer_params = {"lr": lr, "weight_decay": weight_decay}

    # Adjust training parameters for large 256x256 dataset
    run_training(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=10,
        lr=lr,
        tag="ResNet18"
    )


if __name__ == "__main__":
    main()


'''
[Epoch 1/20] Train Loss: 0.4978 | Acc: 0.7572 || Test Loss: 0.2471 | Acc: 0.9045 || Time: 209.1s | Best Acc: 0.9045
[Epoch 2/20] Train Loss: 0.2911 | Acc: 0.8767 || Test Loss: 0.2121 | Acc: 0.9345 || Time: 213.5s | Best Acc: 0.9345
[Epoch 3/20] Train Loss: 0.2056 | Acc: 0.9172 || Test Loss: 0.3250 | Acc: 0.8618 || Time: 216.7s | Best Acc: 0.9345
[Epoch 4/20] Train Loss: 0.1507 | Acc: 0.9411 || Test Loss: 0.3474 | Acc: 0.8542 || Time: 223.0s | Best Acc: 0.9345
[Epoch 5/20] Train Loss: 0.1182 | Acc: 0.9542 || Test Loss: 0.5072 | Acc: 0.7976 || Time: 216.4s | Best Acc: 0.9345
[Epoch 6/20] Train Loss: 0.0978 | Acc: 0.9629 || Test Loss: 0.5803 | Acc: 0.7839 || Time: 208.6s | Best Acc: 0.9345
[Epoch 7/20] Train Loss: 0.0797 | Acc: 0.9698 || Test Loss: 0.6770 | Acc: 0.7941 || Time: 226.9s | Best Acc: 0.9345
[Epoch 8/20] Train Loss: 0.0684 | Acc: 0.9744 || Test Loss: 0.5717 | Acc: 0.7981 || Time: 225.7s | Best Acc: 0.9345
[Epoch 9/20] Train Loss: 0.0613 | Acc: 0.9774 || Test Loss: 0.2971 | Acc: 0.9157 || Time: 222.8s | Best Acc: 0.9345
[Epoch 10/20] Train Loss: 0.0506 | Acc: 0.9809 || Test Loss: 0.7965 | Acc: 0.7514 || Time: 224.5s | Best Acc: 0.9345
[Epoch 11/20] Train Loss: 0.0498 | Acc: 0.9814 || Test Loss: 0.6979 | Acc: 0.7945 || Time: 228.5s | Best Acc: 0.9345
[Epoch 12/20] Train Loss: 0.0386 | Acc: 0.9860 || Test Loss: 0.5084 | Acc: 0.8422 || Time: 243.3s | Best Acc: 0.9345

'''


'''
[Epoch 1/10] Train Loss: 0.4659 | Acc: 0.7788 || Test Loss: 0.2752 | Acc: 0.8993 || Time: 204.6s | Best Acc: 0.8993
[Epoch 2/10] Train Loss: 0.2819 | Acc: 0.8820 || Test Loss: 0.4589 | Acc: 0.7848 || Time: 216.3s | Best Acc: 0.8993
[Epoch 3/10] Train Loss: 0.1985 | Acc: 0.9199 || Test Loss: 0.5448 | Acc: 0.7883 || Time: 221.5s | Best Acc: 0.8993
[Epoch 4/10] Train Loss: 0.1477 | Acc: 0.9419 || Test Loss: 0.5672 | Acc: 0.7979 || Time: 228.0s | Best Acc: 0.8993
[Epoch 5/10] Train Loss: 0.1167 | Acc: 0.9548 || Test Loss: 0.3391 | Acc: 0.8570 || Time: 224.9s | Best Acc: 0.8993
[Epoch 6/10] Train Loss: 0.0938 | Acc: 0.9644 || Test Loss: 0.3703 | Acc: 0.8576 || Time: 230.0s | Best Acc: 0.8993
[Epoch 7/10] Train Loss: 0.0774 | Acc: 0.9700 || Test Loss: 0.9908 | Acc: 0.7539 || Time: 233.2s | Best Acc: 0.8993
[Epoch 8/10] Train Loss: 0.0658 | Acc: 0.9757 || Test Loss: 0.7152 | Acc: 0.7935 || Time: 226.9s | Best Acc: 0.8993
[Epoch 9/10] Train Loss: 0.0549 | Acc: 0.9799 || Test Loss: 0.7778 | Acc: 0.8062 || Time: 209.0s | Best Acc: 0.8993
[Epoch 10/10] Train Loss: 0.0498 | Acc: 0.9816 || Test Loss: 0.7262 | Acc: 0.7996 || Time: 213.8s | Best Acc: 0.8993

Training completed.


'''