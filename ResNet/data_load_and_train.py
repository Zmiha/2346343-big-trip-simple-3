import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Загрузка и пред обработка данных
def load_cifar10_data(batch_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Загружаем дата сет CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Преобразуем в бинарную классификацию (например, только классы "0" и "1")
    train_idx = np.where((np.array(train_dataset.targets) == 0) | (np.array(train_dataset.targets) == 1))[0]
    valid_idx = np.where((np.array(valid_dataset.targets) == 0) | (np.array(valid_dataset.targets) == 1))[0]

    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


# Определение функции обучения
def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer,
                num_epochs: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels.float().unsqueeze(1))

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.unsqueeze(1))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
