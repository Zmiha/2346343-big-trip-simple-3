import torch
from torch import optim
from data_load_and_train import load_cifar10_data, train_model
from ResNet import resnet

# Параметры
batch_size = 32
num_epochs = 5
learning_rate = 0.001

# Загрузка данных
train_loader, valid_loader = load_cifar10_data(batch_size)

# Создание модели
torch.cuda.empty_cache()
model = resnet(in_channels=3, architecture='resnet18')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
model = train_model(model, train_loader, valid_loader, optimizer, num_epochs)

# Сохранение модели
# torch.save(model.state_dict(), 'resnet18_binary_classification.pth')
