import os
import numpy as np
from models.clusternet import *
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from utils.cityscapes_dataset import Cityscapes

# Set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Dataset
dataset_train = Cityscapes(split='train',
                           data_root='cityscapes_dataset/',
                           data_list='cityscapes_dataset/lists/cityscapes/train_set.txt')

# Dataloader
train_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

# Loss Criterion
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Model
model = ClusterNet(layers=50, bins=(2, 3, 6, 8), dropout=0.1, classes=35, zoom_factor=8, use_ppm=True, pretrained=True, criterion=criterion).to(device)

# Optimizer and list of parameters to optimize
modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
modules_new = [model.ppm, model.cls, model.aux]
lr = 0.001
params_list = []
for module in modules_ori:
    params_list.append(dict(params=module.parameters(), lr=lr))
for module in modules_new:
    params_list.append(dict(params=module.parameters(), lr=lr * 10))
optimizer = optim.SGD(params_list, lr=lr, momentum=0.9, weight_decay=0.0001)

timestamp = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")
exp_path = timestamp
os.mkdir(exp_path)

# Train loop
print('Started at ' + timestamp)
train_losses = []
for epoch in range(50):

    torch.cuda.empty_cache()

    train_sum_loss = 0
    for img, mask in train_dataloader:

        # Zero stored gradients
        optimizer.zero_grad()

       # Forward pass
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        _, main_loss, aux_loss = model(img.to(device), mask.to(device))
        loss = torch.mean(main_loss) + 0.4 * torch.mean(aux_loss)
        print(loss)

        # Loss back-propagation
        loss.backward()

        # Optimization step
        optimizer.step()

        train_sum_loss += loss.detach().cpu().numpy()/len(train_dataloader)
    train_losses.append(train_sum_loss)

    print('epoch ' + str(epoch) + ' train loss ' + str(np.round(train_sum_loss,4)))

timestamp = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")
print('Ended at ' + timestamp)

# Save trained model
torch.save(model.state_dict(), exp_path + '/clusternet_' + str(epoch) + 'ep.pth')
torch.save(model, exp_path + '/clusternet_' + str(epoch) + 'ep.pt')
torch.save(train_losses, exp_path + '/clusternet_' + str(epoch) + 'ep_TRAIN_LOSSES.pt')