from tqdm import tqdm
from utils import GetCorrectPredCount
import torch
from torch.utils.data import DataLoader
from dataset import Cifar10Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import os

EPOCHS = 20
Batch_Size = 512

########################################################################################################################################################

def set_optim(model, opt:str = "SGD", learnin_rate=0.01, momentum_=0.9, weight_decay_=None):
    if opt == "SGD":
        return optim.SGD(model.parameters(), lr=learnin_rate, momentum=momentum_)
    elif opt == "Adam":
        return optim.Adam(model.parameters(), lr=learnin_rate, weight_decay=weight_decay_)
    
########################################################################################################################################################

def get_one_cycle_lr_scheduler(optimizer, max_lr, steps_per_epoch, epochs=EPOCHS, anneal_strategy='linear'):
    """Create instance of one cycle lr scheduler

    Args:
        optimizer (torch.optim): Optimizer to be used for Training
        lr (float): base lr value used
        max_lr (float): max lr value used in one cycle ly
        steps_per_epoch (int): Number of steps in each epochs
        epochs (int): number of epochs for which training is done | Default as set in main.py
        anneal_strategy (str, optional): Defaults to 'linear'.

    Returns:
        OneCycleLR: Instance of one cycle lr scheduler
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start = 5/EPOCHS,
        div_factor = 100,
        three_phase = False,
        final_div_factor = 100,
        anneal_strategy='linear'
    )

########################################################################################################################################################

# Data to plot accuracy and loss graphs
train_losses = []
train_acc = []

########################################################################################################################################################
def model_train(model, device, train_loader, optimizer, criterion, scheduler):
    """
        Training method
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate Loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        scheduler.step()
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

########################################################################################################################################################

# Data to plot accuracy and loss graphs
test_losses = []
test_acc = []

########################################################################################################################################################
def model_test(model, device, test_loader, criterion):
    """
        Test method.
    """
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            
            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, correct
    
#########################################################################################################################################################

def get_loader(train_data, test_data, train_transform, test_transform, batch_size=Batch_Size, use_cuda=False, use_mps=False):
    """
    Get instance of train and test loaders

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to False.
        use_mps (bool, optional): Enable/Disable MPS for mac. Defaults to False.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda or use_mps else {}

    train_loader = DataLoader(
            Cifar10Dataset(train_data, transforms=train_transform),
            batch_size=batch_size, shuffle=True, **kwargs
        )
    
    test_loader = DataLoader(
            Cifar10Dataset(test_data, transforms=test_transform),
            batch_size=batch_size, shuffle=True, **kwargs
        )
    
    return train_data, test_data, train_loader, test_loader

#########################################################################################################################################################

def get_lit_loader(data_set, transformation, batch_size=128, shuffle=False):
    """
    Get instance of train and test loaders

    Args:
        datasset : train/val/test
        transformation (Transform): Instance of transform function
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.

    Returns:
        DataLoader: Get instance of train and test data loaders
    """
    #kwargs = {'num_workers': os.cpu_count()-1, 'pin_memory': True}
    kwargs = {'num_workers': 0, 'pin_memory': True}

    dataloader = DataLoader(
            Cifar10Dataset(data_set, transforms=transformation),
            batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs
        )
    
    return dataloader

#########################################################################################################################################################