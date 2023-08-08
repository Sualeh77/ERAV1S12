import os
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch

from typing import Any, Optional
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import random_split
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

from transform import get_a_train_transform, get_a_test_transform
from main import get_lit_loader

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class LitCustomResnet(LightningModule):
    def __init__(self, ip_ch=3 , num_classes=10, data_dir=PATH_DATASETS, learning_rate=0.03, batch_size=32,
                norm:str="bn", num_steps=0, num_epochs=25, max_lr=5.07E-02) -> None:
        super(LitCustomResnet, self).__init__()

        # Set init args
        self.ip_ch = ip_ch
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.norm = norm
        self.num_steps= num_steps
        self.num_epochs = num_epochs
        self.max_lr = max_lr

        # Data specific attr
        self.num_classes = num_classes
        self.train_transform = get_a_train_transform()
        self.test_transform = get_a_test_transform()

        # Pytorch Model
        # Prep Layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.ip_ch, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False), # 3x32x32 > 64x32x32 | RF 1 > 3 | J 1
            nn.GroupNorm(4, 64) if self.norm=="gn" else nn.GroupNorm(1, 64) if self.norm=="ln" else nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False), # 64x32x32 > 128x32x32 | RF 5
            nn.MaxPool2d(2, 2), # 128x32x32 > 128x16x16 | RF 6 | J 2
            nn.GroupNorm(4, 128) if self.norm=="gn" else nn.GroupNorm(1, 128) if self.norm=="ln" else nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # R1
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False), # 128x16x16 > 128x16x16 | RF 10
            nn.GroupNorm(4, 128) if self.norm=="gn" else nn.GroupNorm(1, 128) if self.norm=="ln" else nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False), # 128x16x16 > 128x16x16 | RF 14
            nn.GroupNorm(4, 128) if self.norm=="gn" else nn.GroupNorm(1, 128) if self.norm=="ln" else nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=0, bias=False), # 128x16x16 > 256x14x14 | RF 18
            nn.MaxPool2d(2, 2), # 256x14x14 > 256x7x7 | RF 20 | J 4
            nn.GroupNorm(4, 256) if self.norm=="gn" else nn.GroupNorm(1, 256) if self.norm=="ln" else nn.BatchNorm2d(256),
            nn.ReLU(),
        )


        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), # 256x7x7 > 512x7x7 | RF 28
            nn.MaxPool2d(2, 2), # 512x7x7 > 512x3x3 | RF 32 | J 8
            nn.GroupNorm(4, 512) if self.norm=="gn" else nn.GroupNorm(1, 512) if self.norm=="ln" else nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # R2
        self.r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), # 512x3x3 > 512x3x3 | RF 48
            nn.GroupNorm(4, 512) if self.norm=="gn" else nn.GroupNorm(1, 512) if self.norm=="ln" else nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), # 512x3x3 > 512x3x3 | RF 64
            nn.GroupNorm(4, 512) if self.norm=="gn" else nn.GroupNorm(1, 512) if self.norm=="ln" else nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Maxpool k=4
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1) # 512x3x3 > 512x1x1

        # FC Layer
        self.fc = nn.Linear(512, self.num_classes, bias=False)
        

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = x + self.r1(x)     # Skip Connection - 1
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.r2(x)     # Skip Connection - 2
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
        #return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # self.log to surface up scalars in tensorboard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        # Here we reuse validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-04)
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)   # StepLR Scheduler
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer=optimizer, patience=4, factor=0.1, mode="min" 
        #)

        # CosineAnnealing with One-cycle schedule decay each epoch/batch
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer=optimizer, T_max=self.num_steps * self.num_epochs     # T_max will be num_epochs/step for decay in epoch case
        #)

        # OneCycleLR scheduler 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr = self.max_lr, steps_per_epoch = self.num_steps, epochs = self.num_epochs, pct_start = 5/self.num_epochs,
            div_factor = 100, three_phase = False, final_div_factor = 100, anneal_strategy = "linear"
        )

        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler":scheduler,
                "monitor":"train_loss",
                "interval":"step", # Default = epoch| While using cosine annealing scheduler, epoch means restart LR at every epoch, step means at every batch.
                "frequency":1,   # Default
            },
        }
    
    # Data related Hooks
    def prepare_data(self) -> None:
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_full = datasets.CIFAR10(self.data_dir, train=True)
            self.cifar_train, self.cifar_val = random_split(self.cifar_full, [45000, 5000])
            
        # Assign test dataset for use in dataloader(s)
        if stage=="test" or stage is None:
            self.cifar_test = datasets.CIFAR10(self.data_dir, train=False)
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return get_lit_loader(self.cifar_train, self.train_transform, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return get_lit_loader(self.cifar_val, self.test_transform, batch_size=self.batch_size)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return get_lit_loader(self.cifar_test, self.test_transform, batch_size=self.batch_size)
        