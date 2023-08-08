import torch.nn as nn
import torch.nn.functional as F


class CustomResnet(nn.Module):
    def __init__(self, device, norm:str="bn"):
        super(CustomResnet, self).__init__()

        self.device = device
        self.norm = norm

        # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False), # 3x32x32 > 64x32x32 | RF 1 > 3 | J 1
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

        self.R1 = nn.Sequential(
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

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), # 512x3x3 > 512x3x3 | RF 48
            nn.GroupNorm(4, 512) if self.norm=="gn" else nn.GroupNorm(1, 512) if self.norm=="ln" else nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), # 512x3x3 > 512x3x3 | RF 64
            nn.GroupNorm(4, 512) if self.norm=="gn" else nn.GroupNorm(1, 512) if self.norm=="ln" else nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Maxpool k=4
        self.max = nn.MaxPool2d(kernel_size=4, stride=2, padding=1) # 512x3x3 > 512x1x1

        # FC Layer
        self.fc = nn.Linear(512, 10, bias=False)


    def forward(self, x):
        x = x.to(self.device)
        x = self.preplayer(x)
        x = self.layer1(x)
        x = x + self.R1(x)     # Skip Connection - 1
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.R2(x)     # Skip Connection - 2
        x = self.max(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        #return F.log_softmax(x, dim=1)
        return x