import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self):
        ##A small Unet model with two layers 
        super(DiffusionModel, self).__init__() 
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32*3)
        )

    def forward(self, x):
        return self.main(x)

