import torch.nn as nn

class cifarmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),        # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, (3, 3), padding=1),       # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 32 x 16 x 16
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 64 x 8 x 8
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, (3, 3), padding=1),      # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=1),     # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),            # 128 x 4 x 4
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(2048, 10)
        )

    def forward(self, x_train):
        return self.main(x_train)