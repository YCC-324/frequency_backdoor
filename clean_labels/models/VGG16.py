import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=2):	   # num_classes
        super(VGG16, self).__init__()
        net = models.vgg16(pretrained=False)   # load vgg16 
        net.classifier = nn.Sequential()	# set vgg16's fc layer empty
        self.features = net		# keep vgg16's feature layers
        self.classifier = nn.Sequential(    # define our fc layer
                nn.Linear(512 * 7 * 7, 512),  # 512 * 7 * 7 can't change size, defined by vgg16
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
