import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets




class CovNetMNIST(nn.Module):
    def __init__(self):
        super(CovNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5,stride=1,padding=20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32*7*7, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(x.shape[0], -1)
       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
