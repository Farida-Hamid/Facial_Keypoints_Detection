import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)       #(224-4)/1 + 1= 221/ 2 = 110
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.drop = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(128*26*26, 256) #128*26*26
        self.fc2 = nn.Linear(256, 136)

        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
     
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        return x