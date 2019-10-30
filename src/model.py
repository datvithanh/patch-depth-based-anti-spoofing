import torch 
import torch.nn as nn
import torch.nn.functional as F

class PatchConvLayer(nn.Module):
    def __init__(self, in_channnel, out_channel, pooling_size):
        super(PatchConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channnel, out_channel, 3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.max_pool = nn.MaxPool2d(pooling_size, pooling_size, padding=0, dilation=1)

    def forward(self, X):
        return self.max_pool(F.relu(self.bn(self.conv(X))))

class PatchCNN(nn.Module):
    def __init__(self):
        super(PatchCNN, self).__init__()
        self.conv1 = PatchConvLayer(3, 50, 2)
        self.conv2 = PatchConvLayer(50, 100, 2)
        self.conv3 = PatchConvLayer(100, 200, 2)
        self.conv4 = PatchConvLayer(200, 250, 2)

        self.fc1 = nn.Linear(2250, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1000, 400)
        self.bn2 = nn.BatchNorm1d(400)

        self.fc3 = nn.Linear(400, 2)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)

        X = self.fc1(X.view(X.shape[0], -1))
        X = F.relu(self.bn1(X))
        X = self.dropout1(X)
        
        X = self.fc2(X)
        X = F.relu(self.bn2(X))

        X = self.fc3(X)
        X = F.log_softmax(X, dim=1)

        return X

class DepthCNN(nn.Module):
    def __init__(self):
        pass
