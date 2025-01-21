import torch
from torch import nn
from torch.nn import MaxPool1d

class CNN_VO2_1D(nn.Module):
    def __init__(self, numChannels, numNodes=[128, 128, 128, 64, 256]):
        """
        :param numChannels:
        :param classes:
        :param numNodes: number of nodes in hidden layer
        Structure of CNN: CONV1 => RELU => CONV2 => RELU => POOLING => DROPOUT
        """
        # Call the parent constructor
        super(CNN_VO2_1D, self).__init__()
        # self.classes = classes
        self.channels = numChannels

        conv1 = torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3)
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(p=0.5)
        self.cnnBlock1 = nn.Sequential(conv1, relu1, conv2, relu2, maxpool2, dropout2)

        # initialize second set of CNN
        conv3 = torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        relu3 = torch.nn.ReLU()
        conv4 = torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3)
        relu4 = torch.nn.ReLU()
        maxpool4 = MaxPool1d(kernel_size=2, stride=2)
        dropout4 = nn.Dropout(p=0.5)
        self.cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4)

        # Calculate the size of the flattened feature map
        self._to_linear = None
        self.convs(torch.randn(1, numChannels, 512))

        # Regression output
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, numNodes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(numNodes[4], 1)
        )

    def convs(self, x):
        x = self.cnnBlock1(x)
        x = self.cnnBlock2(x)
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.regressor(x)
        return x
