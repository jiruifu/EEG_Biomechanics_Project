import torch
from torch import nn
from torch.nn import MaxPool1d

class MODCNN_1D_MUST(nn.Module):
    def __init__(self, numChannels, numNodes=[128, 128, 128, 64, 256]):
        """
        :param numChannels:
        :param classes:
        :param numNodes: number of nodes in hidden layer
        Structure of CNN: CONV1 => RELU => CONV2 => RELU => POOLING => DROPOUT
        """
        # Call the parent constructor
        super(MODCNN_1D_MUST, self).__init__()
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

        # Regression output
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(numNodes[3] * (numNodes[2] // 4), numNodes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(numNodes[4], 1)  # Single output for VO2 prediction
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x1 = self.cnnBlock1(x)
        print(f"Shape after cnnBlock1: {x1.shape}")
        x2 = self.cnnBlock2(x1)
        print(f"Shape after cnnBlock2: {x2.shape}")
        ouput = self.regressor(x2)
        print(f"Output shape: {ouput.shape}")
        return ouput

class SODCNN_1D_MUST(nn.Module):
    def __init__(self, numChannels, numNodes=[128, 128, 128, 64, 256]):
        """
        :param numChannels:
        :param classes:
        :param numNodes: number of nodes in hidden layer
        Structure of CNN: CONV1 => RELU => CONV2 => RELU => POOLING => DROPOUT
        """
        # Call the parent constructor
        super(MODCNN_1D_MUST, self).__init__()
        self.classes = 1
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
        self.cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4)\

        self.flatten = nn.Flatten()
        self.outputs = nn.ModuleList()
        # For each motor unit, recognize if the spike is 1 or 0
        self.outputs.append(nn.Sequential(
            nn.LazyLinearLinear(numNodes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(numNodes[4], 1),
            nn.Sigmoid()
        ))

    def forward(self, x):
        x1 = self.cnnBlock1(x)
        x2 = self.cnnBlock2(x1)
        x3 = self.flatten(x2)  # Flatten all dimensions except batch
        must = self.outputs(x3)
        return must