import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NMI_3D(nn.Module):
    def __init__(self, numChannels, classes, numNodes=[256, 256, 256, 64, 128]):
        """
        Build the 3D CNN for spike train identification
        :param numChannels: channels of 3D HD-EMG image which size is (d, h, w)
        :param classes: number of motor units
        :param numNodes: size of squared kernel size of each layer
        """
        super(NMI_3D, self).__init__()
        self.classes = classes
        self.numChannels = numChannels

        #Build the first 3D network which includes one 3D CNN module,
        # a batch normalization module, a dropout layer, and activation function
        conv1 = torch.nn.Conv3d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        bathNorm1 = torch.nn.BatchNorm3d(numNodes[0])
        dropout1 = torch.nn.Dropout3d(p=0.2)
        act1 = torch.nn.ReLU()

        # Build the second 3D network which includes one 3D CNN module,
        # a dropout layer, and activation function
        conv2 = torch.nn.Conv3d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        dropout2 = torch.nn.Dropout3d(p=0.2)
        act2 = torch.nn.ReLU()

        # Build the third 3D network which includes one 3D CNN module,
        # a dropout layer, and activation function
        conv3 = torch.nn.Conv3d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        dropout3 = torch.nn.Dropout3d(p=0.2)
        act3 = torch.nn.ReLU()

        # Build the forth 3D network which includes one 3D CNN module,
        #a dropout layer, a maxpooling layer and activation function
        conv4 = torch.nn.Conv3d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        act4 = torch.nn.ReLU()
        maxpool4 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        dropout4 = torch.nn.Dropout3d(p=0.2)
        self.cnnBlock = nn.Sequential(conv1, bathNorm1, dropout1,act1,conv2,dropout2,
                                      act2,conv3,dropout3,act3,conv4,act4,maxpool4,dropout4)
        self.flatten = torch.nn.Flatten()
        self.outputs = nn.ModuleList()
        for _ in range(classes):
            self.outputs.append(nn.Sequential(
                nn.LazyLinear(numNodes[3]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[3], 1),
                nn.Sigmoid()
            ))
    def forward(self, x):
        x1 = self.cnnBlock(x)
        x2 = self.flatten(x1)
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.outputs[i](x2))

        output = torch.cat(outputlist, dim=1)
        return output

class NMI_3D_Lite(nn.Module):
    def __init__(self, numChannels, classes, numNodes=[16, 16, 16, 8, 32, 16]):
        """
        Build the 3D CNN for spike train identification
        :param numChannels: channels of 3D HD-EMG image which size is (d, h, w)
        :param classes: number of motor units
        :param numNodes: size of squared kernel size of each layer
        """
        super(NMI_3D_Lite, self).__init__()
        self.classes = classes
        self.numChannels = numChannels

        #Build the first 3D network which includes one 3D CNN module,
        # a batch normalization module, a dropout layer, and activation function
        self.conv1 = torch.nn.Conv3d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3, stride=(1, 1, 1), padding = (0, 7, 11))
        self.act1 = torch.nn.ReLU()

        # Build the second 3D network which includes one 3D CNN module,
        # a dropout layer, and activation function
        self.conv2 = torch.nn.Conv3d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3, stride=(1, 1, 1))
        self.act2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bathNorm2 = torch.nn.BatchNorm3d(numNodes[1])
        self.dropout2 = torch.nn.Dropout3d(p=0.25)

        self.conv3 = torch.nn.Conv3d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        self.act3 = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv3d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=(3, 3, 3))
        self.act4 = torch.nn.ReLU()
        self.maxpool4 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.bathNorm4 = torch.nn.BatchNorm3d(numNodes[3])
        self.dropout4 = torch.nn.Dropout3d(p=0.25)

        # self.cnnBlock = nn.Sequential(conv1, act1, conv2, act2, maxpool2, bathNorm2, dropout2,
        #                               conv3,act3, conv4, act4, maxpool4, bathNorm4, dropout4)
        self.flatten = torch.nn.Flatten()
        self.outputs = nn.ModuleList()
        for _ in range(classes):
            self.outputs.append(nn.Sequential(
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.BatchNorm1d(numNodes[4]),
                nn.Dropout(p=0.25),
                nn.Linear(numNodes[4], numNodes[5]),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(numNodes[5], 1),
                nn.Sigmoid()
            ))
    def forward(self, x):
        # Go through the CNN one by one
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x3 = self.conv2(x2)
        x4 = self.act2(x3)
        x5 = self.maxpool2(x4)
        x6 = self.bathNorm2(x5)
        x7 = self.dropout2(x6)
        x8 = self.conv3(x7)
        x9 = self.act3(x8)
        x10 = self.conv4(x9)
        x11 = self.act4(x10)
        x12 = self.maxpool4(x11)
        x13 = self.bathNorm4(x12)
        x14 = self.dropout4(x13)

        # Flatten the 3D output as vector
        x15 = self.flatten(x14)
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.outputs[i](x15))
        output = torch.cat(outputlist, dim=1)
        return output