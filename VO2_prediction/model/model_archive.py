"""
The following two CNN models are replicated from the publication:
Wen Y, Avrillon S, Hernandez-Pavon JC, Kim SJ, Hug F, Pons JL.
A convolutional neural network to identify motor units from high-density surface electromyography signals in real time.
J Neural Eng. 2021 Apr 6;18(5). doi: 10.1088/1741-2552/abeead. PMID: 33721852.
"""
import torch
from torch import nn
from torch.nn import MaxPool1d


class NeuralInterface_MODCNN(nn.Module):
    """
    Multiple parallel fully connected layers corresponding to each motor unit
    """
    def __init__(self, numChannels, classes, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_MODCNN, self).__init__()
        self.classes = classes
        self.channels = numChannels

        # CNN layers remain the same
        conv1 = torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3)
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(p=0.5)
        self.cnnBlock1 = nn.Sequential(conv1, relu1, conv2, relu2, maxpool2, dropout2)

        conv3 = torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        relu3 = torch.nn.ReLU()
        conv4 = torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3)
        relu4 = torch.nn.ReLU()
        maxpool4 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout4 = nn.Dropout(p=0.5)
        self.cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4)

        # conv5 = torch.nn.Conv1d(in_channels=numNodes[3], out_channels=numNodes[4], kernel_size=3)
        # relu5 = torch.nn.ReLU()
        # conv6 = torch.nn.Conv1d(in_channels=numNodes[4], out_channels=numNodes[5], kernel_size=3)
        # relu6 = torch.nn.ReLU()
        # maxpool5 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        # dropout5 = nn.Dropout(p=0.5)
        # self.cnnBlock3 = nn.Sequential(conv5, relu5, conv6, relu6, maxpool5, dropout5)


        self.flatten = nn.Flatten()

        self.outputs = nn.ModuleList()
        for _ in range(classes):
            self.outputs.append(nn.Sequential(
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        x1 = self.cnnBlock1(x)
        x2 = self.cnnBlock2(x1)
        # x3 = self.cnnBlock3(x2)
        x3 = self.flatten(x2)  # Use the flatten layer
        outputlist = []

        for i in range(self.classes):
            outputlist.append(self.outputs[i](x3))

        output = torch.cat(outputlist, dim=1)
        return output


import torch
from torch import nn
from torch.nn import MaxPool1d


class NeuralInterface_SODCNN(nn.Module):
    """
    The decomposed HD-EMG of each MU
    """
    def __init__(self, numChannels, classes, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_SODCNN, self).__init__()
        self.classes = classes
        self.channels = numChannels

        self.network = nn.ModuleList()
        for _ in range(classes):
            self.network.append(nn.Sequential(
                torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1],
                                                                    kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),
                torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3),
                torch.nn.ReLU(),
                MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),
                nn.Flatten(),
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.network[i](x))

        output = torch.cat(outputlist, dim=1)
        return output

# import the necessary packages
from torch.nn import Conv2d, MaxPool2d
from torch.nn import Conv1d, MaxPool1d
from torch import nn
from torch import tensor
from torch import optim
import torch
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import BinaryF1Score

"""
The following two CNN models are replicated from the publication:
Wen Y, Avrillon S, Hernandez-Pavon JC, Kim SJ, Hug F, Pons JL. 
A convolutional neural network to identify motor units from high-density surface electromyography signals in real time. 
J Neural Eng. 2021 Apr 6;18(5). doi: 10.1088/1741-2552/abeead. PMID: 33721852.
"""
class NeuralInterface_MODCNN(nn.Module):
    """
    Multiple parallel fully connected layers corresponding to each motor unit
    """
    def __init__(self, numChannels, classes, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_MODCNN, self).__init__()
        self.classes = classes
        self.channels = numChannels

        # CNN layers remain the same
        conv1 = torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3)
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(p=0.5)
        self.cnnBlock1 = nn.Sequential(conv1, relu1, conv2, relu2, maxpool2, dropout2)

        conv3 = torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        relu3 = torch.nn.ReLU()
        conv4 = torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3)
        relu4 = torch.nn.ReLU()
        maxpool4 = MaxPool1d(kernel_size=2, stride=2)
        dropout4 = nn.Dropout(p=0.5)
        self.cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4)

        self.flatten = nn.Flatten()

        self.outputs = nn.ModuleList()
        for _ in range(classes):
            self.outputs.append(nn.Sequential(
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        x1 = self.cnnBlock1(x)
        x2 = self.cnnBlock2(x1)
        x3 = self.flatten(x2)  # Use the flatten layer
        outputlist = []

        for i in range(self.classes):
            outputlist.append(self.outputs[i](x3))

        output = torch.cat(outputlist, dim=1)
        return output

class NeuralInterface_SODCNN(nn.Module):
    """
    The decomposed HD-EMG of each MU
    """
    def __init__(self, numChannels, classes, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_SODCNN, self).__init__()
        self.classes = classes
        self.channels = numChannels

        self.network = nn.ModuleList()
        for _ in range(classes):
            self.network.append(nn.Sequential(
                torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1],
                                                                    kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),
                torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3),
                torch.nn.ReLU(),
                MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),
                nn.Flatten(),
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.network[i](x))

        output = torch.cat(outputlist, dim=1)
        return output


"""
The following model is replicated from the publication:
Y. Wen, S. J. Kim, S. Avrillon, J. T. Levine, F. Hug and J. L. Pons, 
"A Deep CNN Framework for Neural Drive Estimation From HD-EMG Across Contraction Intensities and Joint Angles," 
in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 2950-2959, 2022, 
doi: 10.1109/TNSRE.2022.3215246
"""

class NeuralInterface_CST(nn.Module):
    """
    Estimate the cumulative spike train using CNN model
    """
    def __init__(self, numChannels, classes, numOutput = 4, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_CST, self).__init__()
        self.classes = classes
        self.outputSize = numOutput
        self.channels = numChannels

        # CNN layers remain the same
        conv1 = torch.nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3)
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(p=0.5)
        self.cnnBlock1 = nn.Sequential(conv1, relu1, conv2, relu2, maxpool2, dropout2)

        conv3 = torch.nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3)
        relu3 = torch.nn.ReLU()
        conv4 = torch.nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3)
        relu4 = torch.nn.ReLU()
        maxpool4 = MaxPool1d(kernel_size=2, stride=2)
        dropout4 = nn.Dropout(p=0.5)
        self.cnnBlock2 = nn.Sequential(conv3, relu3, conv4, relu4, maxpool4, dropout4)

        self.flatten = nn.Flatten()

        self.outputs = nn.ModuleList()
        for _ in range(numOutput):
            self.outputs.append(nn.Sequential(
                nn.LazyLinear(numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        x1 = self.cnnBlock1(x)
        x2 = self.cnnBlock2(x1)
        x3 = self.flatten(x2)  # Use the flatten layer
        outputlist = []

        for i in range(self.outputSize):
            outputlist.append(self.outputs[i](x3))

        output = torch.cat(outputlist, dim=1)
        return output









