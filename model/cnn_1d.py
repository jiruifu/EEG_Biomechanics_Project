import torch
from torch import nn
from torch.nn import MaxPool1d

class CNN_VO2_1D(nn.Module):
    def __init__(self, numChannels, window_size, numNodes=[128, 128, 128, 64, 256]):
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
        with torch.no_grad():
            x_t = torch.zeros(1, numChannels, window_size)
            x1_t = self.cnnBlock1(x_t)
            x2_t = self.cnnBlock2(x1_t)
            flat_size = torch.flatten(x2_t, start_dim=1).shape[1]

        # Regression output
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, numNodes[4]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(numNodes[4], 1),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.cnnBlock1(x)
        x = self.cnnBlock2(x)
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    model = CNN_VO2_1D(numChannels=144, numNodes=[128, 128, 128, 64, 256])
    print(model)
