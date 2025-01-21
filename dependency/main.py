import os
import sys
from datetime import datetime
import torch
from torch import nn, optim

from NMI.nmi_3d_v1 import Trainer3DCNN
from NMI.nmi_1d_v2 import Trainer1DCNN
from torchsummary import summary
from utils.emgUtils import EMG_Loader_3D, EMG_Loader_2D
from model.model_mo_3d import NMI_3D_Lite, NMI_3D
from model.model_1DCNN import MODCNN_1D_MUST

from utils import plotter

def main(epochNum:int=200, batchSize:list=[64, 64], splitRat=0.2, expName:str="exp", trialNum=1,
         muNum:int=4, mode:str="3DMUST", WS:list=[120], ST:list=[20], SG:list=[1, 2, 3],
         resample:bool=True, padding:bool=True, lightweight:bool=True, cnnMode:str="MO"):
    def directory_inspect(directory_path="ckpts"):
        """
        Check if a directory exists. If not, create it.
        Args:
        directory_path (str): The path of the directory to check/create.
        """
        if not os.path.exists(directory_path):
            try:
                os.makedirs(directory_path)
                print(f"Directory created successfully: {directory_path}")
            except OSError as e:
                print(f"Error creating directory {directory_path}: {e}")
        else:
            print(f"Directory already exists: {directory_path}")
    def save_model(model, exp_name):
        modelName = exp_name + "_cnn.pth"
        pathModel = os.path.join("ckpts", modelName)
        torch.save(model.to("cpu").state_dict(), pathModel)
        print("\rThe trained CNN model has been saved in: ", pathModel)
    def write_summary(model, textName, trainData, chNum=1, threeDim:bool=True, path="summary"):
        summaryFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        directory_inspect(summaryFolder)
        summaryPath = os.path.join(summaryFolder, textName)
        if os.path.exists(summaryPath):
            print(f"File '{summaryPath}' exists. It will be overwritten.")
        else:
            print(f"File '{summaryPath}' does not exist. It will be created.")
        with open(summaryPath, "w") as f:
            # Redirect stdout to the file
            sys.stdout = f
            if threeDim:
                summary(model, input_size = (chNum, trainData.shape[1], trainData.shape[2], trainData.shape[3]), device="cuda")
            else:
                summary(model, input_size=(chNum, trainData.shape[2]), device="cuda")
            sys.stdout = sys.__stdout__  # Reset stdout back to normal
        print(f"Model summary saved to {summaryPath}.")

    mu = [i for i in range(muNum)]
    today_date = datetime.today().date()
    # Convert date to string in the format YYYY-MM-DD
    date_string = today_date.strftime("%Y-%m-%d")
    trial = '5_50_GM'
    exp_name = expName + ("_lte_" if lightweight else "_") + mode + "_" + str(trialNum) + "_" + str(muNum) + "_" + date_string
    # Define the step size and window size to load the corresponding
    # mat file
    step_size = ST
    window_size = WS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    textName = exp_name + "_model_summary.txt"
    criterion = nn.BCELoss()
    if mode.upper() == "3DMUST":
        # load train data
        x_data, _, _, y_data = EMG_Loader_3D(TR=trial, SG=SG, steps=step_size,WS=window_size, MU=mu, toResample=resample,pad=padding)
        print("3D HD-EMG Data import is done with {} and {}".format("resample" if resample else "no resample",
                                                                    "padding" if padding else "no padding"))
        if cnnMode == "MO":
            if lightweight:
                cnnModel = NMI_3D_Lite(numChannels=1, classes=muNum).to(device)
            else:
                cnnModel = NMI_3D(numChannels=1, classes=muNum).to(device)
        elif cnnMode == "SO":
            assert "Single output 3-D CNN model is still under construction"

        write_summary(cnnModel, textName, x_data, chNum=1, threeDim=True)

        optimizer3D = optim.RMSprop(cnnModel.parameters(), lr=0.001, alpha=0.9)
        Trainer3D = Trainer3DCNN(device=device, model=cnnModel, x_data=x_data,y_data=y_data,criterion=criterion,
                               optimizer=optimizer3D,classNum=muNum, batch_size=batchSize, splitRat=splitRat)

        result, optimalCNN = Trainer3D(num_epochs=epochNum)
    elif mode.upper() == "1DMUST":
        x_data,  y_data = EMG_Loader_2D(MU=mu, TR= "5_50_GM", SG=SG, ST=ST, WS=WS)
        if cnnMode.upper() == "MO":
            cnnModel = MODCNN_1D_MUST(numChannels=x_data.shape[1], classes=muNum).to(device)
        else:
            assert "Other modes are not ready"
        write_summary(cnnModel, textName, x_data, chNum=x_data.shape[1], threeDim=False)
        optimizer1D = optim.RMSprop(cnnModel.parameters(), lr=0.001, alpha=0.9)
        Trainer1D = Trainer1DCNN(device=device, model=cnnModel, x_data=x_data, y_data=y_data, numMu=muNum, optimizer=optimizer1D,
                                 criterion=criterion, test_size=splitRat, batch_size=batchSize)
        result, optimalCNN = Trainer1D(epoches=epochNum)


    resultName = expName +("_lte_" if lightweight else "_")+ mode+ "_" + str(trialNum) + "_" + str(muNum) + ("_MO_" if mode.upper() == "MO" else "_SO_")
    save_model(model=optimalCNN, exp_name=resultName)
    return result, resultName


if __name__ == "__main__":
    # result = main()
    result, resultName = main(epochNum=200, batchSize=[64, 64], splitRat=0.2, expName="exp3DCNN", trialNum=7,
         muNum=4, mode="3DMUST", WS=[120], ST=[20], SG=[1, 2, 3],
         resample=False, padding=False, lightweight=True, cnnMode="MO")
    plotter.log_iter_result(data=result, exp_name=resultName)
    # Clean up GPU memory
    torch.cuda.empty_cache()