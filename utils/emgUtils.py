from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Tuple
import scipy.io as sio
from os import path
import numpy as np
import os
import torch


class HDemgLoaderFlat:
    def __init__(self, TR: str = "5_50_GM", SG: Tuple[int, int, int]=(0, 1, 2), ST=10,
                 MU=1, WS=120, TF=0):

        # All of the data show save in the dataset folder
        self.root = path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
        self.trial = TR
        self.seg = [1, 2, 3]
        self.SG = SG
        self.ST = ST
        self.MU = MU
        self.WS = WS
        self.TF = TF
    def load_mat_dataset(self):
        # create a list in case multiple segments will be needed
        matfile = []
        # Check if the mat files are in the correct path
        for i, sg in enumerate(self.SG):
            prefix = "{}-SG{}-WS{}-ST{}".format(self.trial,
                                            self.seg[sg],
                                            self.WS,
                                            self.ST)
            # print(prefix)
            matfile.append(path.join(self.root, prefix))

        # Load the data
        # Define the label of data to extract
        vname = ["EMGs", "Spikes"]

        # Load mat files and extract data
        data = [sio.loadmat(path, variable_names=vname) for path in matfile]

        # Concatenate EMGs and Spikes
        x_data = np.concatenate([d["EMGs"] for d in data], axis=0)
        print("The shape of training set is: {}".format(x_data.shape))
        spikes = np.concatenate([d["Spikes"] for d in data], axis=0)

        # Extract the spikes for given motor units
        # Therefore the training data can be either organized as spikes for all motor units
        # or spikes for a single motor unit.
        if isinstance(self.MU, list):
            y_data = [spikes[:, c] if c < spikes.shape[1] else np.zeros_like(spikes[:, 0]) for c in self.MU]
        else:
            y_data = [spikes[:, self.MU]]
        y_data = np.array(y_data)
        y_data = y_data.T

        if self.TF > 0:
            if self.TF == 1:
                x_data, y_data = shuffle(x_data, y_data)
            else:
                x_data, _, y_data, _ = train_test_split(x_data, y_data, train_size=self.TF)
        else:
            print('no shuffle')

        y_data = y_data.T
        y_data.T.tolist()

        return x_data, y_data


def EMG_Loader_2D(MU, TR: str = "5_50_GM", SG:list[int, int, int]=[1, 2, 3], ST:list=[10], WS:list=[120]):
    # create a list in case multiple segments will be needed
    trial = TR
    root = path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
    segs =SG
    matFile = []
    for _, sg in enumerate(segs):
        for _, step in enumerate(ST):
            for _, window in enumerate(WS):
                prefix = "{}-SG{}-WS{}-ST{}".format(trial, sg, window, step)
                matFile.append(path.join(root, prefix))
    vname = ["EMGs", "Spikes"]

    # Load mat files and extract data
    data = [sio.loadmat(path, variable_names=vname) for path in matFile]
    # Concatenate EMGs and Spikes
    x_data = np.concatenate([d["EMGs"] for d in data], axis=0)
    x_data = x_data.transpose(0, 2, 1)

    spikes = np.concatenate([d["Spikes"] for d in data], axis=0)

    # Extract the spikes for given motor units
    # Therefore the training data can be either organized as spikes for all motor units
    # or spikes for a single motor unit.
    if isinstance(MU, list):
        y_data = [spikes[:, c] if c < spikes.shape[1] else np.zeros_like(spikes[:, 0]) for c in MU]
    else:
        y_data = [spikes[:, MU]]
    y_data = np.array(y_data)
    y_data = y_data.T
    print("\rThe training data for one-dimensional CNN is loaded!")
    print("\rThe shape of training set is: {}".format(x_data.shape))
    print("\rThe shape of label set is: {}".format(y_data.shape))
    return x_data, y_data

def EMG_Loader_3D(TR: str = "5_50_GM", SG: list[int, int, int]=[1, 2, 3],
                  MU: list=[0, 1, 2, 3], WS: list=[120], toResample:bool=False,
                  steps: list = [20], inspect: bool=True, pad: bool=False):

    def resampleEMG(data, winLen, stepLen, idx=1):
        """
        Resample the EMG along the direction of time and calculate RMS
        :param data:
        :param winLen:
        :param stepLen:
        :param idx:
        :return:
        """
        numWindows = (data.shape[idx] - winLen) // stepLen + 1
        if idx == 1:
            resampleResult = np.empty((data.shape[0], numWindows,data.shape[2], data.shape[3]))
            # Calculate RMS for each window along the second dimension
            for i in range(numWindows):
                # Define the start and end of the current window
                start = i * stepLen
                end = start + winLen

                # Extract the window, calculate the RMS along the windowed dimension, and store it
                resampleResult[:, i, :, :] = np.sqrt(np.mean(np.square(data[:, start:end, :, :]), axis=1))
        else:
            assert "Can only resample teh 2nd dimension"
        return resampleResult
    def pad_data(unpaddedData, targetSize: list):
        """
        This function pad the third and fourth dimension of the training set
        if the third and fourth dimension are not equal.
        """
        sizeUnpadded = unpaddedData.shape
        sizeX = sizeUnpadded[2]
        sizeY = sizeUnpadded[3]
        deltaX = abs(sizeX - targetSize[0])
        deltaY = abs(sizeY - targetSize[1])
        padWidth = [(0, 0), (0, 0), (int(deltaX/2), int(deltaX/2)), (int(deltaY/2), int(deltaY/2))]
        paddedData = np.pad(unpaddedData, pad_width=padWidth, mode='constant', constant_values=0)
        assert paddedData.shape[2] == paddedData.shape[3], \
            f"Error: Third and fourth dimensions are not equal: {padded_data.shape[2]} != {padded_data.shape[3]}"
        return paddedData

    def inspect_data(rawData, processedData, dataLen, winLen):
        """
        Inspect the reshaped data
        """
        # Define a flag to track if all sampled comparisons match
        all_samples_match = True
        # Iterate over each element in the first (5230) and second (64) dimensions
        for i in range(dataLen):
            for j in range(64):
                # Sample a 1x120 element from the raw data
                raw_sample = rawData[i, j, :]

                # Retrieve the corresponding element from reshaped_data
                reshaped_sample = processedData[i, :, j // 5, j % 5]

                # Compare the sampled element
                if not np.array_equal(raw_sample, reshaped_sample):
                    print(f"Mismatch found at element ({i}, {j}) by method 1")
                    all_samples_match = False
                    break  # Exit early if a mismatch is found

            if not all_samples_match:
                break

        # Final result
        if all_samples_match:
            print("All sampled elements match between the raw and reshaped data by method 1.")
        else:
            print("Some sampled elements do not match between the raw and reshaped data by method 1.")

        # Step 1: Reverse the transpose to get back the padded 5230 x 13 x 5 x 120 shape
        reversed_data = reshaped_data.transpose(0, 2, 3, 1).reshape(dataLen, 65, winLen)

        # Step 2: Remove the padding by slicing the first 64 elements in the second dimension
        recovered_data = reversed_data[:, :64, :]

        # Step 3: Compare the recovered data with the original raw data
        if np.array_equal(recovered_data, emgContSeg):
            print("The reshaped data matches the original raw data by method 2!")
        else:
            print("There is a discrepancy between the reshaped data and the original raw data by method 2.")
    trial = TR
    root = path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
    segs =SG
    matFile = []
    for _, sg in enumerate(segs):
        for _, step in enumerate(steps):
            for _, window in enumerate(WS):
                prefix = "{}-SG{}-WS{}-ST{}".format(trial, sg, window, step)
                matFile.append(path.join(root, prefix))
    vname = ["EMGs", "Spikes"]

    # Load mat files and extract data
    data = [sio.loadmat(path, variable_names=vname) for path in matFile]
    # Concatenate EMGs
    emgContSeg = np.concatenate([d[vname[0]] for d in data], axis=0)
    print("The shape of the original EMG Signal: ", emgContSeg.shape)
    emgContSeg=emgContSeg.transpose(0, 2, 1)
    print("The shape of the original EMG Signal after transpose: ", emgContSeg.shape)
    # Step 1: Pad the last dimension to 65
    padded_data = np.pad(emgContSeg, ((0, 0), (0, 1), (0, 0)), 'constant')
    # Step 2: Reshape each 65-element vector into a 13x5 grid
    dataLen = padded_data.shape[0]
    winLen = padded_data.shape[2]
    reshaped_data = padded_data.reshape(dataLen, 13, 5, winLen)
    # Step 3: If needed, reorder dimensions to (dataLen, winLen, 13, 5) for compatibility
    reshaped_data = reshaped_data.transpose(0, 3, 1, 2)
    x_data = reshaped_data
    raw_xData = x_data
    print("The shape of the remapped data is: ", x_data.shape, "\n")
    # Extract the spikes for given motor units
    # Therefore the training data can be either organized as spikes for all motor units
    # or spikes for a single motor unit.
    spikes = np.concatenate([d["Spikes"] for d in data], axis=0)
    if isinstance(MU, list):
        y_data = [spikes[:, c] if c < spikes.shape[1] else np.zeros_like(spikes[:, 0]) for c in MU]
    else:
        y_data = [spikes[:, MU]]
    y_data = np.array(y_data)
    y_data = y_data.T
    print("The shape of the label array is: ", y_data.shape, "\n")
    if toResample:
        xDataResampled = resampleEMG(x_data, winLen=20, stepLen=10, idx=1)
        print("\rData resampled, the shape of resampled data is: ", xDataResampled.shape)
        # print(xDataResampled[1, 1, :, 1])
    else:
        xDataResampled = np.zeros(x_data.shape)
    if inspect:
        inspect_data(emgContSeg, reshaped_data, dataLen, winLen)
    else:
        pass
    if pad:
        if toResample:
            xDataPadded = pad_data(xDataResampled, [27, 27])
        else:
            xDataPadded = pad_data(x_data, [27, 27])
        print("\rData padded, the shape of padded data is: ", xDataPadded.shape)
    else:
        xDataPadded = np.zeros((x_data.shape[0], x_data.shape[1], 27, 27))
    return raw_xData, xDataResampled, xDataPadded, y_data



if __name__ == "__main__":
    pass
    # a, b, c, d = EMG_Loader_3D(steps=[20])
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)