import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def sliding_window_mean(eeg_data, vo2_data,  num_eeg_channels, window_size, step_size):
    if eeg_data.shape[0] != num_eeg_channels:
        eeg_data = eeg_data.T
    if vo2_data.shape[0] != 1:
        vo2_data = vo2_data.T
    if eeg_data.shape[1] != vo2_data.shape[1]:
        raise ValueError("The length of EEG and VO2 data must be the same")
    eeg_data_raw = eeg_data
    vo2_data_raw = vo2_data
    window_size_target = window_size
    step_size_target = step_size
    data_len = vo2_data_raw.shape[1]
    frame_index = np.arange(1, data_len - window_size_target + 1, step_size_target)
    frame_len = len(frame_index)
    print(f"The length of the frame is {frame_len}")

    eeg_new_container = np.zeros((frame_len, num_eeg_channels, window_size_target))
    vo2_new_container = np.zeros((frame_len, 1))

    for i in range(frame_len):
        frame_t = frame_index[i]
        eeg_new_container[i, :, :] = eeg_data_raw[:, frame_t:frame_t + window_size_target]
        vo2_new_container[i, :] = np.mean(vo2_data_raw[:, frame_t:frame_t + window_size_target], axis=1)
    
    eeg_windowed = eeg_new_container
    vo2_windowed = vo2_new_container

    return eeg_windowed, vo2_windowed
    




class EEGVO2Dataset(Dataset):
    """
    This dataset class is used to load the windowed EEG and VO2 data
    """
    def __init__(self, eeg_data, vo2_data):
        """
        Initialize the dataset class
        Args:
            eeg_data: the EEG data, shape (num_samples, num_eeg_channels, window_length)
            vo2_data: the VO2 data, shape (num_samples, 1)
        
        """
        self.eeg_data = torch.as_tensor(eeg_data, dtype=torch.float32)
        self.vo2_data = torch.as_tensor(vo2_data, dtype=torch.float32)
        # print(self.eeg_data.shape)
        # print(self.vo2_data.shape)

   
    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.vo2_data)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrives a single sample of EEG and its corresponding VO2 data from the dataset
        Args:
            index: the index of the sample to retrieve
        Returns:
            tuple: (eeg_sample, vo2_sample)
        """
        x_t = self.eeg_data[index]
        y_t = self.vo2_data[index]
        return x_t, y_t




