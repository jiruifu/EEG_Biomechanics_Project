import pandas as pd
import mne
import numpy as np
import logging
import traceback
from scipy import signal


def extract_vo2_data(file_path, start_row=147, time_col='J', vo2_col='K', logger:logging.Logger=None):
    try:
        df = pd.read_excel(file_path, skiprows=147)
        time = df.iloc[:, 9].tolist()
        vo2 = df.iloc[:, 10].tolist()
        tentative_vo2_df = pd.DataFrame({'Time': time, 'VO2': vo2})
        tentative_vo2_df['Times'] = pd.TimedeltaIndex(tentative_vo2_df['Time'].astype("str")).total_seconds()
        vo2_np = tentative_vo2_df.to_numpy()
        vo2_time = tentative_vo2_df['Times'].to_numpy()
        vo2_value = tentative_vo2_df['VO2'].to_numpy()
        processed_vo2_df = pd.DataFrame({'Time': vo2_time, 'VO2': vo2_value})
        if logger is not None:
            logger.info("VO2 data extracted successfully")
            logger.info(f"The shape of the VO2 data is {processed_vo2_df.shape}")
            logger.info(f"The time duration of the VO2 data is {processed_vo2_df['Time'].max() - processed_vo2_df['Time'].min()}")
        else:
            print("VO2 data extracted successfully")
            print(f"The shape of the VO2 data is {processed_vo2_df.shape}")
            print(f"The time duration of the VO2 data is {processed_vo2_df['Time'].max() - processed_vo2_df['Time'].min()+1}")
        return processed_vo2_df
    except Exception as e:
        if logger is not None:
            logger.error(f"Error extracting VO2 data: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"Error extracting VO2 data: {e}")
            print(traceback.format_exc())

def extract_mc_data_csv(file_path):
    def time_to_seconds(time_str):
        try:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes * 60 + seconds
        except:
            return None
    df = pd.read_csv(file_path)
    df["seconds"] = df["t"].apply(time_to_seconds)
    df['t'] = df['seconds']
    df = df.drop('seconds', axis=1)
    time_duration = df['t'].max()
    return df, time_duration


def read_bdf_file(file_path, trim_flag:bool=False, new_time_duration:float=None):
    """
    Load the EEG data from the bdf file and trim the first t seconds to match the VO2 data (optional)
    """
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    time_duration = raw.times[-1]
    fsamp = raw.info['sfreq']
    raw_numpy = raw.get_data()
    new_time_duration_default = new_time_duration
    num_channels = raw_numpy.shape[0]
    if trim_flag:
        print("Trimming the EEG data per request")
        samples_to_trim = int(10 * fsamp)
        trimmed_raw = raw.copy().crop(tmin=raw.times[-1]-new_time_duration_default)
        if trimmed_raw.times[-1] != new_time_duration_default:
            new_time_duration_returned = trimmed_raw.times[-1]
            print(f"The new time duration of the EEG data is set to {new_time_duration_returned} because the original time duration {new_time_duration_default} doesn't match.")
        else:
            new_time_duration_returned = new_time_duration_default
        trimmed_raw_numpy = trimmed_raw.get_data()
    else:
        new_time_duration_returned = new_time_duration_default
        trimmed_raw_numpy = np.zeros((raw_numpy.shape[0], int(new_time_duration_default * fsamp)))
        trimmed_raw = trimmed_raw_numpy
    
    return num_channels, fsamp, time_duration, new_time_duration_returned, raw, raw_numpy, trimmed_raw, trimmed_raw_numpy

def downsample_eeg(eeg_data, fsamp, new_fsamp, filter:bool=False):
    """
    Downsample the EEG data to the new sampling rate

    Parameters:
        eeg_data: the EEG data, shape (num_channels, num_samples)
        fsamp: the original sampling rate
        new_fsamp: the new sampling rate
    
    Returns:
        downsampled_eeg: the downsampled EEG data, shape (num_channels, num_samples)
    """
    n_channels, n_samples = eeg_data.shape
    downsample_factor = fsamp // new_fsamp
    if downsample_factor < 1:
        raise ValueError("The new sampling rate is greater than the original sampling rate")
    
    if filter:
        nyquist = new_fsamp/2
        filter_order = 5
        cutoff_freq = nyquist / (fsamp/2)
        b = signal.firwin(filter_order, cutoff_freq)
        a = 1
        n_samples_new = n_samples // downsample_factor
        downsampled_eeg = np.zeros((n_channels, n_samples_new))
        for i in range(n_channels):
            filtered_eeg = signal.lfilter(b, a, eeg_data[i, :], axis=0)
            downsampled_eeg[i, :] = filtered_eeg[::downsample_factor]
    else:
        downsampled_eeg = eeg_data[:, ::fsamp//new_fsamp]

    downsampled_eeg = eeg_data[:, ::fsamp//new_fsamp]
    return downsampled_eeg
    

def interpolate_vo2_data(vo2_df:pd.DataFrame=None, target_length:float=None, method:str="two", logger:logging.Logger=None):
    """
    Interpolate the VO2 data to match the target time duration and sampling rate of EEG data
    """
    def method_one():
        """
        Method 1: This method only interpolate the VO2 data to unify the step size as 1 second
        """
        vo2_df_raw = vo2_df.copy()

        # Create a new index directly from the original time values
        min_time = int(vo2_df_raw['Time'].min())
        max_time = int(vo2_df_raw['Time'].max())
        new_index = pd.Series(range(min_time, max_time + 1))
        vo2_df_interpolated = pd.DataFrame({'Time': new_index})
        vo2_df_interpolated = vo2_df_interpolated.merge(vo2_df_raw, on='Time', how='left')
        vo2_df_interpolated["VO2"] = vo2_df_interpolated["VO2"].interpolate(method='polynomial', order=3)
        return vo2_df_interpolated

    def method_two():
        """
        Method 2: This method interpolate the VO2 data to match the length of EEG data
        """
        vo2_df_raw = vo2_df.copy()

        # Create a new index directly from the original time values
        min_time = int(vo2_df_raw['Time'].min())
        max_time = int(vo2_df_raw['Time'].max())
        new_index = pd.Series(np.linspace(min_time, max_time, target_length))
        vo2_df_interpolated = pd.DataFrame({'Time': new_index})
        vo2_df_interpolated = vo2_df_interpolated.merge(vo2_df_raw, on='Time', how='left')
        vo2_df_interpolated["VO2"] = vo2_df_interpolated["VO2"].interpolate(method='polynomial', order=3)
        return vo2_df_interpolated
    
    try:
        if vo2_df is None:
            raise ValueError("The dataframe of the VO2 data is missing")
        
        else:
            if method.lower() == "one":
                if logger is not None:
                    logger.info("Interpolating the VO2 data using method 1")
                else:
                    print("Interpolating the VO2 data using method 1")
                vo2_df_interpolated = method_one()
            elif method.lower() == "two":
                if logger is not None:
                    logger.info("Interpolating the VO2 data using method 2")
                else:
                    print("Interpolating the VO2 data using method 2")
                vo2_df_interpolated = method_two()
            else:
                raise ValueError("The method is not supported")
        
        vo2_array = vo2_df_interpolated["VO2"].to_numpy()
        vo2_array = vo2_array.reshape(1, -1)
        if logger is not None:
            logger.info(f"The shape of the interpolated VO2 data is {vo2_array.shape}")
            logger.info(f"The time duration of the interpolated VO2 data is {vo2_df_interpolated['Time'].max() - vo2_df_interpolated['Time'].min()}")
            logger.info(f"The time step of the interpolated VO2 data is {vo2_df_interpolated['Time'].diff().min()}")
            logger.info(f"The time step of the interpolated VO2 data is {vo2_df_interpolated['Time'].diff().max()}")
        else:
            print(f"The shape of the interpolated VO2 data is {vo2_array.shape}")
            print(f"The time duration of the interpolated VO2 data is {vo2_df_interpolated['Time'].max() - vo2_df_interpolated['Time'].min()}")
            print(f"The time step of the interpolated VO2 data is {vo2_df_interpolated['Time'].diff().min()}")
            print(f"The time step of the interpolated VO2 data is {vo2_df_interpolated['Time'].diff().max()}")
        return vo2_df_interpolated, vo2_array
    except Exception as e:
        if logger is not None:
            logger.error(f"Error interpolating VO2 data: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"Error interpolating VO2 data: {e}")
            print(traceback.format_exc())

# if __name__ == "__main__":
#     from dataset import EEGVO2Dataset, sliding_window_mean
#     # Load the data
#     eeg_path = r"C:\Users\Jerry Fu\Documents\eeg_dataset\Pilot4_data\EEG\Pilot4Oct_OG_A.bdf"
#     vo2_path = r"C:\Users\Jerry Fu\Documents\eeg_dataset\Pilot4_data\Metabolics\pilot4Oct_OG_A.xlsx"

#     # Read the EEG data
#     num_eeg_channels, fsamp, time_duration, new_time_duration_returned, raw, raw_numpy, trimmed_raw, trimmed_raw_numpy = read_bdf_file(eeg_path, trim_flag=True, new_time_duration=293)

#     # Extract the VO2 data
#     vo2_df = extract_vo2_data(vo2_path)

#     # Interpolate the VO2 data
#     vo2_df_interpolated, vo2_array = interpolate_vo2_data(vo2_df, target_length=trimmed_raw_numpy.shape[1], method="two")

#     # Define the x, and y
#     x = trimmed_raw_numpy
#     y = vo2_array

#     print(x.shape)
#     print(y.shape)

#     eeg_windowed, vo2_windowed = sliding_window_mean(x, y, num_eeg_channels, 64, 32)

#     dataset = EEGVO2Dataset(eeg_windowed, vo2_windowed)

#     sample = dataset.__getitem__(0)

#     eeg_tensor = sample[0]
#     vo2_tensor = sample[1]

#     print(eeg_tensor.shape)
#     print(vo2_tensor)



