import pandas as pd
import mne
import numpy as np
import logging
import traceback


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

def read_bdf_file(file_path, trim_flag:bool=False, new_time_duration:float=293, logger:logging.Logger=None):
    """
    Load the EEG data from the bdf file and trim the first t seconds to match the VO2 data (optional)
    """
    try:
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        time_duration = raw.times[-1]
        fsamp = raw.info['sfreq']
        raw_numpy = raw.get_data()
        new_time_duration_default = new_time_duration
        num_channels = raw_numpy.shape[0]
        if trim_flag:
            if logger is not None:
                logger.info("Trimming the EEG data per request")
            else:
                print("Trimming the EEG data per request")
            samples_to_trim = int(10 * fsamp)
            trimmed_raw = raw.copy().crop(tmin=raw.times[-1]-new_time_duration_default)
            if trimmed_raw.times[-1] != new_time_duration_default:
                new_time_duration_returned = trimmed_raw.times[-1]
                if logger is not None:
                    logger.info(f"The new time duration of the EEG data is set to {new_time_duration_returned} because the original time duration {new_time_duration_default} doesn't match.")
                else:
                    print(f"The new time duration of the EEG data is set to {new_time_duration_returned} because the original time duration {new_time_duration_default} doesn't match.")
            else:
                new_time_duration_returned = new_time_duration_default
            trimmed_raw_numpy = trimmed_raw.get_data()
        else:
            new_time_duration_returned = new_time_duration_default
            trimmed_raw_numpy = np.zeros((raw_numpy.shape[0], int(new_time_duration_default * fsamp)))
            trimmed_raw = trimmed_raw_numpy
        
        return num_channels, fsamp, time_duration, new_time_duration_returned, raw, raw_numpy, trimmed_raw, trimmed_raw_numpy
    except Exception as e:
        if logger is not None:
            logger.error(f"Error reading BDF file: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"Error reading BDF file: {e}")
            print(traceback.format_exc())

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



