import pandas as pd
import mne
import torch
import numpy as np

def extract_vo2_data(file_path, start_row=147, time_col='J', vo2_col='K'):
    df = pd.read_excel(file_path, skiprows=147)

    time = df.iloc[:, 9].tolist()
    vo2 = df.iloc[:, 10].tolist()
    return pd.DataFrame({'Time': time, 'VO2': vo2})


def read_bdf_file(file_path):
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    return raw

def interpolate_vo2_data(vo2_df):    
    vo2_df = vo2_df.copy()
    
    # Create a new index directly from the original time values
    min_time = int(vo2_df['Time'].min())
    max_time = int(vo2_df['Time'].max())
    new_index = pd.Series(range(min_time, max_time + 1))
    
    # Create a new DataFrame with the interpolated values
    new_df = pd.DataFrame({'Time': new_index})
    new_df = new_df.merge(vo2_df, on='Time', how='left')
    new_df['VO2'] = new_df['VO2'].interpolate(method='polynomial', order=3)
    
    return new_df

def interpolate_vo2_data(vo2_df):    
    vo2_df = vo2_df.copy()
    
    # Create a new index directly from the original time values
    min_time = int(vo2_df['Time'].min())
    max_time = int(vo2_df['Time'].max())
    new_index = pd.Series(range(min_time, max_time + 1))
    
    # Create a new DataFrame with the interpolated values
    new_df = pd.DataFrame({'Time': new_index})
    new_df = new_df.merge(vo2_df, on='Time', how='left')
    new_df['VO2'] = new_df['VO2'].interpolate(method='polynomial', order=3)
    
    return new_df


if __name__ == "__main__":
    raw_data  = read_bdf_file(r"C:\Users\Jerry Fu\Documents\GitHub\EEG_Biomechanics_Project\VO2_prediction\EEG\Pilot4Oct_OG_A.bdf")
    sampling_frequecy = raw_data.info['sfreq']
    samples_to_trim = int(10*sampling_frequecy)

    # trim last 10 seconds
    raw_data_trimmed = raw_data.copy().crop(tmax=raw_data.times[-samples_to_trim - 1])
    print(f"New duration: {raw_data_trimmed.times[-1]} seconds")

    raw_data_trimmed.get_data().shape


    path  = r"C:\Users\Jerry Fu\Documents\GitHub\EEG_Biomechanics_Project\VO2_prediction\Metabolics\pilot4Oct_OG_A.xlsx"
    vo2_df = extract_vo2_data(path)
    vo2_df['Time'] = pd.TimedeltaIndex(vo2_df['Time'].astype("str")).total_seconds()

    vo2_time = vo2_df['Time'].to_numpy()
    vo2_value = vo2_df['VO2'].to_numpy()
    print(vo2_df)
