import glob
import mne
import pandas as pd
import scipy.io
import numpy as np
import ast
import re


def load_electrodes(file_path):
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['Data'][0][0]
    chanlocs = data_struct['EEG'][0][0]['chanlocs'][0]

    # Extract electrode positions and names
    ch_pos = {
        chanloc[3][0]: (chanloc[0][0][0], chanloc[1][0][0], chanloc[2][0][0])
        for chanloc in chanlocs
    }

    return ch_pos


def load_electrodes(file_path):
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['Data'][0][0]
    chanlocs = data_struct['EEG'][0][0]['chanlocs'][0]

    # Extract electrode positions and names
    ch_pos = {
        chanloc[3][0]: (chanloc[0][0][0], chanloc[1][0][0], chanloc[2][0][0])
        for chanloc in chanlocs
    }

    return ch_pos

def load_eeg_data(file_path, ch_pos, sampling_rate=1024, separate_baseline=False):
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['Data'][0][0]
    eeg_data = data_struct['EEG'][0][0]['data']  # Shape: (128, timepoints, trials)
    labels = data_struct['Labels'][0][0]
    contrast_labels = labels[2]
    frequency_labels = labels[-1]

    # Get electrode names
    ch_pos_names = list(ch_pos.keys())[:128]  # First 128 electrode names
    column_names = ch_pos_names + ["Contrast", "Frequency"]  # Add label columns

    if separate_baseline:
        # Calculate the number of samples for the trial and baseline
        trial_samples = int(2 / 3 * sampling_rate)  # First 2 seconds
        baseline_samples = int(1 / 3 * sampling_rate)  # Last 1 second

        # Initialize empty DataFrames
        trial_df = pd.DataFrame(columns=column_names)
        baseline_df = pd.DataFrame(columns=column_names)

        # Loop through trials and construct rows
        for i in range(eeg_data.shape[2]):  # Iterate over trials
            trial_row = {}
            baseline_row = {}

            # Extract trial and baseline data for each channel
            for j, ch_name in enumerate(ch_pos_names):  # Iterate over electrodes
                trial_row[ch_name] = np.array(eeg_data[j, :trial_samples, i])  # First 2 seconds
                baseline_row[ch_name] = np.array(eeg_data[j, -baseline_samples:, i])  # Last 1 second

            # Add labels for the trial
            trial_row["Contrast"] = contrast_labels[i]
            trial_row["Frequency"] = frequency_labels[i]

            # Add labels for the baseline
            baseline_row["Contrast"] = contrast_labels[i]
            baseline_row["Frequency"] = frequency_labels[i]

            # Append the rows to the respective DataFrames
            trial_df = pd.concat([trial_df, pd.DataFrame([trial_row])], ignore_index=True)
            baseline_df = pd.concat([baseline_df, pd.DataFrame([baseline_row])], ignore_index=True)

        return trial_df, baseline_df

    else:
        # If not separating baseline, process the entire epoch
        df = pd.DataFrame(columns=column_names)

        # Loop through trials and construct rows
        for i in range(eeg_data.shape[2]):  # Iterate over trials
            trial_row = {}

            # Extract entire epoch data for each channel
            for j, ch_name in enumerate(ch_pos_names):  # Iterate over electrodes
                trial_row[ch_name] = eeg_data[j, :, i]  # Entire epoch

            # Add labels for the trial
            trial_row["Contrast"] = contrast_labels[i]
            trial_row["Frequency"] = frequency_labels[i]

            # Append the row to the DataFrame
            df = pd.concat([df, pd.DataFrame([trial_row])], ignore_index=True)

        return df

def get_std_eeg_data(df, ch_pos):
    # Extract electrode names
    electrode_names = list(ch_pos.keys())[:128]

    # Calculate the standard deviation for each electrode
    std_values = df.iloc[:, :128].applymap(lambda x: np.std(np.stack(x), axis=0))

    # Create an empty DataFrame with 1024 rows and 128 columns
    std_df = pd.DataFrame(index=range(1024), columns=electrode_names)

    # Fill the DataFrame with the calculated standard deviations
    for idx, row in std_values.iterrows():
        std_df.loc[idx] = row.values.flatten()

    return std_df

def prepare_data(original_df, keep_electrodes=[], keep_columns=[]):
    # Filter the original DataFrame to keep only the specified electrodes
    filtered_df = original_df.loc[:, original_df.columns.intersection(keep_electrodes)]

    # Add the columns to keep
    filtered_df = pd.concat([filtered_df, original_df[keep_columns]], axis=1)

    return filtered_df

def filter_data(original_df, keep_electrodes=[], keep_columns=[]):
    # Filter the original DataFrame to keep only the specified electrodes
    filtered_df = original_df.loc[:, original_df.columns.intersection(keep_electrodes)]

    # Add the columns to keep
    filtered_df = pd.concat([filtered_df, original_df[keep_columns]], axis=1)

    return filtered_df

def parse_dv_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line index where the relevant data starts
    start_index = None
    for i, line in enumerate(lines):
        if line.startswith("POOL[0].DATA"):
            start_index = i + 1
            break

    if start_index is None:
        raise ValueError("POOL[0].DATA section not found in the file.")

    # Read until the next section starts (if there is one)
    data_lines = []
    for line in lines[start_index:]:
        if line.strip() == "DATA_END":
            break
        data_lines.append(line.strip())

    # Create a DataFrame
    columns = ['POOL', 'LEVEL', 'VALID', 'HITS', 'REACT_TI', 'isEarly', 'isPractice',
               'oriSign', 'answerDir', 'oriDelta', 'contrast', 'gratingPhase',
               'gratingFreq', 'iti', 'stimOnsetTi']
    data = [list(map(float, line.split())) for line in data_lines]
    df = pd.DataFrame(data, columns=columns)

    return df

class PowerDataLoader:
    def __init__(self, power_file):
        self.power_file = power_file
        self.power_data = None

    def __enter__(self):
        self.power_data = np.load(self.power_file)
        return self.power_data

    def __exit__(self, exc_type, exc_value, traceback):
        if self.power_data is not None:
            self.power_data.close()
            self.power_data = None

def calculate_and_save_power(df, ch_pos, sfreq=256, band='Alpha', output_file='power_data.npy'):
    # Create MNE Info object
    info = mne.create_info(ch_names=list(df.columns), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    n_epochs = 520
    n_channels = df.shape[1]
    n_times = 1024

    # Define frequency bands
    bands = {'Alpha': (8, 12), 'Beta': (15, 24), 'Gamma': (25, 34)}

    # Check if the chosen band is valid
    if band not in bands:
        raise ValueError(f"Invalid band '{band}'. Choose from: {list(bands.keys())}")

    reshaped_data = np.zeros((n_epochs, n_channels, n_times))
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            reshaped_data[epoch, ch] = df.iloc[epoch, ch]

    # Get frequencies for the selected band
    fmin, fmax = bands[band]
    frequencies = np.arange(fmin, fmax + 1, 1)  # 1 Hz resolution

    power_data = np.zeros((n_epochs, n_channels, n_times), dtype=np.float32)


    # Process data in batches
    batch_size = 50
    for start in range(0, n_epochs, batch_size):
        end = min(start + batch_size, n_epochs)
        batch_data = np.array([reshaped_data[i, :, :] for i in range(start, end)], dtype=np.float32)
        batch_power = mne.time_frequency.tfr_array_morlet(
            batch_data, sfreq=sfreq, freqs=frequencies, n_cycles=1, output='power'
        ).mean(axis=2)  # Average power across frequencies in the band

        power_data[start:end] = batch_power

    # Save the power data to a file
    np.savez(output_file, power_data=power_data)
    print(f'Saved power data to {output_file}')


def get_top_electrodes(ch_pos, sfreq=256.0, power_file='power_data.npy'):
    # Create MNE Info object
    info = mne.create_info(ch_names=list(ch_pos.keys()), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    # Define the ranges of electrodes to keep
    electrodes_to_keep = []

    electrodes_to_keep.extend([f'A{i}' for i in range(2, 33)])
    electrodes_to_keep.extend([f'B{i}' for i in range(2, 20)])
    electrodes_to_keep.extend([f'D{i}' for i in range(16, 18)])
    electrodes_to_keep.extend([f'D{i}' for i in range(24, 33)])

    # Filter the channel positions to include only the specified electrodes
    filtered_ch_pos = {k: v for k, v in ch_pos.items() if k in electrodes_to_keep}

    # Load the power data using the context manager
    with PowerDataLoader(power_file) as power_data:
        power_data = power_data['power_data']

        # Filter the power data to include only the specified electrodes
        power_data_filtered = power_data[[list(ch_pos.keys()).index(e) for e in electrodes_to_keep], :, :]

        # Calculate the standard deviation of power across all times
        std_power = np.std(power_data_filtered, axis=2).mean(axis=0)

        # Calculate the maximum power across all times
        max_power = np.max(power_data_filtered, axis=2).mean(axis=0)

        # Get the top 25 electrodes with the highest mean standard deviation power
        ranked_std = np.argsort(std_power)[::-1]
        ranked_std_electrodes = [list(ch_pos.keys())[i] for i in ranked_std]
        ranked_std_values = std_power[ranked_std]


        ranked_max = np.argsort(max_power)[::-1]
        ranked_max_electrodes = [list(ch_pos.keys())[i] for i in ranked_max]
        ranked_max_values = max_power[ranked_max]

        filtered_std_electrodes = [e for e in ranked_std_electrodes if e in electrodes_to_keep]
        filtered_std_values = [ranked_std_values[ranked_std_electrodes.index(e)] for e in filtered_std_electrodes]
        filtered_std_electrodes = [e for e in ranked_std_electrodes if e in electrodes_to_keep]
        filtered_std_values = [ranked_std_values[ranked_std_electrodes.index(e)] for e in filtered_std_electrodes]

        filtered_max_electrodes = [e for e in ranked_max_electrodes if e in electrodes_to_keep]
        filtered_max_values = [ranked_max_values[ranked_max_electrodes.index(e)] for e in filtered_max_electrodes]

        filtered_std_df = pd.DataFrame({'Electrode': filtered_std_electrodes, 'Mean Std Power': filtered_std_values})
        filtered_max_df = pd.DataFrame({'Electrode': filtered_max_electrodes, 'Max Power': filtered_max_values})

        return filtered_std_df, filtered_max_df

"""def load_results(output_prefix, model_name, participant_number):
    # Construct the glob pattern to match the CSV files
    pattern = ''
    if model_name.lower() == 'lda':
        pattern = f'{output_prefix}P{participant_number}__*_LDA_solver_*.csv'
    elif model_name.lower() == 'svm':
        pattern = f'{output_prefix}P{participant_number}__*_SVM_C_*_gamma_*.csv'
    elif model_name.lower() == 'cnn':
        pattern = f'{output_prefix}P{participant_number}__*_CNN_dropout_*_activation_*_last_activation_*_optimizer_*_loss_*.csv'

    # Use glob to find all matching files
    file_paths = glob.glob(pattern)
    results = []
    hyperparameters = []

    for file_path in file_paths:
        # Extract hyperparameters from the file name
        match = None
        if model_name.lower() == 'lda':
            match = re.search(r'solver_(\w+)', file_path)
        elif model_name.lower() == 'svm':
            match = re.search(r'C_([\d\.]+)_gamma_([\d\.]+)', file_path)
        elif model_name.lower() == 'cnn':
            match = re.search(r'dropout_([\d\.]+)_activation_(\w+)_last_activation_(\w+)_optimizer_(\w+)_loss_(\w+)', file_path)

        if match:
            hyperparameters.append(match.groups())
            # Load each CSV file into a DataFrame and store them in a list
            df = pd.read_csv(file_path)
            results.append(df)
        else:
            print(f"Warning: No match found for file {file_path}")

    print(f"Loaded results: {results}")
    print(f"Hyperparameters: {hyperparameters}")

    return results, hyperparameters"""

def load_results(output_prefix, model_name, participant_number):
    # List of SVM file paths
    svm_file_paths = [
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_10_gamma_auto.csv',
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_1_gamma_auto.csv',
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_01_gamma_auto.csv',
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_10_gamma_scale.csv',
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_1_gamma_scale.csv',
        'C:\\Users\\Home\\Documents\\EPFL\\Semester project\\Code\\Tuned_trained_data\\P8__SVM_C_01_gamma_scale.csv'
    ]
    
    results = []
    hyperparameters = []

    for file_path in svm_file_paths:
        # Extract hyperparameters from the file name
        match = re.search(r'C_([\d\.]+)_gamma_([\d\.]+)', file_path)

        if match:
            hyperparameters.append(match.groups())
            # Load each CSV file into a DataFrame and store them in a list
            df = pd.read_csv(file_path)
            results.append(df)
        else:
            print(f"Warning: No match found for file {file_path}")

    print(f"Loaded results: {results}")
    print(f"Hyperparameters: {hyperparameters}")

    return results, hyperparameters