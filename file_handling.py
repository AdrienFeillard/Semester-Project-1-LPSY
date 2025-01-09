import scipy.io
import pandas as pd
import numpy as np


def load_removed_channels(data_struct):
    try:
        # Check if 'removedchans' exists and is non-empty
        if 'removedchans' in data_struct['EEG'][0][0]['chaninfo'].dtype.names:
            removed_chans = data_struct['EEG'][0][0]['chaninfo']['removedchans'][0]

            if len(removed_chans) > 0:  # Ensure it's not empty
                # Extract channel names from the 'labels' field
                removed_chan_names = []
                for chan in removed_chans:
                    # Dynamically access the 'labels' field
                    labels_field = chan['labels'] if 'labels' in chan.dtype.names else chan[3]
                    if labels_field.size > 0:
                        for label_array in labels_field:
                            for label in label_array:
                                removed_chan_names.append(label[0])  # Append the label string

                # Remove duplicates and return
                unique_removed_chan_names = list(np.unique(removed_chan_names))
                return unique_removed_chan_names

    except Exception as e:
        print(f"No bad channels in the data")

    # Return an empty list if no removed channels or on exception
    return []


# Main function to load EEG data
def load_eeg(file_path):
    print(f'Loading EEG from {file_path}')
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['Data'][0][0]

    # Extract EEG data
    eeg_data = data_struct['EEG'][0][0]['data']  # Shape: (128, timepoints, trials)
    labels = data_struct['Labels'][0][0]
    contrast_labels = labels[2]
    frequency_labels = labels[-1]
    chanlocs = data_struct['EEG'][0][0]['chanlocs'][0]

    # Extract electrode positions and names
    ch_pos = {
        chanloc[3][0]: (chanloc[0][0][0], chanloc[1][0][0], chanloc[2][0][0])
        for chanloc in chanlocs
    }

    # Get electrode names
    ch_pos_names = list(ch_pos.keys())[:128]  # First 128 electrode names
    column_names = ch_pos_names + ["Contrast", "Frequency"]  # Add label columns

    # Extract removed channel names
    removed_channel_names = load_removed_channels(data_struct)
    # Create a DataFrame for EEG data
    df = pd.DataFrame(columns=column_names)

    # Loop through trials and construct rows
    for i in range(eeg_data.shape[2]):  # Iterate over trials
        trial_row = {}

        # Extract entire epoch data for each channel
        for j, ch_name in enumerate(ch_pos_names):  # Iterate over electrodes
            trial_row[ch_name] = eeg_data[j, :, i]  # Entire epoch

        # Add labels for the trial
        trial_row["Contrast"] = contrast_labels[i].astype(float)
        trial_row["Frequency"] = frequency_labels[i]

        # Append the row to the DataFrame
        df = pd.concat([df, pd.DataFrame([trial_row])], ignore_index=True)

    return df, ch_pos, removed_channel_names


def parse_dv_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_index = None
    for i, line in enumerate(lines):
        if line.startswith("POOL[0].DATA"):
            start_index = i + 1
            break

    if start_index is None:
        raise ValueError("POOL[0].DATA section not found in the file.")

    data_lines = []
    for line in lines[start_index:]:
        if line.strip() == "DATA_END":
            break
        data_lines.append(line.strip())

    columns = ['POOL', 'LEVEL', 'VALID', 'HITS', 'REACT_TI', 'isEarly', 'isPractice',
               'oriSign', 'answerDir', 'oriDelta', 'contrast', 'gratingPhase',
               'gratingFreq', 'iti', 'stimOnsetTi']
    data = [list(map(float, line.split())) for line in data_lines]
    df = pd.DataFrame(data, columns=columns)

    return df
