import scipy
import pandas as pd


# === File Handling Functions ===
def load_eeg(file_path):
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['Data'][0][0]
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


    # If not separating baseline, process the entire epoch
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

    return df,ch_pos


