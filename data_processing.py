import numpy as np
import mne

# === Data Processing Functions ===
def preprocess_eeg_data(eeg_df, n_channels):
    eeg_data = eeg_df.iloc[:, :n_channels]
    data_array = np.stack(
        eeg_data.apply(lambda row: np.stack(row.values), axis=1).values
    ).astype(np.float32)
    return data_array

def create_mne_info(eeg_df, ch_pos, sfreq=512):
    info = mne.create_info(ch_names=list(eeg_df.columns[:128]), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
    info.set_montage(montage)
    return info

def validate_channels(info, ch_pos):
    missing_channels = [ch for ch in info['ch_names'] if ch not in ch_pos]
    if missing_channels:
        print(f"Channels missing in montage: {missing_channels}")
    else:
        print("All channels in info have corresponding positions in ch_pos.")


def check_zero_channels(eeg_df, ch_names):
    """
    Identifies channels with all-zero data across all trials.

    Parameters:
        eeg_df (pd.DataFrame): DataFrame containing EEG data where columns are channel names.
        ch_names (list): List of channel names to check.

    Returns:
        list: A list of channel names that are entirely zero.
    """
    zero_channels = []

    for channel in ch_names:
        is_zero = all(np.all(trial == 0) for trial in eeg_df[channel])
        if is_zero:
            zero_channels.append(channel)
    return zero_channels

def check_zero_epochs(data_array):

    zero_epochs = []
    for epoch_idx in range(data_array.shape[0]):
        if np.all(data_array[epoch_idx] == 0):
            zero_epochs.append(epoch_idx)

    return zero_epochs


def filter_electrodes(epochs_Array):

    electrodes_to_keep = []

    electrodes_to_keep.extend([f'A{i}' for i in range(2, 33)])
    electrodes_to_keep.extend([f'B{i}' for i in range(2, 20)])
    electrodes_to_keep.extend([f'D{i}' for i in range(16, 18)])
    electrodes_to_keep.extend([f'D{i}' for i in range(24, 33)])
    # Extract data from p8_epochs_train
    epochs_data = epochs_Array.get_data(copy=True)  # Shape: (n_epochs, n_channels, n_time_points)
    channel_names = epochs_Array.info['ch_names']  # List of channel names

    # Create a mask for channels that should be set to 0 (those not in electrodes_to_keep)
    channels_to_zero = [ch not in electrodes_to_keep for ch in channel_names]

    # Apply the mask: Set data to 0 for channels that are not in electrodes_to_keep
    for i, mask in enumerate(channels_to_zero):
        if mask:
            epochs_data[:, i, :] = 0.0  # Set the entire channel data to 0 for this channel

    # Now create a new MNE Epochs object with the modified data
    return mne.EpochsArray(epochs_data, epochs_Array.info)