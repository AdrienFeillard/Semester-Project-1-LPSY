import numpy as np
import mne
import warnings
# === Data Processing Functions ===
import warnings

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Fiducial point nasion not found.*")

def create_epochs(eeg_df, n_channels, ch_pos, bad_channels=None, baseline=(2.0, 3.0), sfreq=512):
    """
    Create MNE Epochs from EEG data, dropping bad channels and zero channels.

    Parameters:
    - eeg_df: DataFrame containing EEG data.
    - n_channels: Number of channels.
    - ch_pos: Dictionary of channel positions.
    - bad_channels: List of bad channel names to drop (optional).
    - baseline: Baseline correction tuple.
    - sfreq: Sampling frequency.

    Returns:
    - epochs: MNE Epochs object.
    - updated_df: DataFrame with rows corresponding to the remaining epochs.
    """
    if bad_channels is None:
        bad_channels = []  # Default to empty list if no bad channels provided

    # Filter out bad channels from the DataFrame and channel positions
    good_channels = [ch for ch in eeg_df.columns[:n_channels] if ch not in bad_channels]
    good_ch_pos = {ch: pos for ch, pos in ch_pos.items() if ch not in bad_channels}

    # Print dropped bad channels
    print(f"Dropping bad channels: {bad_channels}")

    # Extract EEG data for the good channels only
    eeg_data = eeg_df[good_channels]
    data_array = np.stack(
        eeg_data.apply(lambda row: np.stack(row.values), axis=1).values
    ).astype(np.float32)

    # Create MNE Info object and montage
    info = mne.create_info(ch_names=good_channels, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_dig_montage(ch_pos=good_ch_pos)
    info.set_montage(montage)

    # Create MNE Epochs object
    epochs = mne.EpochsArray(data_array, info, baseline=baseline, verbose=False)

    # Drop zero channels and print them
    zero_channels = check_zero_channels(eeg_df, info['ch_names'])
    print(f"Dropping zero channels: {zero_channels}")
    epochs.drop_channels(zero_channels)

    # Drop zero epochs and print them
    zero_epochs = check_zero_epochs(data_array)
    print(f"Dropping zero epochs: {zero_epochs}")
    epochs.drop(zero_epochs, verbose=False)

    # Update the DataFrame to match remaining epochs
    # Keep only rows that correspond to non-dropped epochs
    remaining_indices = [i for i in range(len(eeg_df)) if i not in zero_epochs]
    updated_df = eeg_df.iloc[remaining_indices].reset_index(drop=True)

    return epochs, updated_df
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
    """
    Filters an MNE Epochs object to retain only specified electrodes,
    considering channels that are already removed.

    Parameters:
    epochs_Array : mne.Epochs
        The input Epochs object.

    Returns:
    mne.Epochs
        A new Epochs object containing only the necessary channels.
    """
    import copy

    # Make a copy of the input to avoid modifying it
    epochs_copy = epochs_Array.copy()

    electrodes_to_keep = []

    # Define the electrodes to keep
    electrodes_to_keep.extend([f'A{i}' for i in range(2, 33)])
    electrodes_to_keep.extend([f'B{i}' for i in range(2, 20)])
    electrodes_to_keep.extend([f'D{i}' for i in range(16, 18)])
    electrodes_to_keep.extend([f'D{i}' for i in range(24, 33)])

    # Get the existing channels in the epochs
    existing_channels = set(epochs_copy.info['ch_names'])

    # Intersect with the channels to keep
    valid_channels_to_keep = [ch for ch in electrodes_to_keep if ch in existing_channels]

    # Select only the valid channels
    epochs_filtered = epochs_copy.pick(valid_channels_to_keep)

    return epochs_filtered

