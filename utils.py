import mne

# === Utility Functions ===
def create_epochs(data_array, info, baseline=(2.0, 3.0), raw_sfreq=512):
    epochs = mne.EpochsArray(data_array, info, baseline=baseline, raw_sfreq=raw_sfreq)
    return epochs


def drop_data(epochs, channels_to_drop, epoch_indices_to_drop):
    epochs.drop_channels(channels_to_drop)
    epochs.drop(epoch_indices_to_drop)
    return epochs
