import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper
from scipy.fftpack import fft, ifft
from scipy.linalg import eigh
import mne
import matplotlib

from visualization import plot_initial_signal, plot_target_filtered_signal, plot_reference_filtered_signals, plot_final_ress_component

def remove_high_variance_channels_with_details_epochs(epochs, threshold=3):
    """
    Identify and remove channels with high variance from an mne.EpochsArray object.

    Parameters:
        epochs (mne.EpochsArray): EEG data stored in an mne.EpochsArray object.
        threshold (float): Variance threshold for excluding noisy channels.

    Returns:
        mne.EpochsArray: Epochs object with noisy channels removed.
        list: Names of retained channels.
        pd.DataFrame: DataFrame with channel variance details and removal status.
    """
    # Get data from epochs (n_trials, n_channels, n_times)
    eeg_data = epochs.get_data(copy=True)
    channel_names = epochs.info['ch_names']

    # Compute variance across trials and time for each channel
    channel_variances = np.var(eeg_data, axis=(0, 2))

    # Identify channels to keep and remove based on variance threshold
    median_variance = np.median(channel_variances)
    good_channels = np.where(channel_variances < threshold * median_variance)[0]
    bad_channels = np.where(channel_variances >= threshold * median_variance)[0]

    # Create a summary DataFrame
    channel_details = pd.DataFrame({
        'Channel Name': channel_names,
        'Variance': channel_variances,
        'Status': ['Removed' if i in bad_channels else 'Retained' for i in range(len(channel_variances))]
    })

    # Create a new Epochs object with only retained channels
    retained_channel_names = [channel_names[i] for i in good_channels]
    filtered_epochs = epochs.copy().pick(retained_channel_names)

    return filtered_epochs, retained_channel_names, channel_details

def apply_gaussian_filter_epochs(epochs, center_freq, fwhm):
    """
    Apply a narrow-band Gaussian filter to EEG data in the frequency domain.

    Parameters:
        epochs (mne.EpochsArray): EEG data stored in an mne.EpochsArray object.
        center_freq (float): Center frequency of the Gaussian filter.
        fwhm (float): Full-width at half-maximum of the Gaussian filter.

    Returns:
        np.ndarray: Filtered EEG data of shape (n_trials, n_channels, n_times).
        dict: Details about the filter (center frequency and FWHM).
    """
    # Extract data from epochs
    eeg_data = epochs.get_data(copy=True)  # Shape: (n_trials, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_trials, n_channels, n_times = eeg_data.shape

    # Compute frequency bins for FFT
    frequencies = np.fft.rfftfreq(n_times, d=1/sfreq)

    # Create Gaussian filter
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    gauss_filter = np.exp(-0.5 * ((frequencies - center_freq) / sigma) ** 2)

    # Apply filter in the frequency domain
    filtered_data = np.zeros_like(eeg_data, dtype=np.float32)
    for trial in range(n_trials):
        for channel in range(n_channels):
            # FFT of the signal
            spectrum = fft(eeg_data[trial, channel, :])
            # Apply Gaussian filter
            spectrum_filtered = spectrum[:len(gauss_filter)] * gauss_filter
            # Inverse FFT to get the filtered signal
            filtered_data[trial, channel, :] = np.real(ifft(spectrum_filtered, n=n_times))

    # Filter details for reference
    filter_details = {'center_freq': center_freq, 'fwhm': fwhm}

    return filtered_data, filter_details

def generate_reference_data(raw_data, sfreq, center_freq_low, center_freq_high, fwhm=1.0):
    """
    Generate reference data by filtering EEG at neighboring frequencies.

    Parameters:
        raw_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_times).
        sfreq (float): Sampling frequency of the EEG data.
        center_freq_low (float): Center frequency of the low reference filter.
        center_freq_high (float): Center frequency of the high reference filter.
        fwhm (float): Full-width at half-maximum of the Gaussian filter.

    Returns:
        np.ndarray: Reference data at low frequency (n_trials, n_channels, n_times).
        np.ndarray: Reference data at high frequency (n_trials, n_channels, n_times).
    """
    # Create Gaussian filters
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    n_times = raw_data.shape[2]
    frequencies = np.fft.rfftfreq(n_times, d=1 / sfreq)

    gauss_filter_low = np.exp(-0.5 * ((frequencies - center_freq_low) / sigma) ** 2)
    gauss_filter_high = np.exp(-0.5 * ((frequencies - center_freq_high) / sigma) ** 2)

    # Apply filters in the frequency domain
    ref_data_low = np.zeros_like(raw_data)
    ref_data_high = np.zeros_like(raw_data)

    for trial in range(raw_data.shape[0]):
        for channel in range(raw_data.shape[1]):
            spectrum = np.fft.fft(raw_data[trial, channel, :])
            spectrum_low = spectrum[:len(gauss_filter_low)] * gauss_filter_low
            spectrum_high = spectrum[:len(gauss_filter_high)] * gauss_filter_high
            ref_data_low[trial, channel, :] = np.real(np.fft.ifft(spectrum_low, n=n_times))
            ref_data_high[trial, channel, :] = np.real(np.fft.ifft(spectrum_high, n=n_times))

    return ref_data_low, ref_data_high

def compute_covariance_matrices(signal_data, reference_data_low, reference_data_high, sfreq, tmin, tmax, shrinkage=0.01):
    """
    Compute signal (S) and reference (R) covariance matrices with shrinkage regularization.

    Parameters:
        signal_data (np.ndarray): Signal data of shape (n_trials, n_channels, n_times).
        reference_data_low (np.ndarray): Low-frequency reference data (n_trials, n_channels, n_times).
        reference_data_high (np.ndarray): High-frequency reference data (n_trials, n_channels, n_times).
        sfreq (float): Sampling frequency of the EEG data.
        tmin (float): Start time in seconds for covariance calculation.
        tmax (float): End time in seconds for covariance calculation.
        shrinkage (float): Shrinkage regularization parameter.

    Returns:
        np.ndarray: Signal covariance matrix (S).
        np.ndarray: Regularized reference covariance matrix (R).
    """
    # Convert tmin and tmax to sample indices
    n_times = signal_data.shape[2]
    time = np.arange(n_times) / sfreq  # Time vector in seconds
    idx_start = np.searchsorted(time, tmin)
    idx_end = np.searchsorted(time, tmax)

    # Extract the time window
    signal_data_window = signal_data[:, :, idx_start:idx_end]
    reference_data_low_window = reference_data_low[:, :, idx_start:idx_end]
    reference_data_high_window = reference_data_high[:, :, idx_start:idx_end]

    # Concatenate trials across time for covariance calculation
    signal_concat = signal_data_window.transpose(1, 0, 2).reshape(signal_data_window.shape[1], -1)
    reference_low_concat = reference_data_low_window.transpose(1, 0, 2).reshape(reference_data_low_window.shape[1], -1)
    reference_high_concat = reference_data_high_window.transpose(1, 0, 2).reshape(reference_data_high_window.shape[1], -1)

    # Compute covariance matrices
    cov_signal = np.cov(signal_concat)
    cov_reference_low = np.cov(reference_low_concat)
    cov_reference_high = np.cov(reference_high_concat)

    # Average the reference covariance matrices
    cov_reference = (cov_reference_low + cov_reference_high) / 2

    # Apply shrinkage regularization to the reference covariance matrix
    diag_mean = np.mean(np.diag(cov_reference))
    cov_reference_reg = cov_reference + shrinkage * diag_mean * np.eye(cov_reference.shape[0])

    return cov_signal, cov_reference_reg


def regularize_covariance(cov_reference, shrinkage=0.01):
    """
    Regularize the reference covariance matrix by adding a fraction of its diagonal mean.

    Parameters:
        cov_reference (np.ndarray): Reference covariance matrix.
        shrinkage (float): Shrinkage regularization parameter.

    Returns:
        np.ndarray: Regularized covariance matrix.
    """
    diag_mean = np.mean(np.diag(cov_reference))  # Average of diagonal elements
    cov_reference_reg = cov_reference + shrinkage * diag_mean * np.eye(cov_reference.shape[0])
    return cov_reference_reg


def perform_ged(cov_signal, cov_reference):
    """
    Perform Generalized Eigenvalue Decomposition (GED).

    Parameters:
        cov_signal (np.ndarray): Signal covariance matrix (S) of shape (n_channels, n_channels).
        cov_reference (np.ndarray): Regularized reference covariance matrix (R) of shape (n_channels, n_channels).

    Returns:
        np.ndarray: Eigenvalues (sorted in descending order).
        np.ndarray: Eigenvectors (spatial filters), sorted by corresponding eigenvalue.
    """
    # Regularize the reference covariance matrix
    cov_reference_reg = regularize_covariance(cov_reference, shrinkage=0.01)

    # Perform GED
    eigvals, eigvecs = eigh(cov_signal, cov_reference_reg)


    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    return eigvals, eigvecs

def apply_spatial_filter(eeg_data, spatial_filter):
    """
    Apply a spatial filter to EEG data.

    Parameters:
        eeg_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_times).
        spatial_filter (np.ndarray): Spatial filter of shape (n_channels,).

    Returns:
        np.ndarray: RESS component time series of shape (n_trials, n_times).
    """
    return np.einsum('ijk,j->ik', eeg_data, spatial_filter)


def ress_pipeline(epochs, freq_df, target_freq_map, fwhm_signal=0.6, fwhm_reference=1.0,
                  tmin=0.0, tmax=3.0, shrinkage=0.01, output_dir='', title_prefix="EEG"):

    print("")

    epochs_data = epochs.get_data(copy=True)
    os.makedirs(output_dir, exist_ok=True)
    sfreq = epochs.info['sfreq']
    ch_names = epochs.info['ch_names']


    n_epochs = epochs_data.shape[0]
    n_times = epochs_data.shape[2]
    combined_ress_components = np.zeros((n_epochs, n_times))
    detailed_results = {}

    freq_df = freq_df.reset_index()
    freq_df['Frequency'] = freq_df['Frequency'].apply(
        lambda x: int(x[0]) if isinstance(x, (list, np.ndarray)) else int(x)
    )

    unique_freq_ids = freq_df['Frequency'].unique()
    non_zero_channels = [
        idx for idx, ch in enumerate(epochs.info['ch_names'])
        if not np.all(epochs_data[:, idx, :] == 0)
    ]
    # Plot original PSD (before preprocessing)
    fig, ax = plt.subplots()
    epochs.compute_psd(method='welch', fmin=0.0, fmax=60.0, tmin=tmin, tmax=tmax, picks=non_zero_channels).plot(show=False)
    plt.title("")
    plt.savefig(os.path.join('./Image/PSD_plots', f"{title_prefix}_original_psd.png"))
    plt.close(fig)

    for freq_id in unique_freq_ids:
        print("")
        if freq_id not in target_freq_map:
            print(f"Warning: Frequency ID {freq_id} not found in target_freq_map. Skipping.")
            continue

        target_freq = target_freq_map[freq_id]
        print(f"Processing Frequency ID: {freq_id} -> Target Frequency: {target_freq} Hz")

        freq_trials = freq_df[freq_df['Frequency'] == freq_id].index
        if len(freq_trials) == 0:
            print(f"No trials found for Frequency ID {freq_id}. Skipping.")
            continue

        freq_epochs = epochs[freq_trials]
        filtered_epochs, retained_channels, channel_details = remove_high_variance_channels_with_details_epochs(freq_epochs)
        print("High Variance channels: ", channel_details[channel_details.Status != 'Retained'])


        filtered_signal, _ = apply_gaussian_filter_epochs(filtered_epochs, target_freq, fwhm_signal)
        ref_data_low, ref_data_high = generate_reference_data(
            filtered_epochs.get_data(copy=True), sfreq, target_freq - 1, target_freq + 1, fwhm_reference
        )

        cov_signal, cov_reference_reg = compute_covariance_matrices(
            filtered_signal, ref_data_low, ref_data_high, sfreq, tmin, tmax, shrinkage
        )

        eigvals, eigvecs = perform_ged(cov_signal, cov_reference_reg)
        spatial_filter = eigvecs[:, 0]
        ress_components = apply_spatial_filter(filtered_signal, spatial_filter)

        combined_ress_components[freq_trials, :] = ress_components

        detailed_results[f"Freq_{target_freq}Hz"] = {
            'cov_signal': cov_signal,
            'cov_reference_reg': cov_reference_reg,
            'eigvals': eigvals,
            'eigvecs': eigvecs,
            'ress_components': ress_components
        }

        time_vector = np.linspace(0, n_times / sfreq, n_times)
        plot_initial_signal(epochs_data[freq_trials], time_vector, output_dir,f'{title_prefix}_{target_freq}')
        plot_target_filtered_signal(filtered_signal, time_vector, target_freq, output_dir,f'{title_prefix}_{target_freq}')
        plot_reference_filtered_signals(ref_data_low, ref_data_high, time_vector, target_freq, output_dir,f'{title_prefix}_{target_freq}')
        plot_final_ress_component(ress_components, time_vector, output_dir,f'{title_prefix}_{target_freq}')


# Save covariance matrices
        plt.matshow(cov_signal, cmap='viridis')
        plt.title(f"Covariance Matrix (Signal) - {target_freq} Hz")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title_prefix}_cov_signal_{target_freq}Hz.png"))
        plt.close()

        plt.matshow(cov_reference_reg, cmap='viridis')
        plt.title(f"Covariance Matrix (Reference) - {target_freq} Hz")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title_prefix}_cov_reference_{target_freq}Hz.png"))
        plt.close()

        print(f"Finished processing Frequency ID: {freq_id} -> {target_freq} Hz")


    # Plot PSD of combined RESS components using MNE
    fig, ax = plt.subplots()
    psd, freqs = mne.time_frequency.psd_array_welch(
        combined_ress_components,
        sfreq=sfreq,
        fmin=1,
        fmax=50,
        n_fft=1024,

        verbose=False,
        output='power'
    )

    psd_mean = (psd.mean(axis=0))
    psd_std =(psd.std(axis=0))
    psd_min = (psd.min(axis=0))
    psd_max = (psd.max(axis=0))
# Plot mean with variability
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, psd_mean, label="Mean PSD", color='red', linewidth=2)
    plt.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='blue', alpha=0.3, label="Standard deviation")
    plt.fill_between(freqs, psd_min, psd_max, color='green', alpha=0.2, label="Min/Max Range")

    # Enhance plot
    plt.title("PSD of combined RESS components", fontsize=16, pad=20)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Power (dB)", fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="upper right")
    plt.savefig(os.path.join('./Image/PSD_plots', f"{title_prefix}_ress_psd.png"))
    plt.close(fig)

    print("RESS pipeline completed for all target frequencies.")

    return combined_ress_components


