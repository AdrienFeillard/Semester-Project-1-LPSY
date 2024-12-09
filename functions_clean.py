import glob
import mne
import pandas as pd
import scipy.io
import numpy as np
import ast
import re
import scipy.io
from scipy.fftpack import fft
from scipy.signal import welch
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split


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

def preprocess_eeg_data(eeg_df, n_epochs, n_channels):
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


def create_epochs(data_array, info, baseline=(2.0, 3.0), raw_sfreq=512):
    epochs = mne.EpochsArray(data_array, info, baseline=baseline, raw_sfreq=raw_sfreq)
    return epochs


def drop_data(epochs, channels_to_drop, epoch_indices_to_drop):
    epochs.drop_channels(channels_to_drop)
    epochs.drop(epoch_indices_to_drop)
    return epochs


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

def compute_psd(epochs, fmin=0.0, fmax=60.0, tmin=0.0, tmax=2.0):
    psd = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    return psd


def save_psd_plot(psd, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot(show=False)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
    print(f"PSD plot saved as '{os.path.join(output_dir, f'{filename}.png')}'")


def plot_topomap(psd, bands,output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot_topomap(bands=bands, vlim="joint", show=False, dB=True)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
    print(f"Topomap plot saved as '{os.path.join(output_dir, f'{filename}.png')}'")

def compute_ed_matrix(data_array):
    """
    Computes the electrical distance matrix (ed_matrix) for a given EEG data array.

    Parameters:
    ----------
    data_array : np.ndarray
        EEG data array of shape (n_epochs, n_channels, n_times).

    Returns:
    -------
    ed_matrix : np.ndarray
        Electrical distance matrix of shape (n_epochs, n_channels, n_channels).
    """
    # Extract dimensions
    n_epochs, n_channels, n_times = data_array.shape

    # Initialize the electrical distance matrix
    ed_matrix = np.zeros((n_epochs, n_channels, n_channels))

    # Compute pairwise electrical distances for each epoch
    for epoch_idx in range(n_epochs):
        data_epoch = data_array[epoch_idx, :, :]  # Shape: (n_channels, n_times)

        # Compute pairwise differences
        diff_matrix = data_epoch[:, None, :] - data_epoch[None, :, :]  # Shape: (n_channels, n_channels, n_times)
        ed_matrix[epoch_idx] = np.var(diff_matrix, axis=2)  # Variance along the time axis

    return ed_matrix


def plot_ed_topomap(ed_matrix, ch_pos, info, vlim=(None, 5)):
    """
    Plots a topomap displaying electrical distances.

    Parameters:
    ----------
    ed_matrix : np.ndarray
        Electrical distance matrix of shape (n_epochs, n_channels, n_channels).
    ch_pos : dict
        Dictionary containing channel positions {channel_name: (x, y, z)}.
    info : mne.Info
        MNE Info object with metadata about the EEG data.
    vlim : tuple, optional
        Value limits for the colormap. Default is (None, 5).

    Returns:
    -------
    None
    """
    # Create a Raw object to simulate the EEG structure
    n_channels = ed_matrix.shape[1]
    n_times = 1  # Dummy value for time points
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
    raw = mne.io.RawArray(np.zeros((n_channels, n_times)), info)
    raw.set_montage(montage)

    # Plot the topomap for electrical distances
    mne.viz.plot_bridged_electrodes(
        raw.info,
        bridged_idx=[],  # No predefined bridged indices
        ed_matrix=ed_matrix,  # Full 3D ed_matrix
        title="Electrical Distance Topomap",
        topomap_args=dict(vlim=vlim, average=True),  # Use "average=True" to plot median electrical distances
    )
    plt.show()


def plot_ed_matrix(ed_matrix, title, vmin=None, vmax=None, show=False, output_dir='ed_matrix', filename=''):
    """
    Plots the median electrical distance matrix across all epochs as a heatmap.

    Parameters:
    ----------
    ed_matrix : np.ndarray
        Electrical distance matrix of shape (n_epochs, n_channels, n_channels).
    title : str
        Title for the heatmap.
    vmin : float, optional
        Minimum value for the colormap. Default is None.
    vmax : float, optional
        Maximum value for the colormap. Default is None.
    show : bool, optional
        Whether to display the plot. Default is False.
    output_dir : str, optional
        Directory to save the heatmap. Default is 'ed_matrix'.
    filename : str, optional
        Name of the file to save the heatmap. Default is ''.

    Returns:
    -------
    None
    """
    # Compute the median electrical distance matrix across epochs
    ed_matrix_median = np.median(ed_matrix, axis=0)  # Shape: (n_channels, n_channels)

    # Plot the median electrical distance matrix
    plt.figure(figsize=(10, 8))
    plt.title(title)
    im = plt.imshow(ed_matrix_median, cmap="viridis", aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Electrical Distance (Median Variance)")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(save_path, bbox_inches='tight')  # Save the entire figure
    print(f"Figure saved at {save_path}")

    if show:
        plt.show()

    plt.close()

def save_ica_component_properties(ica, epochs, filename, output_dir="ica_properties_figures", freq_range=(0, 60)):
    """
    Save ICA component properties to individual figures.

    Parameters:
    - ica: ICA object
        The ICA object containing the components.
    - epochs: Epochs object
        The epochs data to use for plotting the ICA properties.
    - output_dir: str, optional
        Directory to save the figures. Default is "ica_properties_figures".
    - freq_range: tuple, optional
        Frequency range for the spectrum plot (default is (0, 60)).
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all ICA components
    for component_idx in range(ica.n_components_):
        # Plot properties for the current component
        fig = ica.plot_properties(epochs, picks=component_idx, show=False)

        # Modify the x-axis range of the spectrum plot
        for ax in fig[0].axes:  # Loop through all axes in the figure
            if ax.get_xlabel() == "Frequency (Hz)":  # Find the spectrum plot
                ax.set_xlim(freq_range)  # Set the desired x-axis range

        # Save the modified figure to the output directory
        fig_path = os.path.join(output_dir, f"{filename}component_{component_idx}_properties.png")
        fig[0].savefig(fig_path)
        plt.close(fig[0])  # Close the figure to free memory

    print(f"All ICA component properties saved to: {output_dir}")

def create_topomap_animation(evoked, output_filename="topomap_animation.mp4", fps=30, ch_type='eeg', cmap='Reds'):
    """
    Creates an animated topomap video from evoked data.

    Parameters:
        evoked (mne.Evoked): The evoked object containing the averaged epochs.
        output_filename (str): Name of the output video file (default is "topomap_animation.mp4").
        fps (int): Frames per second for the animation (default is 30).
        ch_type (str): The channel type to plot (default is 'eeg').
        cmap (str): Colormap for the topomap (default is 'Reds').
    """
    # Define parameters for the animation
    time_points = evoked.times  # All time points in the evoked data
    time_step = int(evoked.info['sfreq'] / fps)  # Step size for animation frames
    time_indices = np.arange(0, len(evoked.times), time_step)  # Indices for frames

    # Initialize the figure
    fig, ax = plt.subplots()
    vmin, vmax = evoked.data.min(), evoked.data.max()  # Dynamic scaling based on data range
    topomap_kwargs = dict(vlim=(vmin, vmax), contours=0, sensors=True)

    # Function to update the topomap at each frame
    def update_topomap(frame_idx):
        ax.clear()  # Clear the axes
        evoked.plot_topomap(times=[evoked.times[frame_idx]], axes=ax, show=False,
                            colorbar=False, ch_type=ch_type, extrapolate='head', cmap=cmap)
        ax.set_title(f"Time: {evoked.times[frame_idx]:.3f} s")

    # Create the animation
    ani = FuncAnimation(fig, update_topomap, frames=time_indices, repeat=False)

    # Save the animation as a video file
    ani.save(output_filename, writer="ffmpeg", fps=fps)
    print(f"Topomap video saved as '{output_filename}'")


def create_topomap_animation(evoked, output_filename="topomap_animation.mp4", fps=30, ch_type='eeg', cmap='Reds'):
    """
    Creates an animated topomap video from evoked data.

    Parameters:
        evoked (mne.Evoked): The evoked object containing the averaged epochs.
        output_filename (str): Name of the output video file (default is "topomap_animation.mp4").
        fps (int): Frames per second for the animation (default is 30).
        ch_type (str): The channel type to plot (default is 'eeg').
        cmap (str): Colormap for the topomap (default is 'Reds').
    """
    # Define parameters for the animation
    time_points = evoked.times  # All time points in the evoked data
    time_step = int(evoked.info['sfreq'] / fps)  # Step size for animation frames
    time_indices = np.arange(0, len(evoked.times), time_step)  # Indices for frames

    # Initialize the figure
    fig, ax = plt.subplots()
    vmin, vmax = evoked.data.min(), evoked.data.max()  # Dynamic scaling based on data range
    topomap_kwargs = dict(vlim=(vmin, vmax), contours=0, sensors=True)

    # Function to update the topomap at each frame
    def update_topomap(frame_idx):
        ax.clear()  # Clear the axes
        evoked.plot_topomap(times=[evoked.times[frame_idx]], axes=ax, show=False,
                            colorbar=False, ch_type=ch_type, extrapolate='head', cmap=cmap)
        ax.set_title(f"Time: {evoked.times[frame_idx]:.3f} s")

    # Create the animation
    ani = FuncAnimation(fig, update_topomap, frames=time_indices, repeat=False)

    # Save the animation as a video file
    ani.save(output_filename, writer="ffmpeg", fps=fps)
    print(f"Topomap video saved as '{output_filename}'")


def sigmoid(x, a, b, c, d):
    """
    Sigmoid function to fit the psychometric curve.
    Parameters:
    -----------
    x : np.ndarray
        Input (e.g., contrast values).
    a : float
        Minimum value of the sigmoid.
    b : float
        Maximum value of the sigmoid.
    c : float
        Slope of the sigmoid.
    d : float
        Contrast at the inflection point (midpoint).
    Returns:
    --------
    np.ndarray
        Sigmoid function values for input x.
    """
    return a + (b - a) / (1.0 + np.exp(-c * (x - d)))

def compute_accuracy_vs_contrast_with_sigmoid(
        epochs_train, adr_train_data, epochs_test, adr_test_data,
        plot=True, min_samples=5
):
    """
    Train an LDA model on frequency labels, compute accuracy as a function of contrast,
    and fit a sigmoid psychometric function.

    Parameters:
    -----------
    epochs_train : mne.Epochs
        Training EEG data.
    adr_train_data : pandas.DataFrame
        DataFrame containing training labels (Frequency and optional Contrast).
    epochs_test : mne.Epochs
        Test EEG data.
    adr_test_data : pandas.DataFrame
        DataFrame containing test labels (Frequency and Contrast).
    plot : bool, optional
        Whether to plot accuracy vs. contrast (default: True).
    min_samples : int, optional
        Minimum number of samples per contrast to compute accuracy (default: 5).

    Returns:
    --------
    sorted_contrasts : np.ndarray
        Array of unique contrast values (sorted).
    sorted_accuracies : np.ndarray
        Array of accuracies corresponding to the sorted contrasts.
    sigmoid_params : tuple
        Parameters of the fitted sigmoid function (a, b, c, d).
    """
    # Step 1: Prepare the training data
    # Extract raw EEG data and flatten
    X_train = epochs_train.get_data()  # Training EEG data
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten to (n_epochs_train, n_features)

    # Convert training frequency labels to a flat 1D array
    y_train = np.array([float(label) for label in adr_train_data.Frequency])

    # Step 2: Train the LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Step 3: Prepare the test data
    X_test = epochs_test.get_data()  # Test EEG data
    X_test = X_test.reshape(X_test.shape[0], -1)  # Flatten to (n_epochs_test, n_features)

    # Convert test frequency labels and contrast values to flat 1D arrays
    y_test = np.array([float(label) for label in adr_test_data.Frequency])
    contrast_test = np.array([float(label) for label in adr_test_data.Contrast])

    # Step 4: Predict the frequency values for the test data
    y_pred = lda.predict(X_test)

    # Step 5: Compute accuracy for each contrast
    accuracy_dict = {}
    unique_contrasts = np.unique(contrast_test)  # Get unique contrast values

    for contrast in unique_contrasts:
        # Find indices of test samples with the current contrast
        idx = np.where(contrast_test == contrast)[0]

        # Skip contrasts with fewer than min_samples
        if len(idx) < min_samples:
            print(f"Skipping contrast {contrast} due to insufficient samples ({len(idx)} samples).")
            continue

        # Compute accuracy for this contrast
        accuracy = accuracy_score(y_test[idx], y_pred[idx])  # Compare true vs predicted
        accuracy_dict[contrast] = accuracy

    # Step 6: Sort contrasts and accuracies
    sorted_contrasts = np.array(sorted(accuracy_dict.keys()))  # Sorted contrast values
    sorted_accuracies = np.array([accuracy_dict[contrast] for contrast in sorted_contrasts])

    # Step 7: Fit a sigmoid function to the accuracy vs contrast data
    initial_guess = [0.5, 1.0, 1.0, np.median(sorted_contrasts)]  # Initial params: [a, b, c, d]
    sigmoid_params, _ = curve_fit(sigmoid, sorted_contrasts, sorted_accuracies, p0=initial_guess)


    plt.figure(figsize=(8, 6))

    # Plot the raw accuracy data
    plt.scatter(sorted_contrasts, sorted_accuracies, color="b", label="Accuracy (data)", zorder=2)

    # Plot the sigmoid fit
    fine_contrasts = np.linspace(sorted_contrasts.min(), sorted_contrasts.max(), 500)
    sigmoid_fit = sigmoid(fine_contrasts, *sigmoid_params)
    plt.plot(fine_contrasts, sigmoid_fit, color="r", linestyle="--", label="Sigmoid fit", zorder=1)

    # Add labels and title
    plt.xlabel("Contrast", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy vs Contrast (with Sigmoid Fit)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    # Step 8: Optionally plot accuracy vs contrast with sigmoid fit

    if plot:

        plt.show()

    return sorted_contrasts, sorted_accuracies, sigmoid_params

def classify_baseline_vs_stimulus_train_test(epochs_train, epochs_test,train_data, test_data,
                                             baseline_range=(0.0, 0.2),
                                             stimulus_range=(0.2, 0.4)):
    """
    Perform classification for each unique `Frequency` label and plot accuracy vs contrast.
    Classify baseline vs. stimulus for each `Frequency` separately.

    Parameters:
    -----------
    epochs_train : mne.Epochs
        Training EEG epochs containing both stimulus and baseline periods.
    epochs_test : mne.Epochs
        Testing EEG epochs containing both stimulus and baseline periods.
    train_data : pandas.DataFrame
        DataFrame containing `Frequency` and `Contrast` labels for the training data.
    test_data : pandas.DataFrame
        DataFrame containing `Frequency` and `Contrast` labels for the testing data.
    baseline_range : tuple, optional
        Time range for the baseline period (default: 0.0 to 2.0 seconds).
    stimulus_range : tuple, optional
        Time range for the stimulus period (default: 2.0 to 3.0 seconds).

    Returns:
    --------
    None
        Prints classification metrics and plots accuracy vs contrast for each `Frequency`.
    """
    # Ensure that train_data and test_data are aligned with epochs
    if len(train_data) != len(epochs_train) or len(test_data) != len(epochs_test):
        raise ValueError(
            f"Mismatch between data and epochs: "
            f"train_data ({len(train_data)}) vs epochs_train ({len(epochs_train)}), "
            f"test_data ({len(test_data)}) vs epochs_test ({len(epochs_test)})"
        )

    # Extract unique frequency labels
    frequencies = np.unique(train_data.Frequency)

    plt.figure(figsize=(10, 6))

    for freq in frequencies:
        print(f"Performing classification for Frequency: {freq}")

        # Step 1: Filter data for the current frequency
        train_idx = train_data.Frequency.values == freq  # Boolean mask for training data
        test_idx = test_data.Frequency.values == freq   # Boolean mask for testing data

        # Ensure the boolean masks are applied correctly to the epochs
        epochs_train_freq = epochs_train[train_idx]
        epochs_test_freq = epochs_test[test_idx]

        # Extract contrast values for the current frequency
        contrast_test = test_data[test_idx].Contrast.values

        # Step 2: Extract data and times
        X_train = epochs_train_freq.get_data()  # Shape: (n_epochs, n_channels, n_times)
        X_test = epochs_test_freq.get_data()    # Shape: (n_epochs, n_channels, n_times)

        times_train = epochs_train_freq.times  # Time points for training epochs
        times_test = epochs_test_freq.times    # Time points for testing epochs

        # Step 3: Label each time point
        y_train = np.zeros(X_train.shape[2], dtype=int)  # Initialize all labels to 0 (baseline)
        y_train[(times_train >= baseline_range[0]) & (times_train < baseline_range[1])] = 0  # Baseline
        y_train[(times_train >= stimulus_range[0]) & (times_train < stimulus_range[1])] = 1  # Stimulus

        y_test = np.zeros(X_test.shape[2], dtype=int)  # Initialize all labels to 0 (baseline)
        y_test[(times_test >= baseline_range[0]) & (times_test < baseline_range[1])] = 0  # Baseline
        y_test[(times_test >= stimulus_range[0]) & (times_test < stimulus_range[1])] = 1  # Stimulus

        # Step 4: Reshape data for LDA
        X_train_flat = X_train.reshape(-1, X_train.shape[1])  # Shape: (n_epochs * n_times, n_channels)
        X_test_flat = X_test.reshape(-1, X_test.shape[1])     # Shape: (n_epochs * n_times, n_channels)

        # Repeat labels for all epochs
        y_train_flat = np.tile(y_train, X_train.shape[0])  # Repeat labels for all epochs
        y_test_flat = np.tile(y_test, X_test.shape[0])     # Repeat labels for all epochs

        # Expand contrast values to match the flattened data
        contrast_test_flat = np.repeat(contrast_test, X_test.shape[2])  # Repeat for each time point

        # Step 5: Train LDA
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_flat, y_train_flat)

        # Step 6: Predict on the test set
        y_pred = clf.predict(X_test_flat)

        # Step 7: Compute accuracy as a function of contrast
        unique_contrasts = np.unique(contrast_test)  # Unique contrast values
        accuracy_dict = {}

        for contrast in unique_contrasts:
            # Ensure contrast is a scalar
            contrast = contrast.item() if isinstance(contrast, np.ndarray) else contrast

            # Find indices of test samples with the current contrast
            idx = np.where(contrast_test_flat == contrast)[0]

            # Compute accuracy for this contrast
            accuracy = accuracy_score(y_test_flat[idx], y_pred[idx])
            accuracy_dict[contrast] = accuracy

        # Sort contrasts and accuracies
        sorted_contrasts = np.array(sorted(accuracy_dict.keys()))  # Sorted contrast values
        sorted_accuracies = np.array([accuracy_dict[contrast] for contrast in sorted_contrasts])

        # Plot accuracy vs contrast for this frequency
        plt.plot(sorted_contrasts, sorted_accuracies, label=f"Frequency {freq}")

        # Print classification report for this frequency
        print(f"Classification Report for Frequency {freq}:")
        print(classification_report(y_test_flat, y_pred))

    # Finalize the plot
    plt.xlabel("Contrast", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy vs Contrast for Each Frequency", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    #return classification_report(y_test_flat, y_pred)