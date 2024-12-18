import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
from sklearn.svm import SVC

# === Computation Functions ===
def compute_psd(epochs, fmin=0.0, fmax=60.0, tmin=0.0, tmax=2.0):
    psd = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    return psd


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
        epochs_train, adr_train_data, epochs_test, adr_test_data, model_name,
        plot=True, min_samples=5, output_dir ='', filename='', max_fev = 10000
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

    os.makedirs(output_dir, exist_ok=True)    # Step 1: Prepare the training data
    # Extract raw EEG data and flatten
    X_train = epochs_train#.get_data(copy=True)  # Training EEG data
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten to (n_epochs_train, n_features)

    # Convert training frequency labels to a flat 1D array
    y_train = np.array([float(label) for label in adr_train_data.Frequency])

    # Step 2: Train the LDA model
    if model_name =='LDA':
        model = LinearDiscriminantAnalysis()
    elif model_name =='SVM':
        model = SVC(verbose=False)


    model.fit(X_train, y_train)

    # Step 3: Prepare the test data
    X_test = epochs_test#.get_data(copy=True)  # Test EEG data
    X_test = X_test.reshape(X_test.shape[0], -1)  # Flatten to (n_epochs_test, n_features)

    # Convert test frequency labels and contrast values to flat 1D arrays
    y_test = np.array([float(label) for label in adr_test_data.Frequency])
    contrast_test = np.array([float(label) for label in adr_test_data.Contrast])

    # Step 4: Predict the frequency values for the test data
    y_pred = model.predict(X_test)

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
    sigmoid_params, _ = curve_fit(sigmoid, sorted_contrasts, sorted_accuracies, p0=initial_guess, maxfev=max_fev)


    plt.figure(figsize=(8, 6))

    # Plot the raw accuracy data
    plt.scatter(sorted_contrasts, sorted_accuracies, color="b", label="Accuracy (data)", zorder=2)

    # Plot the sigmoid fit
    x_fit = np.linspace(sorted_contrasts.min(), sorted_contrasts.max(), 500)
    sigmoid_fit = sigmoid(x_fit, *sigmoid_params)
    plt.plot(x_fit, sigmoid_fit, color="r", linestyle="--", label="Sigmoid fit", zorder=1)

    # Add labels and title
    plt.xlabel("Contrast", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy vs Contrast (with Sigmoid Fit)", fontsize=16)
    plt.legend(fontsize=12)

    plt.axhline(y=0.75, color='r', linestyle='--')

    y_fit = sigmoid(x_fit, *sigmoid_params)
    x_75_index = np.searchsorted(y_fit.ravel(), 0.75)
    if x_75_index < len(x_fit):
        x_75 = x_fit[x_75_index]
        plt.axvline(x=x_75, color='r', linestyle='--', label=f'75% Accuracy: {x_75:.2f}')
        plt.annotate(f'{x_75:.2f}', (x_75, 0.75), textcoords="offset points", xytext=(0, 10), ha='center')
    else:
        print(f"Warning: 75% accuracy not reached for fit.")

    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(save_path)

    if plot:

        plt.show()
    plt.close()
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

        """# Print classification report for this frequency
        print(f"Classification Report for Frequency {freq}:")
        print(classification_report(y_test_flat, y_pred))"""

    # Finalize the plot
    plt.xlabel("Contrast", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy vs Contrast for Each Frequency", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    #return classification_report(y_test_flat, y_pred)


def augment_with_shuffled_baseline(eeg_data, contrast_df, ress =False):
    """
    Create a combined array of original EEG data and shuffled artificial baselines with metadata.

    Parameters:
        eeg_data (np.ndarray): EEG data array of shape (n_trials, n_channels, n_times).
        contrast_df (pd.DataFrame): DataFrame containing contrast values for each trial.

    Returns:
        np.ndarray: Augmented data array of shape (2 * n_trials, n_channels, n_times).
        pd.DataFrame: DataFrame with contrast values and marker ('Original' or 'Shuffled').
    """
    # Initialize an array for the shuffled baselines
    shuffled_baselines = np.zeros_like(eeg_data)

    if ress:
        n_trials, n_times = eeg_data.shape
        for trial in range(n_trials):
            shuffled_baselines[trial, :] = np.random.permutation(eeg_data[trial, :])
    else:
        n_trials, n_channels, n_times = eeg_data.shape
        for trial in range(n_trials):
            for channel in range(n_channels):
                shuffled_baselines[trial, channel, :] = np.random.permutation(eeg_data[trial, channel, :])

    # Combine original data and shuffled baselines
    augmented_data = np.vstack((eeg_data, shuffled_baselines))

    # Create metadata for the combined data
    original_metadata = pd.DataFrame({
        'Contrast': contrast_df['Contrast'],
        'Frequency': 1
    })

    shuffled_metadata = pd.DataFrame({
        'Contrast': contrast_df['Contrast'],
        'Frequency': 2
    })

    # Concatenate metadata
    augmented_metadata = pd.concat([original_metadata, shuffled_metadata], ignore_index=True)

    return augmented_data, augmented_metadata