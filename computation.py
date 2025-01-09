import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.svm import SVC
from visualization import plot_accuracy_vs_contrast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D, Dropout
import mne

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

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
def build_cnn(input_shape, num_classes):
    """
    Build a simple CNN using TensorFlow/Keras.

    Parameters:
    -----------
    input_shape : tuple
        Shape of a single input sample (e.g. (n_channels, n_times, 1)).
    num_classes : int
        Number of distinct output classes.

    Returns:
    --------
    model : tf.keras.Model
        Compiled CNN model.
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def synchronize_channels_directly(epochs_train, epochs_test):
    """
    Synchronize channels directly between train and test datasets by keeping only common channels.

    Parameters:
    - epochs_train: mne.Epochs object for training data.
    - epochs_test: mne.Epochs object for testing data.

    Returns:
    - epochs_train: Modified mne.Epochs object for training with common channels.
    - epochs_test: Modified mne.Epochs object for testing with common channels.
    """
    # Get the list of channel names for train and test
    train_channels = epochs_train.info['ch_names']
    test_channels = epochs_test.info['ch_names']

    # Find the common channels
    common_channels = [ch for ch in train_channels if ch in test_channels]

    epochs_train_copy = epochs_train.copy().pick(common_channels)
    epochs_test_copy = epochs_test.copy().pick(common_channels)

    return epochs_train_copy, epochs_test_copy



def compute_accuracy_vs_contrast_with_sigmoid(
        epochs_train, train_data,
        epochs_test, test_data,
        model_name,
        predict_baseline=False,  # New parameter to switch target class
        plot=True,
        ress_filtered_data = False,
        min_samples=5,
        output_dir='',
        filename='',
        max_fev=10000,
        epochs_cnn=5,  # number of epochs for CNN training
        batch_size_cnn=32,  # batch size for CNN training
        verbose_cnn = 1,
        title = ''
):
    """
    Train a model on frequency labels or baseline/stimulus classification,
    compute accuracy as a function of contrast, and optionally fit a sigmoid psychometric function.
    """


    # Step 2: Prepare the training data
    if not ress_filtered_data:
        X_train_raw = epochs_train.get_data(copy=True)
        X_test_raw = epochs_test.get_data(copy=True)
    else:
        X_train_raw = epochs_train
        X_test_raw = epochs_test

    # Decide the target labels based on `predict_baseline`
    if predict_baseline:
        y_train_raw = np.array(train_data['Stimulus or Baseline'])
        y_test_raw = np.array(test_data['Stimulus or Baseline'])
    else:
        y_train_raw = np.array([float(label) for label in train_data.Frequency])
        y_test_raw = np.array([float(label) for label in test_data.Frequency])

    # -----------------------------------------------------
    # Model selection and training
    # -----------------------------------------------------
    if model_name == 'LDA':
        # Flatten data: (n_epochs, n_features)
        X_train = X_train_raw.reshape(len(X_train_raw), -1)
        X_test = X_test_raw.reshape(len(X_test_raw), -1)

        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train_raw)

        # Predictions
        y_pred = model.predict(X_test)

    elif model_name == 'SVM':
        # Flatten data: (n_epochs, n_features)
        X_train = X_train_raw.reshape(len(X_train_raw), -1)
        X_test = X_test_raw.reshape(len(X_test_raw), -1)

        model = SVC(verbose=False)
        model.fit(X_train, y_train_raw)

        # Predictions
        y_pred = model.predict(X_test)

    elif model_name == 'CNN':
        num_classes = 2

        X_train = X_train_raw#[..., np.newaxis]
        X_test = X_test_raw#[..., np.newaxis]
        #print(y_train_raw)
        if predict_baseline:
            y_train = y_train_raw - 1  # Convert 1/2 to 0/1 for CNN compatibility
            y_test = y_test_raw - 1  # Convert 1/2 to 0/1 for CNN compatibility
        else:
            unique_freqs = np.unique(y_train_raw)
            freq2idx = {freq: i for i, freq in enumerate(unique_freqs)}
            idx2freq = {i: freq for freq, i in freq2idx.items()}

            y_train = np.array([freq2idx[freq] for freq in y_train_raw])
            y_test = np.array([freq2idx[freq] for freq in y_test_raw])

        # Build and train CNN
        model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
        model.fit(X_train, y_train, epochs=epochs_cnn, batch_size=batch_size_cnn, verbose=verbose_cnn)

        # Predict class indices
        y_pred = np.argmax(model.predict(X_test), axis=1)
        if not predict_baseline:
            y_pred = np.array([idx2freq[i] for i in y_pred])

    else:
        raise ValueError("model_name should be one of: 'LDA', 'SVM', 'CNN'")

    # -----------------------------------------------------
    # Compute accuracy or sigmoid analysis
    # -----------------------------------------------------
    if predict_baseline:
        sorted_contrasts = {}
        sorted_accuracies = {}

        frequencies = np.unique(np.concatenate(test_data['Frequency'].values))
        for freq in frequencies:
            freq_idx = np.where(test_data['Frequency'] == freq)[0]  # Get indices for this frequency
            freq_y_test = y_test_raw[freq_idx]
            freq_y_pred = y_pred[freq_idx]

            # Compute accuracy per contrast for this frequency
            freq_contrasts = test_data['Contrast'].iloc[freq_idx].values
            unique_contrasts = np.unique(freq_contrasts)

            contrast_accuracy_dict = {}
            for contrast in unique_contrasts:
                contrast_idx = np.where(freq_contrasts == contrast)[0]
                contrast_y_test = freq_y_test[contrast_idx]
                contrast_y_pred = freq_y_pred[contrast_idx]
                accuracy = accuracy_score(contrast_y_test, contrast_y_pred)
                contrast_accuracy_dict[contrast.item() if isinstance(contrast, np.ndarray) else contrast] = accuracy


            # Sort contrasts and accuracies
            sorted_contrasts[freq] = np.array(sorted(contrast_accuracy_dict.keys()))
            sorted_accuracies[freq] = np.array([contrast_accuracy_dict[c] for c in sorted_contrasts[freq]])

        if plot:
            plot_accuracy_vs_contrast(
                sorted_contrasts,
                sorted_accuracies,
                sigmoid_params=None,
                output_dir=output_dir,
                filename=filename,
                predict_baseline=True,
                frequencies=frequencies,
                title=title,
                max_fev=max_fev
            )

    else:
        # Compute accuracy as a function of contrast
        accuracy_dict = {}
        contrast_test = np.array([float(label) for label in test_data.Contrast])
        unique_contrasts = np.unique(contrast_test)

        for contrast in unique_contrasts:
            idx = np.where(contrast_test == contrast)[0]
            if len(idx) < min_samples:
                print(f"Skipping contrast {contrast} due to insufficient samples ({len(idx)} samples).")
                continue

            true_labels = y_test_raw[idx]
            predicted_labels = y_pred[idx]

            accuracy = accuracy_score(true_labels, predicted_labels)
            accuracy_dict[contrast] = accuracy

        sorted_contrasts = np.array(sorted(accuracy_dict.keys()))
        sorted_accuracies = np.array([accuracy_dict[c] for c in sorted_contrasts])



        initial_guess = [0.5, 1.0, 1.0, np.median(sorted_contrasts)]
        sigmoid_params, _ = curve_fit(sigmoid, sorted_contrasts, sorted_accuracies, p0=initial_guess, maxfev=max_fev)

        if plot:
            plot_accuracy_vs_contrast(sorted_contrasts, sorted_accuracies, sigmoid_params, output_dir=output_dir, filename=filename, predict_baseline=predict_baseline, title=title, max_fev=max_fev)

def augment_with_shuffled_baseline(eeg_data_epochs, contrast_df, ress=False):
    """
    Create a combined MNE EpochsArray of original EEG data and shuffled artificial baselines
    and return the augmented metadata as a DataFrame.

    Parameters:
        eeg_data_epochs (mne.Epochs): Input EEG epochs.
        contrast_df (pd.DataFrame): DataFrame containing contrast values for each trial.
        ress (bool): Whether to shuffle across all channels (True) or per channel (False).

    Returns:
        mne.EpochsArray: Augmented epochs array with original and shuffled baselines.
        pd.DataFrame: DataFrame with updated metadata for the augmented data.
    """

    # Extract original EEG data and metadata
    if not ress:
        eeg_data = eeg_data_epochs.get_data(copy=True)  # Shape: (n_trials, n_channels, n_times)
        info = eeg_data_epochs.info  # Retain original metadata

    else:
        eeg_data = eeg_data_epochs
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

    # Create metadata for Stimulus or Baseline
    stimulus_or_baseline = np.concatenate([
        np.ones(len(eeg_data)),  # Original data marked as 1
        np.full(len(shuffled_baselines), 2)  # Shuffled data marked as 2
    ])

    # Combine metadata
    original_metadata = contrast_df.copy()
    original_metadata['Stimulus or Baseline'] = 1

    shuffled_metadata = contrast_df.copy()
    shuffled_metadata['Stimulus or Baseline'] = 2

    augmented_metadata = pd.concat([original_metadata, shuffled_metadata], ignore_index=True)
    #augmented_metadata = augmented_metadata.sample(frac=1).reset_index(drop=True)

    # Create new epochs with augmented data
    if not ress:
        augmented_epochs = mne.EpochsArray(
            augmented_data,
            info,
            verbose=False
        )

        return augmented_epochs, augmented_metadata
    else:
        return augmented_data, augmented_metadata
