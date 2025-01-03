import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns
import gc
from scipy.optimize import curve_fit

# === Visualization Functions ===
def plot_topomap(psd, bands,output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot_topomap(bands=bands, vlim="joint", show=False, dB=True)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
    plt.close(fig)
    print(f"Topomap plot saved as '{os.path.join(output_dir, f'{filename}.png')}'")


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

def save_psd_plot(psd, output_dir, filename, title =''):
    """
    Save the plot of an MNE PSD object to a file.

    Parameters:
    psd : mne.time_frequency.PSD
        The PSD object returned by mne.time_frequency.compute_psd.
    output_dir : str
        Directory where the plot will be saved.
    filename : str
        Name of the output plot file (without extension).
    """
    import matplotlib
    matplotlib.use('Agg')  # Use the non-interactive Agg backend for saving plots

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot using MNE's plotting method
    fig = psd.plot()  # This creates and returns a Matplotlib figure
    filepath = os.path.join(output_dir, f'{filename}.png')

    if title:
        ax = fig.axes[0]  # Access the first (and likely only) axes of the figure
        ax.set_title(title)

    # Save the figure
    fig.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory

    print(f"PSD plot saved as '{filepath}'")



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
    os.makedirs(output_dir, exist_ok=True)

    for component_idx in range(ica.n_components_):
        fig = ica.plot_properties(epochs, picks=component_idx, show=False, verbose=False)

        for ax in fig[0].axes:
            if ax.get_xlabel() == "Frequency (Hz)":
                ax.set_xlim(freq_range)

        fig_path = os.path.join(output_dir, f"{filename}component_{component_idx}_properties.png")
        fig[0].savefig(fig_path)
        plt.close(fig[0])  # Close the specific figure
        for f in fig[1:]:  # Close any additional figures
            plt.close(f)
    gc.collect()
    print(f"All ICA component properties saved to: {output_dir}")


def plot_eigenvalues_and_vectors(eigvals, eigvecs, ch_names, top_n=5):
    """
    Plot the eigenvalues as a heatmap and the top eigenvectors as line plots.

    Parameters:
        eigvals (np.ndarray): Eigenvalues, sorted in descending order.
        eigvecs (np.ndarray): Eigenvectors corresponding to the eigenvalues, shape (n_channels, n_channels).
        ch_names (list): List of channel names.
        top_n (int): Number of top eigenvectors to plot.

    Returns:
        None: Displays the plots.
    """
    # Heatmap of eigenvalues
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.diag(eigvals[:top_n]), annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title(f"Top {top_n} Eigenvalues")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue Index")
    plt.show()

    # Plot the top eigenvectors
    plt.figure(figsize=(12, 6))
    for i in range(top_n):
        plt.plot(eigvecs[:, i], label=f"Eigenvector {i+1}")
    plt.title(f"Top {top_n} Eigenvectors")
    plt.xlabel("Channels")
    plt.ylabel("Weight")
    plt.xticks(ticks=range(len(ch_names)), labels=ch_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_filtered_signal_spectrum(filtered_data, sfreq, info, title="Filtered Signal Spectrum"):
    """
    Plot the power spectrum of the filtered signal using MNE.

    Parameters:
        filtered_data (np.ndarray): Filtered EEG data of shape (n_trials, n_channels, n_times).
        sfreq (float): Sampling frequency of the EEG data.
        info (mne.Info): MNE info object with channel information.
        title (str): Title for the spectrum plot.

    Returns:
        None: Displays the power spectrum plot.
    """
    # Create an MNE EpochsArray object from the filtered data
    epochs_filtered = mne.EpochsArray(filtered_data, info)

    # Plot the PSD
    fig = epochs_filtered.plot_psd(fmax=100, show=False)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_contrast(
        sorted_contrasts,
        sorted_accuracies,
        sigmoid_params=None,
        output_dir='',
        filename='',
        predict_baseline=False,
        frequencies=None,
        title ='',
        max_fev = 10000,
):
    """
    Plot accuracy as a function of contrast with sigmoid fit or baseline/stimulus predictions.

    Parameters:
    -----------
    sorted_contrasts : dict or np.ndarray
        Sorted array of unique contrast values or dictionary for each frequency.
    sorted_accuracies : dict or np.ndarray
        Accuracies corresponding to the sorted contrasts or dictionary for each frequency.
    sigmoid_params : tuple, optional
        Parameters of the fitted sigmoid function (a, b, c, d). Required for frequency prediction.
    output_dir : str, optional
        Directory to save the plots.
    filename : str, optional
        Filename prefix for saving the plots.
    predict_baseline : bool, optional
        If True, plots results for baseline/stimulus predictions.
    frequencies : list, optional
        List of unique frequencies for frequency-specific plots. Required for baseline/stimulus case.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'PNG'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SVG'), exist_ok=True)

    if not predict_baseline:
        # Case 1: Frequency prediction
        if sigmoid_params is None:
            raise ValueError("Sigmoid parameters are required for frequency prediction.")

        # Unpack sigmoid parameters
        a, b, c, d = sigmoid_params

        # Sigmoid function for plotting
        def sigmoid(x, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (x - d)))

        # Generate x values for smooth plotting
        x_fit = np.linspace(sorted_contrasts.min(), sorted_contrasts.max(), 500)
        y_fit = sigmoid(x_fit, a, b, c, d)

        # Find the x-value at 75% accuracy
        y_75 = 0.75
        x_75_idx = np.where(y_fit >= y_75)[0][0] if np.any(y_fit >= y_75) else None
        if x_75_idx is not None:
            x_75 = x_fit[x_75_idx]

        # Calculate R²
        residuals = sorted_accuracies - sigmoid(sorted_contrasts, a, b, c, d)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((sorted_accuracies - np.mean(sorted_accuracies))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(sorted_contrasts, sorted_accuracies, color="b", label="Accuracy (data)", zorder=2)
        plt.plot(x_fit, y_fit, color="r", linestyle="-", label=f"Sigmoid fit ($R^2$={r_squared:.3f})", zorder=1)

        # Highlight the x-value at 75% accuracy
        if x_75_idx is not None:
            plt.annotate(f"{x_75:.2f}", (x_75, y_75), textcoords="offset points", xytext=(5, 5), ha="center", fontsize=12)

        # Additional plot details
        plt.axhline(y=0.75, color="g", linestyle="--", label="75% Accuracy")
        plt.xlabel("Contrast", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(title, fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save plot
        save_path = os.path.join(output_dir, f'SVG/{filename}.svg')
        plt.savefig(save_path)
        save_path = os.path.join(output_dir, f'PNG/{filename}.png')
        plt.savefig(save_path)
        plt.close()

    else:
        # Case 2: Baseline/Stimulus prediction
        def sigmoid(x, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (x - d)))

        # 1. Overall plot (aggregate all frequencies)
        plt.figure(figsize=(8, 6))

        # Gather all unique contrast values across frequencies
        all_unique_contrasts = np.unique(np.concatenate([sorted_contrasts[freq] for freq in frequencies]))

        # Compute mean accuracy for each unique contrast
        mean_accuracies = []
        for contrast in all_unique_contrasts:
            accuracies_at_contrast = [
                sorted_accuracies[freq][np.where(sorted_contrasts[freq] == contrast)[0][0]]
                for freq in frequencies if contrast in sorted_contrasts[freq]
            ]
            mean_accuracies.append(np.mean(accuracies_at_contrast))

        mean_accuracies = np.array(mean_accuracies)

        # Plot mean values
        plt.scatter(all_unique_contrasts, mean_accuracies, label="Mean Accuracy", color="b", marker="o")

        # Fit sigmoid for mean data
        if len(all_unique_contrasts) > 0:
            combined_params, _ = curve_fit(sigmoid, all_unique_contrasts, mean_accuracies, p0=[0.5, 1.0, 1.0, np.median(all_unique_contrasts)], maxfev=max_fev)
            x_fit = np.linspace(min(all_unique_contrasts), max(all_unique_contrasts), 500)
            y_fit = sigmoid(x_fit, *combined_params)

            # Calculate R²
            residuals = mean_accuracies - sigmoid(all_unique_contrasts, *combined_params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mean_accuracies - np.mean(mean_accuracies))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Plot sigmoid fit
            plt.plot(x_fit, y_fit, color="r", linestyle="-", label=f"Sigmoid fit (Mean, $R^2$={r_squared:.3f})")

            # Add 0.75 marker
            y_75 = 0.75
            x_75_idx = np.where(y_fit >= y_75)[0][0] if np.any(y_fit >= y_75) else None
            if x_75_idx is not None:
                x_75 = x_fit[x_75_idx]
                plt.annotate(f"{x_75:.2f}", (x_75, y_75), textcoords="offset points", xytext=(5, 5), ha="center")

        plt.xlabel("Contrast", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(title, fontsize=10)
        plt.legend(fontsize=12)
        plt.axhline(y=0.75, color="g", linestyle="--", label="75% Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        save_path = os.path.join(output_dir, f'SVG/{filename}_overall.svg')
        plt.savefig(save_path)
        save_path = os.path.join(output_dir, f'PNG/{filename}_overall.png')
        plt.savefig(save_path)
        plt.close()

        # 2. Frequency-specific plots
        for freq in frequencies:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                sorted_contrasts[freq],
                sorted_accuracies[freq],
                label=f"Frequency {freq} Hz",
                marker="o"
            )

            # Fit sigmoid for each frequency
            if len(sorted_contrasts[freq]) > 0:
                freq_params, _ = curve_fit(sigmoid, sorted_contrasts[freq], sorted_accuracies[freq], p0=[0.5, 1.0, 1.0, np.median(sorted_contrasts[freq])],maxfev=max_fev)
                x_fit = np.linspace(sorted_contrasts[freq].min(), sorted_contrasts[freq].max(), 500)
                y_fit = sigmoid(x_fit, *freq_params)

                # Calculate R²
                residuals = sorted_accuracies[freq] - sigmoid(sorted_contrasts[freq], *freq_params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((sorted_accuracies[freq] - np.mean(sorted_accuracies[freq]))**2)
                r_squared = 1 - (ss_res / ss_tot)

                plt.plot(x_fit, y_fit, color="r", linestyle="-", label=f"Sigmoid fit ($R^2$={r_squared:.3f})")

                # Add 0.75 marker
                y_75 = 0.75
                x_75_idx = np.where(y_fit >= y_75)[0][0] if np.any(y_fit >= y_75) else None
                if x_75_idx is not None:
                    x_75 = x_fit[x_75_idx]
                    plt.annotate(f"{x_75:.2f}", (x_75, y_75), textcoords="offset points", xytext=(5, 5), ha="center")

            plt.xlabel("Contrast", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.title(f"{title} (Frequency ID: {freq})", fontsize=10)
            plt.legend(fontsize=12)
            plt.axhline(y=0.75, color="g", linestyle="--", label="75% Accuracy")
            plt.grid(True, linestyle="--", alpha=0.6)

            # Save plot
            save_path = os.path.join(output_dir, f'SVG/{filename}_freq_{freq}.svg')
            plt.savefig(save_path)
            save_path = os.path.join(output_dir, f'PNG/{filename}_freq_{freq}.png')
            plt.savefig(save_path)
            plt.close()

        # 3. Combined plot
        plt.figure(figsize=(8, 6))

        # Plot individual frequency-specific curves and fit sigmoids
        for freq in frequencies:
            plt.scatter(
                sorted_contrasts[freq],
                sorted_accuracies[freq],
                label=f"Frequency {freq} Hz",
                marker="o"
            )

            # Fit sigmoid for each frequency
            if len(sorted_contrasts[freq]) > 0:
                freq_params, _ = curve_fit(
                    sigmoid,
                    sorted_contrasts[freq],
                    sorted_accuracies[freq],
                    p0=[0.5, 1.0, 1.0, np.median(sorted_contrasts[freq])],
                    maxfev=max_fev)
                x_fit = np.linspace(sorted_contrasts[freq].min(), sorted_contrasts[freq].max(), 500)
                y_fit = sigmoid(x_fit, *freq_params)

                # Calculate R²
                residuals = sorted_accuracies[freq] - sigmoid(sorted_contrasts[freq], *freq_params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((sorted_accuracies[freq] - np.mean(sorted_accuracies[freq]))**2)
                r_squared = 1 - (ss_res / ss_tot)

                plt.plot(x_fit, y_fit, label=f"Sigmoid fit (Freq {freq} Hz, $R^2$={r_squared:.3f})", linestyle="-")

        # Add mean accuracy curve
        plt.scatter(
            all_unique_contrasts,
            mean_accuracies,
            label="Mean Accuracy",
            color="r",
            marker="o"
        )

        # Fit sigmoid for mean data
        if len(all_unique_contrasts) > 0:
            combined_params, _ = curve_fit(sigmoid, all_unique_contrasts, mean_accuracies, p0=[0.5, 1.0, 1.0, np.median(all_unique_contrasts)], maxfev=max_fev)
            x_fit = np.linspace(min(all_unique_contrasts), max(all_unique_contrasts), 500)
            y_fit = sigmoid(x_fit, *combined_params)

            # Calculate R²
            residuals = mean_accuracies - sigmoid(all_unique_contrasts, *combined_params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mean_accuracies - np.mean(mean_accuracies))**2)
            r_squared = 1 - (ss_res / ss_tot)

            plt.plot(x_fit, y_fit, color="r", linestyle="-", label=f"Sigmoid fit (Mean, $R^2$={r_squared:.3f})")

        plt.xlabel("Contrast", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(title, fontsize=10)
        plt.legend(fontsize=12)
        plt.axhline(y=0.75, color="g", linestyle="--", label="75% Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save combined plot
        save_path = os.path.join(output_dir, f'SVG/{filename}_combined.svg')
        plt.savefig(save_path)
        save_path = os.path.join(output_dir, f'PNG/{filename}_combined.png')
        plt.savefig(save_path)
        plt.close()


def plot_accuracy_behavior(df,output_dir, filename, title):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'PNG'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SVG'), exist_ok=True)

    accuracy_data = df.groupby('contrast').agg(
        accuracy=('HITS', 'mean')
    ).reset_index()

    # Extract contrasts and accuracies
    contrasts = accuracy_data['contrast'].values
    accuracies = accuracy_data['accuracy'].values

    # Initial guess for the parameters
    p0 = [max(accuracies), np.median(contrasts), 1, min(accuracies)]
    def sigmoid(x, a, b, c, d):
        return a + (b - a) / (1 + np.exp(-c * (x - d)))
    # Fit the sigmoid function to the data
    try:
        popt, _ = curve_fit(
            sigmoid, contrasts, accuracies, p0,
            method='dogbox', maxfev=100000
        )
    except RuntimeError as e:
        print(f"Error fitting curve: {e}")
        return

    # Generate x values for the fit line
    x_fit = np.linspace(min(contrasts), max(contrasts), 500)
    y_fit = sigmoid(x_fit, *popt)

    # Find the x value where the curve reaches 75% accuracy
    x_75 = x_fit[np.searchsorted(y_fit, 0.75)]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(contrasts, accuracies, 'o', label='Data points')
    plt.plot(x_fit, y_fit, '-', label='Sigmoid fit', color='red')
    plt.axhline(0.75, color='green', linestyle='--', label='75% Accuracy')
    plt.scatter(x_75, 0.75, color='blue', zorder=5)
    plt.text(x_75, 0.75, f'({x_75:.2f}, 0.75)', fontsize=9, verticalalignment='bottom')

    plt.title(title)
    plt.xlabel('Contrast')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f'SVG/{filename}.svg')
    plt.savefig(save_path)
    save_path = os.path.join(output_dir, f'PNG/{filename}.png')
    plt.savefig(save_path)
    plt.close()

