import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# === Visualization Functions ===
def plot_topomap(psd, bands,output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot_topomap(bands=bands, vlim="joint", show=False, dB=True)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
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

def save_psd_plot(psd, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot(show=False)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
    print(f"PSD plot saved as '{os.path.join(output_dir, f'{filename}.png')}'")


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

