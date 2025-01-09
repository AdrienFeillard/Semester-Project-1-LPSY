import os
import mne
import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np
from matplotlib import ticker
from matplotlib.animation import FuncAnimation
import seaborn as sns
import gc

from matplotlib.ticker import LogLocator, AutoLocator, AutoMinorLocator
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd

# === Visualization Functions ===
def plot_topomap(psd, bands, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig = psd.plot_topomap(bands=bands, vlim="joint", show=False, dB=True, outlines='head')
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
    Creates an animated topomap video from evoked data, averaging across ten time points for each frame.

    Parameters:
        evoked (mne.Evoked): The evoked object containing the averaged epochs.
        output_filename (str): Name of the output video file (default is "topomap_animation.mp4").
        fps (int): Frames per second for the animation (default is 30).
        ch_type (str): The channel type to plot (default is 'eeg').
        cmap (str): Colormap for the topomap (default is 'Reds').
    """
    # Define parameters for the animation
    sfreq = evoked.info['sfreq']  # Sampling frequency
    n_points = int(sfreq / fps)  # Step size for animation frames
    avg_window = 50  # Number of time points to average

    # Get time indices for frames, averaging every `avg_window` points
    time_indices = np.arange(0, len(evoked.times) - avg_window, n_points)
    time_bins = [(idx, idx + avg_window) for idx in time_indices]

    # Initialize the figure
    fig, ax = plt.subplots()
    vmin, vmax = evoked.data.min(), evoked.data.max()  # Dynamic scaling based on data range
    topomap_kwargs = dict(vlim=(vmin, vmax), contours=0, sensors=True)

    # Function to update the topomap at each frame
    def update_topomap(frame_idx):
        start, end = time_bins[frame_idx]
        time_avg = evoked.times[start:end].mean()  # Average time for title

        # Clear the axes and plot topomap
        ax.clear()
        evoked.plot_topomap(
            times=[time_avg],
            axes=ax,
            show=False,
            colorbar=False,
            ch_type=ch_type,
            sphere=(0, 0, 0, 100),
            outlines="head",
            extrapolate="head",
            cmap=cmap,
        )
        ax.set_title(f"Time: {time_avg:.3f} s")

    # Create the animation
    ani = FuncAnimation(fig, update_topomap, frames=len(time_bins), repeat=False)

    # Save the animation as a video file
    ani.save(output_filename, writer="ffmpeg", fps=fps)
    print(f"Topomap video saved as '{output_filename}'")



def save_psd_plot(psd, output_dir, filename, title=''):
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
        plt.plot(eigvecs[:, i], label=f"Eigenvector {i + 1}")
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


def sigmoid(x, a, b, c, d):
    exp_term = np.exp(-c * (x - d))
    # Prevent overflow by limiting the range of the exponential term
    exp_term = np.clip(exp_term, a_min=np.finfo(np.float64).tiny, a_max=np.finfo(np.float64).max)
    return a + (b - a) / (1 + exp_term)


def ensure_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'PNG'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SVG'), exist_ok=True)

def set_plot_styles(ax):
    ax.set_xscale('log')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.grid(False)

def set_ticks(ax, contrasts):
    x_ticks = np.logspace(np.log10(1), np.log10(10), num=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def calculate_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def plot_sigmoid_fit(ax, contrasts, accuracies, sigmoid_params, color='r', label=None,display_x_75 =True):
    x_fit = np.logspace(np.log10(contrasts.min()), np.log10(contrasts.max()), 500)
    y_fit = sigmoid(x_fit, *sigmoid_params)
    r_squared = calculate_r_squared(accuracies, sigmoid(contrasts, *sigmoid_params))
    ax.plot(x_fit, y_fit, color=color, linestyle='-', label=f"{label} ($R^2$={r_squared:.3f})")

    # Highlight the x-value at 75% accuracy
    y_75 = 0.75
    x_75_idx = np.where(y_fit >= y_75)[0][0] if np.any(y_fit >= y_75) else None
    if display_x_75:
        if x_75_idx is not None:
            x_75 = x_fit[x_75_idx]
            ax.annotate(
                f"{x_75:.2f}",
                (x_75, y_75),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=10,
                ha="left"
            )
    return x_fit, y_fit

def save_plot(fig, output_dir, filename):
    fig.savefig(os.path.join(output_dir, f'SVG/{filename}.svg'))
    fig.savefig(os.path.join(output_dir, f'PNG/{filename}.png'))
    plt.close(fig)

def plot_frequency_specific(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title):
    colors = ["blue", "green", "orange", "purple", "cyan"]
    color_map = {freq: colors[i % len(colors)] for i, freq in enumerate(frequencies)}

    for freq in frequencies:
        fig, ax = plt.subplots(figsize=(8, 6))
        set_plot_styles(ax)
        ax.scatter(sorted_contrasts[freq], sorted_accuracies[freq], marker="o", color="black")

        if len(sorted_contrasts[freq]) > 0:
            freq_params, _ = curve_fit(
                sigmoid,
                sorted_contrasts[freq],
                sorted_accuracies[freq],
                p0=[0.5, 1.0, 1.0, np.median(sorted_contrasts[freq])],
                bounds=([0, 0, 0, 0], [1, 1, 10, np.max(sorted_contrasts[freq])]),
                maxfev=max_fev
            )
            x_fit, y_fit = plot_sigmoid_fit(ax, sorted_contrasts[freq], sorted_accuracies[freq], freq_params, label=f"Sigmoid ({freq_mapping.get(freq, f'Frequency {freq}')})")

        set_ticks(ax, sorted_contrasts[freq])
        ax.set_xlabel("Contrast (Log Scale)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{title} - {freq_mapping.get(freq, f'Frequency {freq}')}", fontsize=10, fontweight="bold")
        ax.axhline(y=0.75, color="black", linestyle="--", label="75% Accuracy")
        ax.legend(loc="lower right", framealpha=0.7, fontsize=10)

        save_plot(fig, output_dir, f"{filename}_{freq}")

def plot_combined(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    set_plot_styles(ax)

    colors = ["blue", "green", "orange", "purple", "cyan"]
    color_map = {freq: colors[i % len(colors)] for i, freq in enumerate(frequencies)}

    for freq in frequencies:
        ax.scatter(sorted_contrasts[freq], sorted_accuracies[freq], color=color_map[freq], marker="o")

        if len(sorted_contrasts[freq]) > 0:
            freq_params, _ = curve_fit(
                sigmoid,
                sorted_contrasts[freq],
                sorted_accuracies[freq],
                p0=[0.5, 1.0, 1.0, np.median(sorted_contrasts[freq])],
                bounds=([0, 0, 0, 0], [1, 1, 10, np.max(sorted_contrasts[freq])]),
                maxfev=max_fev
            )
            plot_sigmoid_fit(ax, sorted_contrasts[freq], sorted_accuracies[freq], freq_params, color=color_map[freq], label=f"Sigmoid ({freq_mapping.get(freq, f'Frequency {freq}')})", display_x_75=False)

    # Add mean accuracy data points and sigmoid fit
    all_contrasts = np.unique(np.concatenate([sorted_contrasts[freq] for freq in frequencies]))
    mean_accuracies = [
        np.mean([
            sorted_accuracies[freq][np.where(sorted_contrasts[freq] == contrast)[0][0]]
            for freq in frequencies if contrast in sorted_contrasts[freq]
        ]) for contrast in all_contrasts
    ]

    ax.scatter(all_contrasts, mean_accuracies, color="black", marker="o")

    combined_params, _ = curve_fit(
        sigmoid,
        all_contrasts,
        mean_accuracies,
        p0=[0.5, 1.0, 1.0, np.median(all_contrasts)],
        bounds=([0, 0, 0, 0], [1, 1, 10, np.max(all_contrasts)]),
        maxfev=max_fev
    )
    plot_sigmoid_fit(ax, all_contrasts, mean_accuracies, combined_params, color="red", label="Mean Sigmoid Fit", display_x_75=False)

    set_ticks(ax, np.concatenate([sorted_contrasts[freq] for freq in frequencies]))
    ax.set_xlabel("Contrast (Log Scale)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}", fontsize=10, fontweight="bold")
    ax.axhline(y=0.75, color="black", linestyle="--", label="75% Accuracy")
    ax.legend(loc="lower right", framealpha=0.7, fontsize=10)

    save_plot(fig, output_dir, f"{filename}_combined")

def plot_mean_accuracy(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    set_plot_styles(ax)

    all_contrasts = np.unique(np.concatenate([sorted_contrasts[freq] for freq in frequencies]))
    mean_accuracies = [
        np.mean([
            sorted_accuracies[freq][np.where(sorted_contrasts[freq] == contrast)[0][0]]
            for freq in frequencies if contrast in sorted_contrasts[freq]
        ]) for contrast in all_contrasts
    ]

    ax.scatter(all_contrasts, mean_accuracies, color="black", marker="o")

    combined_params, _ = curve_fit(
        sigmoid,
        all_contrasts,
        mean_accuracies,
        p0=[0.5, 1.0, 1.0, np.median(all_contrasts)],
        bounds=([0, 0, 0, 0], [1, 1, 10, np.inf]),
        maxfev=max_fev
    )
    plot_sigmoid_fit(ax, all_contrasts, mean_accuracies, combined_params, label="Mean Sigmoid Fit")

    set_ticks(ax, all_contrasts)
    ax.set_xlabel("Contrast (Log Scale)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}", fontsize=10, fontweight="bold")
    ax.axhline(y=0.75, color="black", linestyle="--", label="75% Accuracy")
    ax.legend(loc="lower right", framealpha=0.7, fontsize=10)

    save_plot(fig, output_dir, f"{filename}_mean_accuracy")

def plot_accuracy_vs_contrast(sorted_contrasts, sorted_accuracies, sigmoid_params=None, output_dir='', filename='', predict_baseline=False, frequencies=None, title='', max_fev=10000):
    ensure_directories(output_dir)

    freq_mapping = {1: "40 Hz", 2: "48 Hz"} if "adr" in title.lower() else {1: "12 Hz", 2: "20 Hz"}

    if not predict_baseline:
        if sigmoid_params is None:
            raise ValueError("Sigmoid parameters are required for frequency prediction.")

        fig, ax = plt.subplots(figsize=(8, 6))
        set_plot_styles(ax)
        ax.scatter(sorted_contrasts, sorted_accuracies, color="black", zorder=2)
        plot_sigmoid_fit(ax, sorted_contrasts, sorted_accuracies, sigmoid_params, label="Sigmoid Fit")

        set_ticks(ax, sorted_contrasts)
        ax.set_xlabel("Contrast (Log Scale)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axhline(y=0.75, color="black", linestyle="--", label="75% Accuracy")
        ax.legend(loc="lower right", framealpha=0.7, fontsize=10)

        save_plot(fig, output_dir, filename)
    else:
        plot_frequency_specific(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title)
        plot_combined(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title)
        plot_mean_accuracy(sorted_contrasts, sorted_accuracies, frequencies, freq_mapping, output_dir, filename, max_fev, title)



def plot_accuracy_behavior(df, output_dir, filename, title):
    """
    Function to plot accuracy behavior with a Weibull fit and dynamically adjusting ticks.
    """
    os.makedirs(os.path.join(output_dir, 'PNG'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SVG'), exist_ok=True)

    # Group and process data
    accuracy_data = df.groupby('contrast').agg(
        accuracy=('HITS', 'mean')
    ).reset_index()
    contrasts = accuracy_data['contrast'].values
    accuracies = accuracy_data['accuracy'].values

    # Ensure contrasts are positive
    if np.any(contrasts <= 0):
        print("Warning: Non-positive contrasts detected. Ensure contrasts are strictly positive.")
        return

    # Weibull function definition
    def weibull(x, gamma, lambda_, k):
        x = np.clip(x, 1e-10, None)  # Avoid division by zero
        return gamma * (1 - np.exp(-((x / lambda_) ** k)))

    # Fit Weibull model
    p0_weibull = [1.0, np.max(contrasts) / 2, 1.0]
    try:
        popt_weibull, _ = curve_fit(
            weibull, contrasts, accuracies, p0=p0_weibull,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=100000
        )
    except RuntimeError as e:
        print(f"Error fitting Weibull curve: {e}")
        return


    # Generate Weibull curve
    x_fit = np.logspace(np.log10(min(contrasts)), np.log10(max(contrasts)), 500)
    y_fit_weibull = weibull(x_fit, *popt_weibull)

    # Calculate R² for Weibull fit
    residuals_weibull = accuracies - weibull(contrasts, *popt_weibull)
    r_squared_weibull = 1 - (np.sum(residuals_weibull**2) / np.sum((accuracies - np.mean(accuracies))**2))

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    ax.set_facecolor('white')

    # Set x-axis ticks dynamically
    """ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[], numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))"""
    # Set custom ticks and labels for the x-axis (log scale)
    x_ticks = np.logspace(np.log10((1)), np.log10((10)), num=2)  # 5 ticks between min and max
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)  # Display real values


# Set y-axis ticks automatically
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Plot data and Weibull fit
    ax.plot(contrasts, accuracies, 'o', color='black')
    ax.plot(x_fit, y_fit_weibull, '-', label=f'Weibull fit (R²={r_squared_weibull:.2f})', color='red')
    ax.axhline(0.75, color='black', linestyle='--', label='75% Accuracy')

    # Add 75% accuracy point if it exists
    if np.any(y_fit_weibull >= 0.75):
        f_weibull = interp1d(y_fit_weibull, x_fit, bounds_error=False, fill_value="extrapolate")
        x_75_weibull = f_weibull(0.75)

        ax.text(
            x_75_weibull * 1.1, 0.75,  # Dynamically position text to the right of the point
            f'{x_75_weibull:.2f}', fontsize=9,
            verticalalignment='bottom', horizontalalignment='left', zorder=10
        )

    # Add labels, title, and legend
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Contrast (Log Scale)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(
        loc='lower right',
        framealpha=0.7,
        fontsize=10
    )

    # Save plot
    fig.savefig(os.path.join(output_dir, 'SVG', f'{filename}.svg'))
    fig.savefig(os.path.join(output_dir, 'PNG', f'{filename}.png'))
    plt.close()

def plot_initial_signal(raw_signal, time_vector, output_dir, title_prefix):
    """
    Plot the initial raw EEG signals across all electrodes with a different shade of gray for each,
    and the mean ± standard deviation in red.
    """
    output_dir = os.path.join(output_dir, 'Raw signal')
    os.makedirs(output_dir, exist_ok=True)

    # Average across epochs for visualization
    avg_signal = np.mean(raw_signal, axis=0)  # Shape: [n_electrodes, n_times]
    mean_signal = np.mean(avg_signal, axis=0)  # Shape: [n_times]
    std_signal = np.std(avg_signal, axis=0)    # Shape: [n_times]

    plt.figure(figsize=(10, 6))

    # Plot each electrode's signal in black with random transparency
    """for electrode_signal in avg_signal:
        random_alpha = np.random.uniform(0.7, 0.9)  # Random transparency between 0.1 and 0.6
        plt.plot(time_vector, electrode_signal, color='black', alpha=random_alpha)"""

    # Plot mean and standard deviation
    plt.fill_between(time_vector, mean_signal - std_signal, mean_signal + std_signal,
                     color='red', alpha=0.5, label="Mean ± SD")
    print(mean_signal.shape)
    plt.plot(time_vector, mean_signal, color='black', label="Mean Signal", linewidth=2)




    plt.title(f"Initial Signal (Averaged Across Epochs)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_initial_signal.png"))
    plt.close()


def plot_target_filtered_signal(filtered_signal, time_vector, target_freq,output_dir,title_prefix):
    """
    Plot the signal filtered for the target frequency (averaged across epochs).
    """
    output_dir = os.path.join(output_dir, "Target Filtered Signal")
    os.makedirs(output_dir, exist_ok=True)

    avg_signal = np.mean(filtered_signal, axis=0)  # Shape: [n_electrodes, n_times]
    mean_signal = np.mean(avg_signal, axis=0)     # Shape: [n_times]
    std_signal = np.std(avg_signal, axis=0)       # Shape: [n_times]

    plt.figure(figsize=(10, 6))

    # Plot mean signal and standard deviation
    plt.fill_between(time_vector, mean_signal - std_signal, mean_signal + std_signal,
                     color='red', alpha=0.2, label=f"Mean ± SD ({target_freq} Hz)")
    plt.plot(time_vector, mean_signal, color='black', label=f"Mean Filtered Signal ({target_freq} Hz)", linewidth=2)

    plt.title(f" Target Frequency Filtered Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_target_frequency_filtered.png"))
    plt.close()


def plot_reference_filtered_signals(ref_low, ref_high, time_vector,target_freq, output_dir,title_prefix):
    """
    Plot the signals filtered for the reference frequencies (averaged across epochs).
    """
    output_dir = os.path.join(output_dir, "Ref filtered Signals")
    os.makedirs(output_dir, exist_ok=True)

    avg_low = np.mean(ref_low, axis=0)   # Shape: [n_electrodes, n_times]
    avg_high = np.mean(ref_high, axis=0)  # Shape: [n_electrodes, n_times]

    mean_low = np.mean(avg_low, axis=0)  # Shape: [n_times]
    std_low = np.std(avg_low, axis=0)    # Shape: [n_times]

    mean_high = np.mean(avg_high, axis=0)  # Shape: [n_times]
    std_high = np.std(avg_high, axis=0)    # Shape: [n_times]

    plt.figure(figsize=(10, 6))

    # Plot mean and standard deviation for low reference
    plt.fill_between(time_vector, mean_low - std_low, mean_low + std_low,
                     color='green', alpha=0.2, label=f"Low Reference Mean ± SD ({target_freq-1} Hz)")
    plt.plot(time_vector, mean_low, color='darkgreen', label=f"Low Reference Mean ({target_freq-1} Hz)", linewidth=2)

    # Plot mean and standard deviation for high reference
    plt.fill_between(time_vector, mean_high - std_high, mean_high + std_high,
                     color='red', alpha=0.2, label=f"High Reference Mean ± SD ({target_freq+1} Hz)")
    plt.plot(time_vector, mean_high, color='darkred', label=f"High Reference Mean ({target_freq+1} Hz)", linewidth=2)

    plt.title(f" Reference Frequency Filtered Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_reference_frequencies_filtered.png"))
    plt.close()


def plot_final_ress_component(ress_component, time_vector, output_dir,title_prefix):
    """
    Plot the final RESS component (dimension-reduced, averaged across epochs).
    """
    output_dir = os.path.join(output_dir, "Final RESS Component")
    os.makedirs(output_dir, exist_ok=True)

    avg_component = np.mean(ress_component, axis=0)  # Shape: [n_times]

    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, avg_component, label="Filtered Signal", color="black", linewidth=2)
    plt.title("Filtered Signal (Averaged Across Epochs)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_final_ress_component.png"))
    plt.close()

def compute_and_plot_mean_psd_topo(train_epochs, test_epochs, fmin=0.0, fmax=60.0, color='r', save_path='./Image/test.png', title=''):
    """
    Compute and plot the mean PSD across train and test datasets using Welch's method.

    Parameters:
    train_epochs : mne.Epochs
        The epochs object containing train EEG data.
    test_epochs : mne.Epochs
        The epochs object containing test EEG data.
    fmin : float
        Minimum frequency for PSD computation.
    fmax : float
        Maximum frequency for PSD computation.
    save_path : str
        Path to save the resulting plot.

    Returns:
    None
    """
    """# Compute PSD for train and test datasets
    train_psd = train_epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=0.0, tmax=3.0)
    test_psd = test_epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=0.0, tmax=3.0)"""

    # Combine PSD data from train and test
    combined_data = np.concatenate((train_epochs.get_data(copy=True), test_epochs.get_data(copy=True)), axis=0)

    combined_epochs = mne.EpochsArray(
        data=combined_data,
        info=train_epochs.info,
        tmin=train_epochs.tmin,
        events=None,
        event_id=None,
        verbose=False
    )
    combined_spectrum = combined_epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=0.0, tmax=3.0)


    """fig = combined_spectrum.plot(average=True, color=color)
    fig.suptitle(title, fontsize=16)  # Set the title for the plot

    fig.savefig(save_path)
    plt.close(fig)"""
    bands = {'11 - 13 Hz': (11, 13),
             '19 - 21 Hz': (19, 21),
             '39 - 41 Hz': (39, 41),
             '47 - 49 Hz': (47, 49),}
    electrodes_to_keep = []
    electrodes_to_keep.extend([f'A{i}' for i in range(2, 33)])
    electrodes_to_keep.extend([f'B{i}' for i in range(2, 20)])
    electrodes_to_keep.extend([f'D{i}' for i in range(16, 18)])
    electrodes_to_keep.extend([f'D{i}' for i in range(24, 33)])

    # Create the mask
    channel_names = combined_epochs.info['ch_names']
    mask = np.array([ch in electrodes_to_keep for ch in channel_names])
    fig = combined_spectrum.plot_topomap(bands=bands, ch_type=None, normalize=False, agg_fun=None, dB=True, sensors=True, show_names=False, mask=mask, mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0, markersize=2), contours=0, outlines='head', sphere=(0, 0, 0,100), image_interp='cubic', extrapolate='head', border='mean', res=1080, size=100, cmap=None, vlim=(None, None), cnorm=None, colorbar=False, cbar_fmt='auto', units=None, axes=None, show=False)
    fig.suptitle(title, fontsize=16)  # Set the title for the plot

    fig.savefig(save_path)
    plt.close(fig)