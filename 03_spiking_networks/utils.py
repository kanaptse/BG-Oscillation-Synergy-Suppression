import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy import signal
from network_simulation import NetworkSimulation
import yaml
import os
from scipy.fft import fft, fftfreq

class NetworkVisualization:
    def __init__(self, simulation):
        """
        Initialize the visualization class with a reference to the simulation.

        Args:
            simulation (EINetworkSimulation): The simulation object containing the data to visualize.
        """
        self.simulation = simulation

    def calculate_firing_rate(self, pop, bin_width=10, normalize=False, drop_initial=0):
        """
        Calculate the firing rate over time for a population.

        Args:
            pop (str): The name of the population to plot ('exc' or 'inh').
            bin_width (int): The width of each bin in milliseconds.
            normalize (bool): Whether to normalize the firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.

        Returns:
            tuple: A tuple containing the bin edges and the firing rate.
        """
        # Get the spike train from the simulation results
        spike_train = self.simulation.get_spike_trains()[pop]

        # Concatenate all spike times for the population
        all_spikes = np.concatenate(spike_train)

        if drop_initial > 0:
            # Drop the initial portion of data
            all_spikes = all_spikes[all_spikes >= drop_initial]

            # Define the bins such that each bin edge represents the end of the look-back period
            bins = np.arange(drop_initial, self.simulation.T, bin_width)
        else:
            # Define the bins such that each bin edge represents the end of the look-back period
            bins = np.arange(0, self.simulation.T, bin_width)

        # Calculate the histogram
        spike_counts, _ = np.histogram(all_spikes, bins=bins)

        # Calculate the firing rate (spikes per second)
        firing_rate = spike_counts / (bin_width / 1000.0 * len(spike_train))

        # Normalize if required
        if normalize:
            firing_rate = firing_rate / firing_rate.max()

        return bins[1:], firing_rate

    def plot_firing_rate(self, pop, bin_width=10, ax=None, color='blue', normalize=False, drop_initial=0, label=None):
        """
        Plot the firing rate over time for a population.

        Args:
            pop (str): The name of the population to plot ('exc' or 'inh').
            bin_width (int): The width of each bin in milliseconds.
            ax (matplotlib.axes.Axes): The axes to plot on.
            color (str): The color of the plot line.
            normalize (bool): Whether to normalize the firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate the firing rate
        bin_right, firing_rate = self.calculate_firing_rate(pop, bin_width, normalize, drop_initial)

        if label is None:
            label = f'{pop} Firing Rate'
        # Plot the firing rate
        ax.plot(bin_right, firing_rate, label=label, color=color)

        # Set the title and labels
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized" if normalize else "Firing rate (Hz)")

        if ax is None:
            plt.show()

    def plot_raster(self, pop):
        """
        Plot a raster plot for the specified population.

        Args:
            pop (str): The name of the population to plot ('exc' or 'inh').
        """
        spike_trains = self.simulation.get_results()[f'{pop}_spikes']

        # Set up the figure
        fig, ax = plt.subplots()

        # Loop over each neuron and plot its spike times
        for i, spike_train in enumerate(spike_trains):
            # i is the neuron index, spike_train is the list of firing times
            ax.scatter(spike_train, [i] * len(spike_train), marker='|', color='black')

        # Label the axes
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron Index')
        ax.set_title(f'Raster Plot of {pop.capitalize()} Neuron Spike Trains')

        # Set the y-axis limits to ensure each neuron has its own space
        ax.set_ylim(-0.5, len(spike_trains) - 0.5)

        # Show the plot
        plt.show()

    def plot_power_spectrum(self, pop, bin_width=10, nperseg=256, ax=None, normalize=False, drop_initial=0, legend=None):
        """
        Plot the power spectrum of the firing rate for a given population.

        Args:
            pop (str): The name of the population to analyze ('exc' or 'inh').
            bin_width (int): The width of each bin in milliseconds.
            nperseg (int): Length of each segment for the Welch method.
            ax (matplotlib.axes.Axes): The axes to plot on.
            normalize (bool): Whether to normalize the firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Calculate the firing rate
        bin_right, firing_rate = self.calculate_firing_rate(pop, bin_width, normalize, drop_initial)

        # Calculate the sampling frequency from bin_right
        fs = 1000.0 / (bin_right[1] - bin_right[0])  # Convert bin width from ms to Hz

        # Calculate the power spectrum using Welch's method
        freqs, power = welch(firing_rate, fs=fs, nperseg=nperseg, nfft=nperseg)

        # Plot the power spectrum
        if legend is not None:
            ax.plot(freqs, power, label=legend)
        else:
            ax.plot(freqs, power)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'Power Spectrum of {pop} Firing Rate')
        ax.set_xlim(0, 100)


        if ax is None:
            plt.show()

    def calculate_beta_power(self, pop, bin_width=10, nperseg=256, normalize=False, drop_initial=0):
        """
        Calculate the beta power (average power between 12.5 and 30 Hz) of the firing rate for a given population.

        Args:
            pop (str): The name of the population to analyze ('exc' or 'inh').
            bin_width (int): The width of each bin in milliseconds.
            nperseg (int): Length of each segment for the Welch method.
            normalize (bool): Whether to normalize the firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.

        Returns:
            float: The average power in the beta frequency range (12.5-30 Hz).
        """
        # Calculate the firing rate
        bin_right, firing_rate = self.calculate_firing_rate(pop, bin_width, normalize, drop_initial)

        # Calculate the sampling frequency from bin_right
        fs = 1000.0 / (bin_right[1] - bin_right[0])  # Convert bin width from ms to Hz

        # Calculate the power spectrum using Welch's method
        freqs, power = welch(firing_rate, fs=fs, nperseg=nperseg, nfft=nperseg)

        # Find the indices of the frequencies in the beta range (12.5-30 Hz)
        beta_indices = np.where((freqs >= 12.5) & (freqs <= 30))[0]

        # Calculate the average power in the beta range
        ave_beta_freq = np.sum(freqs[beta_indices]*power[beta_indices])/np.sum(power[beta_indices])
        beta_power = np.mean(power[beta_indices])

        return ave_beta_freq, beta_power

    def compute_spectrogram(self, pop, bin_width=10, nperseg=256, noverlap=128,
                            demean=True, drop_initial=0):
        """
        Compute the spectrogram of the firing rate for a given population.

        Args:
            pop (str): The name of the population to analyze.
            bin_width (int): The width of each bin in milliseconds.
            nperseg (int): Length of each segment for the spectrogram.
            noverlap (int): Overlap between segments.
            demean (bool): Whether to subtract the mean from firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.

        Returns:
            tuple: (frequencies, times, Sxx, fs, times_ms)
        """
        # Calculate the firing rate
        bin_right, firing_rate = self.calculate_firing_rate(pop, bin_width, False, drop_initial)

        # Z-score the firing rate if requested
        if demean:
            firing_rate = firing_rate - np.mean(firing_rate)

        # Calculate the sampling frequency from bin_right
        fs = 1000.0 / (bin_right[1] - bin_right[0])  # Convert bin width from ms to Hz

        # Calculate the spectrogram
        frequencies, times, Sxx = signal.spectrogram(firing_rate, fs=fs,
                                                     nperseg=nperseg,
                                                     noverlap=noverlap,
                                                     scaling='spectrum')

        # Calculate time offset for proper alignment
        time_offset = drop_initial + (bin_width / 2)  # Add half bin width for center alignment
        times_ms = times * 1000 + time_offset  # Convert to ms and apply offset

        return frequencies, times, Sxx, fs, times_ms

    def plot_spectrogram(self, pop, bin_width=10, nperseg=256, noverlap=128, ax=None,
                         demean=True, drop_initial=0, max_freq=100, cmap='viridis',
                         global_vmin=None, global_vmax=None, title=None):
        """
        Plot the spectrogram of the firing rate for a given population.

        Args:
            pop (str): The name of the population to analyze.
            bin_width (int): The width of each bin in milliseconds.
            nperseg (int): Length of each segment for the spectrogram.
            noverlap (int): Overlap between segments.
            ax (matplotlib.axes.Axes): The axes to plot on.
            demean (bool): Whether to z-score the firing rate.
            drop_initial (int): The initial portion of data to drop in milliseconds.
            max_freq (int): Maximum frequency to display in Hz.
            cmap (str): Colormap for the spectrogram.
            global_vmin (float, optional): Global minimum value for color scaling.
            global_vmax (float, optional): Global maximum value for color scaling.

        Returns:
            matplotlib.image.AxesImage: The spectrogram image
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Compute the spectrogram
        frequencies, times, Sxx, fs, times_ms = self.compute_spectrogram(
            pop, bin_width, nperseg, noverlap, demean, drop_initial
        )

        # Use global vmin and vmax if provided, otherwise use local min and max
        vmin = global_vmin if global_vmin is not None else Sxx.min()
        vmax = global_vmax if global_vmax is not None else Sxx.max()

        # Plot the spectrogram
        im = ax.pcolormesh(times_ms, frequencies, Sxx,
                           shading='gouraud', cmap=cmap,
                           vmin=vmin, vmax=vmax)

        # Add a colorbar
        plt.colorbar(im, ax=ax, label='Power (dB)')

        # Set labels and title
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f'Spectrogram of {pop} Firing Rate')
        ax.set_ylim(0, max_freq)  # Set maximum frequency

        if ax is None:
            plt.show()

        return im

    def calculate_phase_difference(self, pop1, pop2, bin_width=2, drop_initial=100, min_power_threshold=1e-6):
        """
        Calculate the phase difference between two populations' firing rates.

        Args:
            pop1 (str): The name of the first population.
            pop2 (str): The name of the second population.
            bin_width (int): The width of each bin in milliseconds.
            drop_initial (int): The initial portion of data to drop in milliseconds.

        Returns:
        --------
        phase_diff : float
            Phase difference in radians (wrapped to [-π, π])
        dominant_freq : float
            Dominant frequency in Hz
        """

        # FFT of both signals
        _, signal1 = self.calculate_firing_rate(pop1, bin_width=bin_width, drop_initial=drop_initial)
        _, signal2 = self.calculate_firing_rate(pop2, bin_width=bin_width, drop_initial=drop_initial)
        fft1 = fft(signal1 - signal1.mean())
        fft2 = fft(signal2 - signal2.mean())

        fs = 1000.0 / bin_width  # Sampling frequency in Hz

        # Frequency array (positive frequencies only)
        freqs = fftfreq(len(signal1), 1 / fs)
        pos_freqs = freqs[:len(freqs) // 2]
        pos_fft1 = fft1[:len(freqs) // 2]
        pos_fft2 = fft2[:len(freqs) // 2]

        # Calculate power spectra
        power1 = np.abs(pos_fft1) ** 2
        power2 = np.abs(pos_fft2) ** 2

        # Find dominant frequency from signal1 (our reference)
        dominant_idx = np.argmax(np.abs(pos_fft1))
        dominant_freq = pos_freqs[dominant_idx]
        max_power1 = power1[dominant_idx]
        max_power2 = power2[dominant_idx]

        # Check if signals have sufficient power at the dominant frequency
        if max_power1 < min_power_threshold:
            print(f"Warning: Signal 1 ({pop1}) has no significant frequency components (likely constant or noise)")
            return np.nan, np.nan

        if max_power2 < min_power_threshold:
            print("Warning: Signal 2 ({pop2}) has no significant frequency components (likely constant or noise)")
            return np.nan, np.nan

        # Extract phases at dominant frequency
        phase1 = np.angle(pos_fft1[dominant_idx])
        phase2 = np.angle(pos_fft2[dominant_idx])

        # Calculate phase difference with signal1 as reference
        phase_diff = phase2 - phase1

        # Wrap to [-π, π]
        phase_diff = np.degrees(np.arctan2(np.sin(phase_diff), np.cos(phase_diff)))

        return round(phase_diff, 1), round(dominant_freq, 1)

from scipy.optimize import minimize

class StopOptimization(Exception):
    pass

def optimize_firing_rates(params_pop, params_conn_all, target_rates,
                          T=200, dt=0.5, use_gpu=True, drop_time=100, error_threshold=5):
    populations = list(target_rates.keys())
    last_results = [{'error': float('inf'), 'rates': {pop: 0 for pop in populations}}]

    def objective_function(currents):
        params_pop_copy = {pop: params_pop[pop].copy() for pop in params_pop}
        for i, pop in enumerate(populations):
            params_pop_copy[pop]['I'] = currents[i]

        sim = NetworkSimulation(params_pop_copy, params_conn_all, T, dt, use_gpu)
        sim.simulate(drop_time=drop_time)

        error = 0
        rates = {}
        for i, pop in enumerate(populations):
            actual_rate = sim.calculate_average_firing_rate(sim.pop_dict[pop].spike_train, start_time=drop_time)
            rates[pop] = actual_rate
            if pop == 'D2':
                error += (actual_rate - target_rates[pop])**2 * 25
            if pop == 'Proto':
                error += (actual_rate - target_rates[pop])**2 / 2
            else:
                error += (actual_rate - target_rates[pop])**2
            print(f"{pop}: current={currents[i]:.2f}, rate={actual_rate:.2f} Hz (target: {target_rates[pop]} Hz)")

        print(f"Total error: {error:.2f}")
        print("-----------------------------------")

        last_results[0] = {'error': error, 'rates': rates}

        if error < error_threshold:
            raise StopOptimization  # 💥 Force exit

        return error

    initial_currents = [params_pop[pop]['I'] for pop in populations]

    try:
        result = minimize(
            objective_function,
            initial_currents,
            method='Powell',
            options={
                'maxiter': 20,
                'disp': True,
                'ftol': 5e-2
            }
        )
    except StopOptimization:
        print("Optimization stopped early due to error threshold.")
        result = None

    optimized_params = {pop: params_pop[pop].copy() for pop in params_pop}
    if result and result.success:
        for i, pop in enumerate(populations):
            optimized_params[pop]['I'] = result.x[i]
    else:
        # fallback: use last known good currents
        for i, pop in enumerate(populations):
            optimized_params[pop]['I'] = initial_currents[i]

    return optimized_params, result, last_results[0]['rates']

def simulate_stn_intervention(T, params_pop, params_conn_all, dt=0.5, STN_start=400, STN_end=1200, seed = 422, colorbar_range=None):
    drop_time = 100

    print('STN inactive')
    network_sim_1 = NetworkSimulation(params_pop, params_conn_all, T, dt, seed=seed)

    activate_population_1 = {'pop_name': 'STN', 'start': 6000, 'end': 7000}
    network_sim_1.simulate(drop_time=drop_time, activate_population=activate_population_1)

    print('STN active')
    network_sim_2 = NetworkSimulation(params_pop, params_conn_all, T, dt, seed=seed)
    activate_population_2 = {'pop_name': 'STN', 'start': STN_start, 'end': STN_end}
    network_sim_2.simulate(sin_amp=5, drop_time=drop_time, activate_population=activate_population_2)

    network_vis_1 = NetworkVisualization(network_sim_1)
    network_vis_2 = NetworkVisualization(network_sim_2)

    bin_width = 2
    # Compute spectrograms first to determine global color scaling
    spec1_freq, spec1_times, spec1_Sxx, _, _ = network_vis_1.compute_spectrogram(
        pop='Proto',
        bin_width=bin_width,
        drop_initial=drop_time,
        nperseg=64,
        noverlap=32
    )
    spec2_freq, spec2_times, spec2_Sxx, _, _ = network_vis_2.compute_spectrogram(
        pop='Proto',
        bin_width=bin_width,
        drop_initial=drop_time,
        nperseg=64,
        noverlap=32
    )

    # # Compute difference of spectrograms (signed)
    spec_diff = spec2_Sxx - spec1_Sxx
    # print('Difference min',spec_diff.min(), 'max', spec_diff.max())

    # Create figure with custom axes
    fig = plt.figure(figsize=(20, 10))
    ax_spec_diff = fig.add_axes((0.1, 0.7, 1, 0.2))
    ax_rate = fig.add_axes((0.1, 0.4, 0.8, 0.2))
    ax_stn_proto_rate = fig.add_axes((0.1, 0.1, 0.8, 0.2))

    vmin, vmax = (colorbar_range if colorbar_range is not None else (None, None))
    im_diff = ax_spec_diff.pcolormesh(
        spec2_times * 1000 + drop_time + (bin_width / 2),
        spec2_freq,
        spec_diff,
        shading='gouraud',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(im_diff, ax=ax_spec_diff, label='Power Difference (dB)')
    ax_spec_diff.set_xlabel('Time (ms)')
    ax_spec_diff.set_ylabel('Frequency (Hz)')
    ax_spec_diff.set_title('Difference Spectrogram (STN active - STN inactive)')
    ax_spec_diff.set_ylim(10, 30)

    network_vis_2.plot_firing_rate('Proto', ax=ax_rate, normalize=False, drop_initial=drop_time, color='green',
                                   bin_width=bin_width, label='STN active')
    network_vis_1.plot_firing_rate('Proto', ax=ax_rate, normalize=False, drop_initial=drop_time, color='red', bin_width=bin_width, label='STN inactive')


    ax_rate.set_xlim(ax_spec_diff.get_xlim())
    ax_rate.set_title('Proto Firing Rate')
    ax_rate.legend()
    ax_rate.axvspan(STN_start, STN_end, color='green', alpha=0.2)

    ax_stn_proto_rate_twin = ax_stn_proto_rate.twinx()
    network_vis_2.plot_firing_rate('Proto', ax=ax_stn_proto_rate, drop_initial=drop_time, color='green', bin_width=bin_width)
    network_vis_2.plot_firing_rate('STN', ax=ax_stn_proto_rate_twin, drop_initial=drop_time, color='blue', bin_width=bin_width)
    ax_stn_proto_rate.axvspan(STN_start, STN_end, color='green', alpha=0.2)
    ax_stn_proto_rate.set_ylabel('Proto Firing Rate (Hz)')
    ax_stn_proto_rate_twin.set_ylabel('STN Firing Rate (Hz)')
    ax_stn_proto_rate.set_title('Proto and STN Firing Rate (STN active)')
    ax_stn_proto_rate.set_xlim(ax_spec_diff.get_xlim())
    plt.show()

    phase_diff, ave_freq = network_vis_2.calculate_phase_difference('Proto', 'STN', bin_width=bin_width, drop_initial=STN_start)

    print(f"Phase difference between Proto and STN during intervention: {phase_diff} degrees at {ave_freq} Hz")

    return network_sim_1, network_sim_2


def calculate_synchronization_index(membrane_potentials):
    """
    Calculate the synchronization index (chi-squared) from membrane potentials.

    Parameters:
    -----------
    membrane_potentials : numpy.ndarray
        Array of shape (n_neurons, n_timepoints) containing membrane potential traces
        for each neuron over time.

    Returns:
    --------
    chi_squared : float
        The synchronization index, bounded to [0, 1] interval.
    """
    # Number of neurons
    N = membrane_potentials.shape[0]

    # Calculate the LFP-like signal (average membrane potential across neurons)
    lfp = np.mean(membrane_potentials, axis=0)

    # Calculate variance of the LFP signal
    var_lfp = np.var(lfp)

    # Calculate variance of each neuron's membrane potential
    var_individual = np.var(membrane_potentials, axis=1)

    # Calculate the sum of individual variances
    sum_individual_var = np.sum(var_individual)

    # Calculate chi-squared
    chi_squared = (N * var_lfp) / sum_individual_var

    return chi_squared

def load_params_yaml(file_path='params1.yaml', dt=0.5):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML file

    Returns:
        dict: Loaded parameters with dt placeholders replaced
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file not found: {file_path}")

    # Load the YAML file
    with open(file_path, 'r') as file:
        # Load YAML content
        params = yaml.safe_load(file)

    # Replace ${dt} placeholders with actual dt value
    params_str = yaml.dump(params)
    params_str = params_str.replace('${dt}', str(dt))
    params = yaml.safe_load(params_str)

    params_pop = params.get('params_pop', {})
    params_conn_all = params.get('params_conn_all', [])
    return params_pop, params_conn_all

def get_selected_params(params_pop, params_conn_all, selected_pops):
    """
    Get selected population parameters and their connections.

    Args:
        params_pop (dict): Dictionary of population parameters
        params_conn_all (list): List of connection dictionaries
        selected_pops (list): List of population names to include.
                             If None, returns Proto, FSI/FSN, and D2.
    Returns:
        tuple: (selected_pop_params, selected_connections)
    """
    # Create a deep copy to avoid modifying the original
    selected_pop_params = {}
    for pop in selected_pops:
        if 'STN' in pop:
            selected_pop_params['STN'] = params_pop[pop]
        elif pop in params_pop:
            selected_pop_params[pop] = params_pop[pop]
        else:
            raise ValueError(f"Population '{pop}' not found in params_pop.")

    # Only include connections between selected populations
    selected_connections = []
    for conn in params_conn_all:
        source = conn['source']
        target = conn['target']

        # Include connection if both source and target are in selected populations
        if source in selected_pop_params and target in selected_pop_params:
            selected_connections.append(conn.copy())

    return selected_pop_params, selected_connections

