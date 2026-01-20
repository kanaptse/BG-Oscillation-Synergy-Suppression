import numpy as np
from collections import deque

from neuron_population import NeuronPopulation

# Check if CuPy is available
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class NetworkSimulation:
    def __init__(self, params_pop, params_conn_all, T, dt, use_gpu=True, seed=42):
        """
        Initialize the network simulation with optional GPU support.

        Args:
            params_pop(dict): Parameters for different populations.
            params_conn_all (list): Connection probabilities, strengths and synaptic delays.
            T (float): Total simulation time.
            dt (float): Time step.
            use_gpu (bool): Whether to use GPU acceleration if available.
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.dt = dt
        self.T = T
        self.time_array = self.xp.arange(0, T, dt)
        self.params_pop = params_pop
        self.params_conn_all = params_conn_all
        self.ave_fr = {}

        self.rng_dict = self._create_rngs(seed)

        # Create neuron populations with GPU setting
        self.pop_dict = {pop: NeuronPopulation(params_pop[pop], use_gpu=self.use_gpu, rng=self.rng_dict[pop])
                         for pop in params_pop}
        self.setup_synapses()

    def _create_rngs(self, base_seed=42):
        """
        Creates one RNG per population.
        Chooses cp or np based on self.use_gpu.
        """
        rngs = {}
        for i, pop_name in enumerate(self.params_pop):
            seed = base_seed + i
            if self.use_gpu:
                rngs[pop_name] = cp.random.RandomState(seed)
            else:
                rngs[pop_name] = np.random.default_rng(seed)
        return rngs

    def setup_synapses(self):
        for params_conn in self.params_conn_all:
            source, target = params_conn['source'], params_conn['target']
            source_neuron_num = self.params_pop[source]['n_neurons']
            self.pop_dict[target].create_synapses(params_conn, source_neuron_num)

    def simulate(self, drop_time=2000, print_fr=True, activate_population=None, sin_input_target=None, sin_freq=15, sin_amp=100):
        """
        Run the network simulation.

        Parameters:
            drop_time (float): Time in ms before which spikes are excluded from rate calculations.
            activate_population (dict): Dictionary specifying when a population is active.
                Format: {'pop_name': name_str, 'start': time_ms, 'end': time_ms}
                Example: {'pop_name': STN, 'start': 1000, 'end': 3000} - E population active from 1-3 seconds
            sin_input_target (str): Name of population to receive sinusoidal input.
            sin_freq (float): Frequency of sinusoidal input in Hz.
            sin_amp (float): Amplitude of sinusoidal input.
        """

        # Generate sinusoidal input if needed
        if sin_input_target:
            sin_input = self.generate_sin_input(sin_freq, sin_amp)
        else:
            sin_input = None

        for idx, t in enumerate(self.time_array):
            # First update all synapses (could be done in parallel per target)
            self.update_all_synapses(t, activate_population)

            # Then update all neuron voltages (could be done in parallel)
            self.update_all_voltages(t, idx, sin_input, sin_input_target)

        # Calculate average firing rate for all populations
        for pop, pop_instance in self.pop_dict.items():
            pop_instance.transfer_to_cpu()
            avg_firing_rate = self.calculate_average_firing_rate(pop, start_time=drop_time)
            if print_fr:
                print(f"Average firing rate for {pop} population: {avg_firing_rate:.2f} Hz")

    def update_all_synapses(self, t, activate_population):
        for params_conn in self.params_conn_all:
            source, target = params_conn['source'], params_conn['target']
            current_spike_vector = self.pop_dict[source].spike_vector
            if activate_population and source == activate_population['pop_name']:
                if t < activate_population['start'] or t > activate_population['end']:
                    current_spike_vector = self.xp.zeros_like(current_spike_vector)
            self.pop_dict[target].update_synapses(source, current_spike_vector)

    def update_all_voltages(self, t, idx, sin_input, sin_input_target):
        for pop, pop_class in self.pop_dict.items():
            external_input = None
            if sin_input is not None and pop == sin_input_target:
                external_input = sin_input[idx]
            pop_class.update_voltages(t, external_input)

    def generate_sin_input(self, freq, amp):
        """Generate a sinusoidal input current."""
        period = 1000 / freq
        # Use appropriate array library (numpy/cupy)
        time_array = self.xp.asnumpy(self.time_array) if self.use_gpu else self.time_array
        input_signal = amp * self.xp.sin(2 * self.xp.pi * time_array / period)
        return -input_signal * (input_signal > 0)

    def get_spike_trains(self):
        return {pop: pop_instance.spike_train for pop, pop_instance in self.pop_dict.items()}

    def get_voltage_history(self):
        return {pop: pop_instance.get_voltage_history() for pop, pop_instance in self.pop_dict.items()}

    def get_synpse_history(self, source, target):
        return self.pop_dict[target].get_synapse_history(source)

    def calculate_average_firing_rate(self, pop, start_time=2000.0):
        """Calculate average firing rate, filtering spikes before start_time."""
        # Simplified CPU-only implementation
        pop_instance = self.pop_dict[pop]
        spike_trains = pop_instance.spike_train

        all_spike_arrays = []
        for st in spike_trains:
            if len(st) > 0:
                all_spike_arrays.append(np.array(st))

        if all_spike_arrays:
            all_spikes = np.concatenate(all_spike_arrays)
            filtered_spike_trains = all_spikes[all_spikes >= start_time]
        else:
            filtered_spike_trains = np.array([])

        # Calculate the total number of spikes after the start time
        total_spikes = len(filtered_spike_trains) / len(spike_trains)

        # Calculate the effective time window in seconds
        effective_time = (self.T - start_time) / 1000.0

        # Calculate the average firing rate
        average_firing_rate = total_spikes / effective_time

        self.ave_fr[pop] = average_firing_rate

        return average_firing_rate

    def get_average_firing_rate(self, pop):
        """Get the average firing rates for a specific population."""
        if pop in self.ave_fr:
            return self.ave_fr[pop]
        else:
            raise ValueError(f"Average firing rate for population '{pop}' has not been calculated yet.")
