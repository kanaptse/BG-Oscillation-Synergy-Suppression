import numpy as np
from collections import deque

# Check if CuPy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class NeuronPopulation:
    def __init__(self, params, use_gpu=False, rng=None):
        """
        Initialize the neuron population with option to use GPU acceleration.

        Args:
            params (dict): Dictionary containing neuron parameters
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        # Determine which array module to use
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.rng = rng or (cp.random if self.use_gpu else np.random)

        # Initialize parameters
        self.n_neurons = params['n_neurons']
        self.model_type = params.get('model_type', 'LIF')

        # General neuron parameters (unchanged)
        self.C_m = params.get('C_m', 1.0)
        self.E_L = params.get('E_L', -65.0)
        self.V_th = params.get('V_th', -50.0)
        self.V_reset = params.get('V_reset', -60.0)
        self.dt = params.get('dt', 0.1)
        self.sig = params.get('sig', 0.0)
        # self.I = params.get('I', 0.0)
        I_mean = params.get('I', 0.0)
        I_std = params.get('I_dev', 0.0) * I_mean  # Standard deviation

        # Create the heterogeneous input currents with Gaussian distribution
        self.I = self.rng.standard_normal(self.n_neurons) * I_std + I_mean

        self.k = params.get('k', 1.0)
        random_init = params.get('random_init', False)

        # Initialize neuron states with appropriate array type
        if random_init:
            self.V = self.rng.uniform(self.V_reset, self.V_th, (self.n_neurons,))
        else:
            self.V = self.xp.ones(self.n_neurons) * self.V_reset
        self.spike_vector = self.xp.zeros(self.n_neurons)

        # Record voltage history and spike train
        self.voltage_history = []
        self.spike_train = [[] for _ in range(self.n_neurons)]
        self.synapse_dict = {}

    def create_synapses(self, params_conn, source_neuron_num):
        self.synapse_dict[params_conn['source']] = Synapses(
            self.dt, params_conn, source_neuron_num, self.n_neurons, use_gpu=self.use_gpu, rng=self.rng
        )

    def update_synapses(self, source, spike_train):
        self.synapse_dict[source].update_synaptic_variables(spike_train)

    def update_voltages(self, t, extra_external_input=None):
        # Calculate total synaptic current
        total_synaptic_current = self.xp.zeros(self.n_neurons)
        for source, synapse in self.synapse_dict.items():
            total_synaptic_current += synapse.get_synaptic_current()

        # Generate noise
        noise = self.sig * self.rng.standard_normal(self.n_neurons) / self.xp.sqrt(self.dt / 1000)

        # Handle external input
        if extra_external_input is not None:
            # Convert to GPU if needed
            if self.use_gpu and not isinstance(extra_external_input, cp.ndarray):
                extra_external_input = cp.array(extra_external_input)
            external_input = self.I + extra_external_input
        else:
            external_input = self.I

        # Update voltage based on model
        if self.model_type == 'LIF':
            dV = (-(self.V - self.E_L) + total_synaptic_current + external_input + noise) * (self.dt / self.C_m)
            self.V += dV
            spiking_neurons = self.xp.where(self.V >= self.V_th)[0]
        elif self.model_type == 'QIF':
            dV = (self.k * (self.V - self.E_L) * (
                        self.V - self.V_th) + total_synaptic_current + external_input + noise) * (self.dt / self.C_m)
            self.V += dV
            spiking_neurons = self.xp.where(self.V >= 0)[0]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Record voltage history (convert to CPU if using GPU)
        if self.use_gpu:
            self.voltage_history.append(cp.asnumpy(self.V.copy()))
        else:
            self.voltage_history.append(self.V.copy())

        # Reset spike vector
        self.spike_vector = self.xp.zeros(self.n_neurons)
        self.spike_vector[spiking_neurons] = 1

        # Reset membrane potentials for neurons that spiked
        self.V[spiking_neurons] = self.V_reset

        # Record spikes (need CPU array for indexing list)
        if self.use_gpu:
            spiking_neurons_cpu = cp.asnumpy(spiking_neurons)
            t_float = float(t)
            for neuron in spiking_neurons_cpu:
                self.spike_train[int(neuron)].append(t_float)
        else:
            for neuron in spiking_neurons:
                self.spike_train[neuron].append(t)

    def transfer_to_cpu(self):
        """Transfer all GPU data to CPU memory at once for visualization"""
        if not self.use_gpu:
            return  # No action needed if already using CPU

        # Convert voltage history if stored in GPU
        # Note: voltage_history is already being stored in CPU in update_voltages

        # Convert main neuron state variables
        self.V = self.V.get()

        # Convert synapse data
        for source, synapse in self.synapse_dict.items():
            synapse.synaptic_variable = synapse.synaptic_variable.get()
            synapse.connectivity_matrix = synapse.connectivity_matrix.get()
            # History is already being stored in CPU in update_synaptic_variables

    def get_synapse_history(self, source):
        return np.array(self.synapse_dict[source].history) * self.synapse_dict[source].g

    def get_voltage_history(self):
        """Return the voltage history as a numpy array."""
        return np.array(self.voltage_history)

    def get_spike_train(self):
        """Return the spike train as a numpy array."""
        return self.spike_train


class Synapses:
    def __init__(self, dt, params_conn, source_neuron_num, target_neuron_num, use_gpu=False, rng=None):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.rng = rng or (cp.random if self.use_gpu else np.random)
        self.xp = cp if self.use_gpu else np

        self.dt = dt
        self.tau = params_conn['tau']
        self.g = params_conn['g']

        # Create connectivity matrix with appropriate array type
        self.connectivity_matrix = self.create_connections(target_neuron_num, source_neuron_num,
                                                           params_conn['conn_prob'])

        # Keep delay buffers on CPU as they're queue-based
        self.delay_buffers = self.create_buffers(params_conn['delay'], source_neuron_num)
        self.synaptic_variable = self.xp.zeros(target_neuron_num)
        self.history = []

    def create_buffers(self, delay, source_neuron_num):
        delay_steps = int(delay/self.dt) + 1
        delay_buffers = deque(
            [self.xp.zeros(source_neuron_num)
             for _ in range(delay_steps)],
            maxlen=delay_steps
        )
        return delay_buffers

    def create_connections(self, num_row, num_col, conn_prob):
        return self.rng.uniform(0, 1, (num_row, num_col)) < conn_prob

    def update_synaptic_variables(self, spike_train):
        # Ensure spike_train is on CPU for buffer
        if self.use_gpu and isinstance(spike_train, cp.ndarray):
            self.delay_buffers.append(cp.asnumpy(spike_train))
        else:
            self.delay_buffers.append(spike_train)

        # Get delayed spikes and convert to GPU if needed
        delayed_spikes = self.delay_buffers[0]
        if self.use_gpu:
            delayed_spikes = cp.array(delayed_spikes)

        # Update synaptic variables
        self.synaptic_variable += -self.synaptic_variable * (self.dt / self.tau) + self.xp.dot(self.connectivity_matrix,
                                                                                               delayed_spikes)

        # Store history (convert to CPU if using GPU)
        if self.use_gpu:
            self.history.append(cp.asnumpy(self.synaptic_variable))
        else:
            self.history.append(self.synaptic_variable.copy())

    def get_synaptic_current(self):
        return self.g * self.synaptic_variable
