import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QSlider, QLabel, QWidget, QPushButton, QComboBox,
                            QFrame, QRadioButton, QButtonGroup, QLineEdit, QGridLayout,
                            QGroupBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QDoubleValidator
import yaml

def load_parameters(yaml_file='params_single_neuron.yaml'):
    """Load parameters from YAML file"""
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

class NeuronModelVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        parameters = load_parameters()

        self.lif_params_set1 = parameters['lif_model']['preset1']
        self.lif_params_set2 = parameters['lif_model']['preset2']
        self.qif_params_set1 = parameters['qif_model']['preset1']
        self.qif_params_set2 = parameters['qif_model']['preset2']

        # Set initial parameters to first set of LIF
        self.lif_params = self.lif_params_set1
        self.qif_params = self.qif_params_set1

        # Current model parameters (start with LIF)
        self.current_params = self.lif_params.copy()

        self.animation_speed = 1  # Default speed multiplier

        # Simulation data
        self.prepare_simulation_data()


        # Setup UI components
        self.init_ui()

        # Initial plot update
        self.update_plot()

    def prepare_simulation_data(self):
        """Run simulation and prepare data for visualization"""
        # Extract parameters
        model = self.current_params["model"]
        I = self.current_params["I"]
        inh_amp = self.current_params["inh_amp"]
        mid_point = self.current_params.get("mid_point", 0.0)
        v_rest = self.current_params.get("v_rest", 0.0)
        V_th = self.current_params["V_th"]
        V_reset = self.current_params["V_reset"]
        dt = self.current_params["dt"]
        T = self.current_params["T"]
        inh_period = self.current_params["inh_period"]

        # Initialize time and arrays
        self.time = np.arange(0, T, dt)
        self.inh = inh_amp * (np.sin(2 * np.pi * self.time / inh_period) > 0).astype(float)
        self.v_trace = np.zeros_like(self.time)
        self.v_trace[0] = v_rest if model == "LIF" else 0
        self.v_dot_trace = np.zeros_like(self.time)

        # Simulate the neuron
        for t in range(1, len(self.time)):
            if model == "LIF":
                v_dot = -(self.v_trace[t-1] - v_rest) + I - self.inh[t-1]
            elif model == "QIF":
                v_dot = I + (self.v_trace[t-1] - mid_point)**2 - self.inh[t-1]

            self.v_dot_trace[t-1] = v_dot
            self.v_trace[t] = self.v_trace[t-1] + v_dot * dt

            if self.v_trace[t] >= V_th:
                self.v_trace[t] = V_reset

        # Calculate v_dot for the last timestep
        if model == "LIF":
            self.v_dot_trace[-1] = -(self.v_trace[-1] - v_rest) + I - self.inh[-1]
        elif model == "QIF":
            self.v_dot_trace[-1] = I + (self.v_trace[-1] - mid_point)**2 - self.inh[-1]

        # Detect transitions in inhibition
        inh_diff = np.diff(self.inh)
        self.inh_on_indices = np.where(inh_diff > 0)[0]
        self.inh_off_indices = np.where(inh_diff < 0)[0]

        # Phase plane data
        self.v = np.linspace(V_reset-0.1, V_th+0.1, 1000)
        if model == "LIF":
            self.v_dot_inh_off = -(self.v - v_rest) + I
            self.v_dot_inh_on = -(self.v - v_rest) + I - inh_amp
        elif model == "QIF":
            self.v_dot_inh_off = I + (self.v - mid_point)**2
            self.v_dot_inh_on = I + (self.v - mid_point)**2 - inh_amp

        # Store model-specific parameters for later use
        self.model = model
        self.I = I
        self.mid_point = mid_point
        self.v_rest = v_rest
        self.inh_amp = inh_amp

        # Set total number of frames
        self.num_frames = len(self.time)
        self.current_frame = 0


    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Neuron Model Visualization')
        self.setGeometry(100, 100, 1300, 700)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create top controls panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Left side: Model selection
        model_group_box = QGroupBox("Model Type")
        model_layout = QVBoxLayout(model_group_box)

        model_group = QButtonGroup(self)

        lif_radio = QRadioButton("LIF Model")
        qif_radio = QRadioButton("QIF Model")

        if self.current_params["model"] == "LIF":
            lif_radio.setChecked(True)
        else:
            qif_radio.setChecked(True)

        model_group.addButton(lif_radio, 1)
        model_group.addButton(qif_radio, 2)
        model_group.buttonClicked.connect(self.change_model)

        model_layout.addWidget(lif_radio)
        model_layout.addWidget(qif_radio)

        control_layout.addWidget(model_group_box)

        # Middle: Parameter Presets
        presets_group_box = QGroupBox("Parameter Presets")
        presets_layout = QVBoxLayout(presets_group_box)

        self.preset1_button = QPushButton("Preset 1")
        self.preset1_button.setCheckable(True)
        self.preset1_button.setChecked(True)
        self.preset1_button.clicked.connect(self.use_preset1)

        self.preset2_button = QPushButton("Preset 2")
        self.preset2_button.setCheckable(True)
        self.preset2_button.clicked.connect(self.use_preset2)

        # Add buttons to button group to make them exclusive
        preset_group = QButtonGroup(self)
        preset_group.addButton(self.preset1_button)
        preset_group.addButton(self.preset2_button)
        preset_group.setExclusive(True)

        presets_layout.addWidget(self.preset1_button)
        presets_layout.addWidget(self.preset2_button)

        control_layout.addWidget(presets_group_box)

        # Right side: Parameter inputs
        params_group_box = QGroupBox("Custom Parameters")
        params_layout = QGridLayout(params_group_box)

        # Input current (I) text box
        params_layout.addWidget(QLabel("Input current (I):"), 0, 0)
        self.i_text = QLineEdit(str(self.current_params["I"]))
        self.i_text.setValidator(QDoubleValidator())
        self.i_text.textChanged.connect(self.update_params)
        params_layout.addWidget(self.i_text, 0, 1)

        # Inhibition amplitude text box
        params_layout.addWidget(QLabel("Inhibition amplitude:"), 1, 0)
        self.inh_amp_text = QLineEdit(str(self.current_params["inh_amp"]))
        self.inh_amp_text.setValidator(QDoubleValidator())
        self.inh_amp_text.textChanged.connect(self.update_params)
        params_layout.addWidget(self.inh_amp_text, 1, 1)

        # Apply button for parameter changes
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_param_changes)
        params_layout.addWidget(self.apply_button, 2, 0, 1, 2)

        # Add parameter inputs to control layout
        control_layout.addWidget(params_group_box)

        # Add control panel to main layout
        main_layout.addWidget(control_panel)

        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Set up the axes - side by side layout
        self.axes_phase = self.fig.add_subplot(121)  # Phase plane
        self.axes_time = self.fig.add_subplot(122)   # Time series

        # Create time slider and label
        time_control_layout = QHBoxLayout()

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.num_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(QSlider.TicksBelow)
        self.time_slider.setTickInterval(self.num_frames // 10)
        self.time_slider.valueChanged.connect(self.update_time)

        self.time_label = QLabel(f"Time: {self.time[0]:.2f} ms")
        time_control_layout.addWidget(QLabel("Time:"))
        time_control_layout.addWidget(self.time_slider)
        time_control_layout.addWidget(self.time_label)

        # Add animation controls
        animation_control_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_play)
        animation_control_layout.addWidget(self.play_button)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_animation)
        animation_control_layout.addWidget(self.reset_button)

        # Speed control dropdown
        speed_label = QLabel("Speed:")
        animation_control_layout.addWidget(speed_label)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x", "32x"])
        self.speed_combo.setCurrentIndex(2)  # Default to 1x
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        animation_control_layout.addWidget(self.speed_combo)

        # Add layouts to main layout
        main_layout.addLayout(time_control_layout)
        main_layout.addLayout(animation_control_layout)

        # Set the central widget
        self.setCentralWidget(main_widget)

        # Timer for animation
        self.is_playing = False
        self.timer_id = None

    def update_params(self):
        """Handler for parameter text input changes"""
        # Uncheck both preset buttons when user modifies values manually
        self.preset1_button.setChecked(False)
        self.preset2_button.setChecked(False)

    def apply_param_changes(self):
        """Apply parameter changes when apply button is clicked"""
        # Pause animation if running
        was_playing = False
        if self.is_playing:
            was_playing = True
            self.play_button.setChecked(False)
            self.toggle_play(False)

        # Get current frame position as percentage
        position = self.current_frame / self.num_frames if self.num_frames > 0 else 0

        # Update parameters from text inputs
        try:
            # Get the current I value
            i_value = float(self.i_text.text())
            self.current_params["I"] = i_value

            # Get the current inhibition amplitude
            inh_value = float(self.inh_amp_text.text())
            self.current_params["inh_amp"] = inh_value

            # Update the respective parameter set based on the current model
            if self.current_params["model"] == "LIF":
                self.lif_params["I"] = i_value
                self.lif_params["inh_amp"] = inh_value
            else:
                self.qif_params["I"] = i_value
                self.qif_params["inh_amp"] = inh_value

            # Update simulation with new parameters
            self.prepare_simulation_data()

            # Restore frame position
            self.current_frame = int(position * self.num_frames)
            self.time_slider.setMaximum(self.num_frames - 1)
            self.time_slider.setValue(self.current_frame)

            # Update plot
            self.update_plot()

            # Resume animation if it was playing
            if was_playing:
                self.play_button.setChecked(True)
                self.toggle_play(True)
        except ValueError:
            # Invalid input - ignore
            pass

    def use_preset1(self):
        """Switch to parameter set 1 for current model"""
        # Save current frame position
        position = self.current_frame / self.num_frames if self.num_frames > 0 else 0

        # Update current params based on model type
        if self.current_params["model"] == "LIF":
            self.lif_params = self.lif_params_set1.copy()
            self.current_params = self.lif_params.copy()
        else:
            self.qif_params = self.qif_params_set1.copy()
            self.current_params = self.qif_params.copy()

        # Update text inputs
        self.i_text.setText(str(self.current_params["I"]))
        self.inh_amp_text.setText(str(self.current_params["inh_amp"]))

        # Recalculate simulation data
        self.prepare_simulation_data()

        # Update slider
        self.time_slider.setMaximum(self.num_frames - 1)
        self.current_frame = int(position * self.num_frames)
        self.time_slider.setValue(self.current_frame)

        # Update plot
        self.update_plot()

    def use_preset2(self):
        """Switch to parameter set 2 for current model"""
        # Save current frame position
        position = self.current_frame / self.num_frames if self.num_frames > 0 else 0

        # Update current params based on model type
        if self.current_params["model"] == "LIF":
            self.lif_params = self.lif_params_set2.copy()
            self.current_params = self.lif_params.copy()
        else:
            self.qif_params = self.qif_params_set2.copy()
            self.current_params = self.qif_params.copy()

        # Update text inputs
        self.i_text.setText(str(self.current_params["I"]))
        self.inh_amp_text.setText(str(self.current_params["inh_amp"]))

        # Recalculate simulation data
        self.prepare_simulation_data()

        # Update slider
        self.time_slider.setMaximum(self.num_frames - 1)
        self.current_frame = int(position * self.num_frames)
        self.time_slider.setValue(self.current_frame)

        # Update plot
        self.update_plot()

    def change_model(self, button):
        """Change between LIF and QIF models"""
        # Save current frame position
        position = self.current_frame / self.num_frames if self.num_frames > 0 else 0

        # Check which preset is selected
        preset1_selected = self.preset1_button.isChecked()

        if button.text() == "LIF Model":
            if preset1_selected:
                self.current_params = self.lif_params_set1.copy()
                self.lif_params = self.lif_params_set1.copy()
            else:
                self.current_params = self.lif_params_set2.copy()
                self.lif_params = self.lif_params_set2.copy()
        else:
            if preset1_selected:
                self.current_params = self.qif_params_set1.copy()
                self.qif_params = self.qif_params_set1.copy()
            else:
                self.current_params = self.qif_params_set2.copy()
                self.qif_params = self.qif_params_set2.copy()

        # Update text inputs with new values
        self.i_text.setText(str(self.current_params["I"]))
        self.inh_amp_text.setText(str(self.current_params["inh_amp"]))

        # Recalculate simulation data
        self.prepare_simulation_data()

        # Update slider
        self.time_slider.setMaximum(self.num_frames - 1)
        self.current_frame = int(position * self.num_frames)
        self.time_slider.setValue(self.current_frame)

        # Update plot
        self.update_plot()

    @pyqtSlot(int)
    def update_time(self, frame):
        """Handle slider value change"""
        self.current_frame = frame

        # Update time label
        self.time_label.setText(f"Time: {self.time[frame]:.2f} ms")

        # Update plot
        self.update_plot()

    def update_plot(self):
        """Update all plots for the current frame"""
        # Clear existing plots
        self.axes_phase.clear()
        self.axes_time.clear()

        # Current frame index
        frame = self.current_frame

        # Extract parameters for readability
        V_th = self.current_params["V_th"]
        V_reset = self.current_params["V_reset"]
        model = self.model  # Use the stored model type

        # --- Phase Plane Plot ---
        # Plot nullcline
        self.axes_phase.plot(self.v, np.zeros_like(self.v), 'k-', lw=0.5)

        # Plot dv/dt curve based on current inhibition state
        if self.inh[frame] > 0:
            # Inhibition is ON
            self.axes_phase.plot(self.v, self.v_dot_inh_on, 'r-', label='dv/dt (Inh ON)')
        else:
            # Inhibition is OFF
            self.axes_phase.plot(self.v, self.v_dot_inh_off, 'b-', label='dv/dt (Inh OFF)')

        # Plot current state as green dot ONLY on the x-axis (not on the dv/dt curve)
        self.axes_phase.plot(self.v_trace[frame], 0, 'go', ms=8, label='Current State')

        # Plot inhibition transition events that have occurred so far ON THE X-AXIS
        for i, idx in enumerate(self.inh_on_indices):
            if idx <= frame:
                alpha = 0.3 + 0.7 * (1 - (frame - idx) / frame) if frame > 0 else 1.0
                alpha = max(0.3, min(1.0, alpha))
                self.axes_phase.scatter(self.v_trace[idx], 0, color='r', alpha=alpha)

        for i, idx in enumerate(self.inh_off_indices):
            if idx <= frame:
                alpha = 0.3 + 0.7 * (1 - (frame - idx) / frame) if frame > 0 else 1.0
                alpha = max(0.3, min(1.0, alpha))
                self.axes_phase.scatter(self.v_trace[idx], 0, color='b', alpha=alpha)

        # Plot thresholds
        self.axes_phase.axvline(x=V_th, color='k', linestyle='--', alpha=0.5, label='Threshold')
        self.axes_phase.axvline(x=V_reset, color='k', linestyle=':', alpha=0.5, label='Reset')

        self.axes_phase.set_xlabel('Membrane Potential (v)')
        self.axes_phase.set_ylabel('dv/dt')
        self.axes_phase.set_title(f'{model} Neuron Phase Plane')
        self.axes_phase.legend(loc='upper right')

        # Calculate phase plane data for y-axis limits
        if model == "LIF":
            y_min = min(np.min(self.v_dot_inh_off), np.min(self.v_dot_inh_on)) * 1.1
            y_max = max(np.max(self.v_dot_inh_off), np.max(self.v_dot_inh_on)) * 1.1
        elif model == "QIF":
            # For QIF, the phase plane can have larger ranges, so we calculate more precisely
            v_range = np.linspace(V_reset-0.1, V_th+0.1, 1000)
            I = self.I
            mid_point = self.mid_point
            inh_amp = self.inh_amp
            inh_off_values = I + (v_range - mid_point)**2
            inh_on_values = I + (v_range - mid_point)**2 - inh_amp
            y_min = min(np.min(inh_off_values), np.min(inh_on_values)) * 1.1
            y_max = max(np.max(inh_off_values), np.max(inh_on_values)) * 1.1

        # Ensure y_min is always negative to include x-axis
        y_min = min(y_min, -0.1)

        # Set the y-axis limits for phase plane
        self.axes_phase.set_ylim([y_min, y_max])

        # --- Time Series Plot ---
        # Plot time series up to current frame
        self.axes_time.plot(self.time[:frame+1], self.v_trace[:frame+1], 'g-', label='Membrane Potential')

        # Plot inhibition as a dashed line (scaled to be visible against voltage)
        scale_factor = (V_th - V_reset) / self.inh_amp
        self.axes_time.plot(self.time[:frame+1], self.inh[:frame+1] * scale_factor + V_reset,
                           'k--', label='Inhibition')

        # Plot inhibition transition events that have occurred so far
        for i, idx in enumerate(self.inh_on_indices):
            if idx <= frame:
                alpha = 0.3 + 0.7 * (1 - (frame - idx) / frame) if frame > 0 else 1.0
                alpha = max(0.3, min(1.0, alpha))
                self.axes_time.scatter(self.time[idx], self.v_trace[idx], color='r', alpha=alpha)
                # No vertical blue/red lines as requested

        for i, idx in enumerate(self.inh_off_indices):
            if idx <= frame:
                alpha = 0.3 + 0.7 * (1 - (frame - idx) / frame) if frame > 0 else 1.0
                alpha = max(0.3, min(1.0, alpha))
                self.axes_time.scatter(self.time[idx], self.v_trace[idx], color='b', alpha=alpha)
                # No vertical blue/red lines as requested

        # Add current time indicator
        self.axes_time.axvline(x=self.time[frame], color='g', linestyle='-', alpha=0.7)

        # No threshold or reset lines as requested

        self.axes_time.set_xlabel('Time [ms]')
        self.axes_time.set_ylabel('Membrane Potential / Inhibition')
        self.axes_time.set_title(f'{model} Neuron Dynamics')
        self.axes_time.legend(loc='upper right')
        # self.axes_time.grid(True, linestyle='--', alpha=0.6)
        self.axes_time.set_xlim([0, self.time[-1]])
        # Set the y-limits based on the data instead of threshold/reset values
        v_min = min(np.min(self.v_trace), V_reset-0.1)
        v_max = max(np.max(self.v_trace), V_th+0.3)
        self.axes_time.set_ylim([v_min, v_max])

        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def change_speed(self, index):
        """Change animation speed based on dropdown selection"""
        speed_options = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        self.animation_speed = speed_options[index]

        # If animation is playing, restart timer with new speed
        if self.is_playing:
            self.killTimer(self.timer_id)
            # Timer interval in ms (smaller = faster)
            interval = int(50 / self.animation_speed)
            self.timer_id = self.startTimer(max(1, interval))

    def toggle_play(self, checked):
        """Handle play/pause button toggle"""
        if checked:
            self.play_button.setText("Pause")
            self.is_playing = True
            # Start timer with appropriate interval
            # Use a faster interval for the higher resolution simulation
            interval = int(20 / self.animation_speed)  # Faster base interval (20ms instead of 50ms)
            self.timer_id = self.startTimer(max(1, interval))
        else:
            self.play_button.setText("Play")
            self.is_playing = False
            if self.timer_id is not None:
                self.killTimer(self.timer_id)
                self.timer_id = None

    def reset_animation(self):
        """Reset animation to the beginning"""
        self.current_frame = 0
        self.time_slider.setValue(0)
        # If playing, stop
        if self.is_playing:
            self.play_button.setChecked(False)
            self.toggle_play(False)

    def timerEvent(self, event):
        """Timer event handler for animation playback"""
        # Calculate frame increment based on speed
        # With higher resolution, we need larger increments to maintain visual speed
        base_increment = max(1, int(self.animation_speed * 50))  # Multiplied by 10 for 0.1ms resolution

        # Check if we've reached the end
        if self.current_frame + base_increment < self.num_frames:
            self.current_frame += base_increment
            self.time_slider.setValue(self.current_frame)
        else:
            # Stop at the end
            self.time_slider.setValue(self.num_frames - 1)
            self.play_button.setChecked(False)
            self.toggle_play(False)

# Run the application if executed directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuronModelVisualizer()
    window.show()
    sys.exit(app.exec_())