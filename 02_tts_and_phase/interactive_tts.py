import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QHBoxLayout, QLabel, QMessageBox, QRadioButton, QButtonGroup,
    QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt


def load_parameters(yaml_file='params_single_neuron.yaml'):
    """Load parameters from YAML file"""
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuron Model Simulator")
        self.setGeometry(100, 100, 1200, 700)

        # Default parameters
        self.model_type = "lif"  # Default to LIF model
        self.preset = "preset1"  # Default to preset 1


        self.params = load_parameters()
        # Set initial parameters from YAML
        self.apply_preset()

        self.initUI()

    def apply_preset(self):
        """Apply the selected preset parameters"""
        model_params = self.params[f"{self.model_type}_model"][self.preset]

        # Common parameters
        self.I_input = model_params.get('I', 1.14)
        self.inh_amp = model_params.get('inh_amp', 0.9)
        self.inh_period = model_params.get('inh_period', 2.0)
        self.V_th = model_params.get('V_th', 1.0)
        self.V_reset = model_params.get('V_reset', -0.2)
        self.ylim_min = 1
        self.ylim_max = 4

        print(self.I_input, self.inh_amp, self.inh_period, self.V_th, self.V_reset)


        # QIF-specific parameter
        if self.model_type == "qif":
            self.mid_point = model_params.get('mid_point', 0.0)

    def initUI(self):
        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        model_preset_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # Create model selection and preset selection
        model_group = QGroupBox("Model Type")
        model_layout = QVBoxLayout()

        # Radio buttons for model selection
        self.lif_radio = QRadioButton("LIF Model")
        self.qif_radio = QRadioButton("QIF Model")

        # Group the radio buttons
        self.model_group = QButtonGroup()
        self.model_group.addButton(self.lif_radio)
        self.model_group.addButton(self.qif_radio)

        # Set default model
        self.lif_radio.setChecked(True if self.model_type == "lif" else False)
        self.qif_radio.setChecked(True if self.model_type == "qif" else False)

        # Connect signals
        self.lif_radio.toggled.connect(self.on_model_change)
        self.qif_radio.toggled.connect(self.on_model_change)

        model_layout.addWidget(self.lif_radio)
        model_layout.addWidget(self.qif_radio)
        model_group.setLayout(model_layout)

        # Create preset selection
        preset_group = QGroupBox("Parameter Presets")
        preset_layout = QVBoxLayout()

        # Preset buttons
        self.preset1_button = QPushButton("Preset 1")
        self.preset2_button = QPushButton("Preset 2")

        # Style the buttons
        for btn in [self.preset1_button, self.preset2_button]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("QPushButton { background-color: #4A90E2; color: white; border-radius: 8px; }")

        # Highlight the current preset
        if self.preset == "preset1":
            self.preset1_button.setStyleSheet(
                "QPushButton { background-color: #2273D5; color: white; border-radius: 8px; font-weight: bold; }")
        else:
            self.preset2_button.setStyleSheet(
                "QPushButton { background-color: #2273D5; color: white; border-radius: 8px; font-weight: bold; }")

        # Connect signals
        self.preset1_button.clicked.connect(lambda: self.on_preset_change("preset1"))
        self.preset2_button.clicked.connect(lambda: self.on_preset_change("preset2"))

        preset_layout.addWidget(self.preset1_button)
        preset_layout.addWidget(self.preset2_button)
        preset_group.setLayout(preset_layout)

        model_preset_layout.addWidget(model_group)
        model_preset_layout.addWidget(preset_group)

        # Canvas for Matplotlib
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.fig)

        # Parameter controls grid
        parameters_group = QGroupBox("Parameters")
        param_grid = QGridLayout()

        # Input current control
        self.I_input_label = QLabel("Input Current:")
        self.I_input_slider = QSlider(Qt.Horizontal)
        self.I_input_slider.setRange(50, 200)
        self.I_input_slider.setValue(int(self.I_input * 100))
        self.I_input_slider.valueChanged.connect(self.update_I_input_from_slider)

        self.I_input_value = QLineEdit(str(self.I_input))
        self.I_input_value.editingFinished.connect(self.update_I_input_from_value)

        self.I_input_min = QLineEdit("1")
        self.I_input_min.editingFinished.connect(self.update_I_input_range)

        self.I_input_max = QLineEdit("2.0")
        self.I_input_max.editingFinished.connect(self.update_I_input_range)

        # Inhibition amplitude control
        self.inh_amp_label = QLabel("Inh Amplitude:")
        self.inh_amp_slider = QSlider(Qt.Horizontal)
        self.inh_amp_slider.setRange(10, 200)
        self.inh_amp_slider.setValue(int(self.inh_amp * 100))
        self.inh_amp_slider.valueChanged.connect(self.update_inh_amp_from_slider)

        self.inh_amp_value = QLineEdit(str(self.inh_amp))
        self.inh_amp_value.editingFinished.connect(self.update_inh_amp_from_value)

        self.inh_amp_min = QLineEdit("0.1")
        self.inh_amp_min.editingFinished.connect(self.update_inh_amp_range)

        self.inh_amp_max = QLineEdit("2.0")
        self.inh_amp_max.editingFinished.connect(self.update_inh_amp_range)

        # Inhibition period control
        self.inh_period_label = QLabel("Inh Period:")
        self.inh_period_input = QLineEdit(str(self.inh_period))
        self.inh_period_input.editingFinished.connect(self.update_inh_period)

        # Y-limit controls
        self.ylim_label = QLabel("Y-Limits (Second Plot):")
        self.ylim_min_input = QLineEdit(str(self.ylim_min))
        self.ylim_min_input.editingFinished.connect(self.update_ylim)

        self.ylim_max_input = QLineEdit(str(self.ylim_max))
        self.ylim_max_input.editingFinished.connect(self.update_ylim)

        # Add to grid
        param_grid.addWidget(self.I_input_label, 0, 0)
        param_grid.addWidget(QLabel("Min:"), 0, 1)
        param_grid.addWidget(self.I_input_min, 0, 2)
        param_grid.addWidget(QLabel("Max:"), 0, 3)
        param_grid.addWidget(self.I_input_max, 0, 4)
        param_grid.addWidget(QLabel("Value:"), 0, 5)
        param_grid.addWidget(self.I_input_value, 0, 6)
        param_grid.addWidget(self.I_input_slider, 0, 7)

        param_grid.addWidget(self.inh_amp_label, 1, 0)
        param_grid.addWidget(QLabel("Min:"), 1, 1)
        param_grid.addWidget(self.inh_amp_min, 1, 2)
        param_grid.addWidget(QLabel("Max:"), 1, 3)
        param_grid.addWidget(self.inh_amp_max, 1, 4)
        param_grid.addWidget(QLabel("Value:"), 1, 5)
        param_grid.addWidget(self.inh_amp_value, 1, 6)
        param_grid.addWidget(self.inh_amp_slider, 1, 7)

        param_grid.addWidget(self.inh_period_label, 3, 0)
        param_grid.addWidget(self.inh_period_input, 3, 6, 1, 2)

        param_grid.addWidget(self.ylim_label, 4, 0)
        param_grid.addWidget(QLabel("Min:"), 4, 1)
        param_grid.addWidget(self.ylim_min_input, 4, 2)
        param_grid.addWidget(QLabel("Max:"), 4, 3)
        param_grid.addWidget(self.ylim_max_input, 4, 4)

        parameters_group.setLayout(param_grid)

        # Add layouts to main layout
        main_layout.addLayout(model_preset_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(parameters_group)

        self.central_widget.setLayout(main_layout)

        # Initial Plot
        self.update_plot()

    def on_model_change(self):
        """Handle model change"""
        if self.lif_radio.isChecked():
            self.model_type = "lif"
        else:
            self.model_type = "qif"

        self.apply_preset()
        self.update_ui_from_params()
        self.update_plot()

    def on_preset_change(self, preset):
        """Handle preset change"""
        self.preset = preset

        # Update button styling
        if preset == "preset1":
            self.preset1_button.setStyleSheet(
                "QPushButton { background-color: #2273D5; color: white; border-radius: 8px; font-weight: bold; }")
            self.preset2_button.setStyleSheet(
                "QPushButton { background-color: #4A90E2; color: white; border-radius: 8px; }")
        else:
            self.preset1_button.setStyleSheet(
                "QPushButton { background-color: #4A90E2; color: white; border-radius: 8px; }")
            self.preset2_button.setStyleSheet(
                "QPushButton { background-color: #2273D5; color: white; border-radius: 8px; font-weight: bold; }")

        self.apply_preset()
        self.update_ui_from_params()
        self.update_plot()

    def update_ui_from_params(self):
        """Update UI elements to reflect current parameters"""
        # Update sliders and input fields
        self.I_input_slider.setValue(int(self.I_input * 100))
        self.I_input_value.setText(str(self.I_input))

        self.inh_amp_slider.setValue(int(self.inh_amp * 100))
        self.inh_amp_value.setText(str(self.inh_amp))

        self.inh_period_input.setText(str(self.inh_period))
        self.ylim_min_input.setText(str(self.ylim_min))
        self.ylim_max_input.setText(str(self.ylim_max))

    def simulate_lif_neuron(self, ax, I_lif, inh_amp, inh_period, dt=0.01, T=50):
        """Simulate a Leaky Integrate-and-Fire neuron"""
        time = np.arange(0, T, dt)
        inh = inh_amp * (np.sin(2 * np.pi * time / inh_period) > 0).astype(float)
        V_lif = self.V_reset
        V_lif_trace = np.zeros_like(time)
        num_spikes_lif = 0

        for t in range(len(time)):
            dV_lif = (I_lif - V_lif - inh[t]) * dt
            V_lif += dV_lif
            V_lif_trace[t] = V_lif
            if V_lif >= self.V_th:
                V_lif = self.V_reset
                num_spikes_lif += 1

        rescaled_inh = (inh / inh_amp) * (self.V_th - self.V_reset) + self.V_reset
        ax.clear()
        ax.plot(time, V_lif_trace, label="LIF Neuron")
        ax.plot(time, rescaled_inh, 'k--', label="Rescaled Inhibition")
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("Membrane Potential (a.u.)")
        # ax.set_title(f"LIF Spikes: {num_spikes_lif}. Inhibition occurrence: {int(T / inh_period)}.")
        ax.set_title(f"LIF Neuron")
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlim([T - 5 * inh_period, T])
        # ax.legend()

    def simulate_qif_neuron(self, ax, I_qif, inh_amp, inh_period, dt=0.01, T=50):
        """Simulate a Quadratic Integrate-and-Fire neuron"""
        time = np.arange(0, T, dt)
        inh = inh_amp * (np.sin(2 * np.pi * time / inh_period) > 0).astype(float)
        V_qif = self.V_reset
        V_qif_trace = np.zeros_like(time)
        num_spikes_qif = 0

        for t in range(len(time)):
            dV_qif = (I_qif + V_qif**2 - inh[t]) * dt
            V_qif += dV_qif
            V_qif_trace[t] = V_qif
            if V_qif >= self.V_th:
                V_qif = self.V_reset
                num_spikes_qif += 1

        rescaled_inh = (inh / inh_amp) * (self.V_th - self.V_reset) + self.V_reset
        ax.clear()
        ax.plot(time, V_qif_trace, label="QIF Neuron")
        ax.plot(time, rescaled_inh, 'k--', label="Rescaled Inhibition")
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("Membrane Potential (a.u.)")
        # ax.set_title(f"QIF Spikes: {num_spikes_qif}. Inhibition occurrence: {int(T / inh_period)}.")
        ax.set_title(f"QIF Neuron")
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlim([T-5*inh_period, T])
        # ax.legend()


    def simulate_lif_for_tts(self, ax, I_lif, inh_amp, inh_period, dt=0.001, T=10):
        """Simulate Time-To-Spike for LIF neuron"""
        time = np.arange(0, T, dt)
        center_array = np.linspace(0, inh_period, 500)
        V_lif = self.V_reset * np.ones(len(center_array))
        V_lif_trace = np.zeros((len(time), len(center_array)))

        for t in range(len(time)):
            inh = inh_amp * (np.sin(2 * np.pi * (time[t] - center_array) / inh_period + np.pi / 2) > 0)
            dV_lif = (I_lif - V_lif - inh) * dt
            V_lif += dV_lif
            V_lif_trace[t, :] = V_lif

        first_pass_lif = np.argmax(V_lif_trace > self.V_th, axis=0)
        ax.clear()
        ax.plot(center_array, time[first_pass_lif], label="TTS")
        ax.axhline(y=inh_period, color="red", linestyle="--", label="Period of inhibition")
        ax.axvline(x=inh_period / 4, color="green", linestyle="--", label="Inhibition off")
        ax.set_xlabel("Distance from inhibition center (a.u.)")
        ax.set_ylabel("Time (a.u.)")
        ax.set_title("TTS plot for LIF")
        ax.set_ylim(self.ylim_min, self.ylim_max)  # Apply custom ylim
        ax.legend()

    def simulate_qif_for_tts(self, ax, I_qif, inh_amp, inh_period, dt=0.001, T=10):
        time = np.arange(0, T, dt)
        center_array = np.linspace(0, inh_period, 500)
        V_qif = self.V_reset * np.ones(len(center_array))
        V_qif_trace = np.zeros((len(time), len(center_array)))

        for t in range(len(time)):
            inh = inh_amp * (np.sin(2 * np.pi * (time[t] - center_array) / inh_period + np.pi / 2) > 0)
            dV_qif = (I_qif + V_qif**2 - inh) * dt
            V_qif += dV_qif
            V_qif_trace[t, :] = V_qif

        first_pass_qif = np.argmax(V_qif_trace > self.V_th, axis=0)
        ax.clear()
        ax.plot(center_array, time[first_pass_qif], label="TTS")
        ax.axhline(y=inh_period, color="red", linestyle="--", label="Period of inhibition")
        ax.axvline(x=inh_period / 4, color="green", linestyle="--", label="Inhibition off")
        ax.set_xlabel("Distance from inhibition center (a.u.)")
        ax.set_ylabel("Time (a.u.)")
        ax.set_title("TTS plot for QIF")
        ax.set_ylim(self.ylim_min, self.ylim_max)  # Apply custom ylim
        ax.legend()



    def update_plot(self):
        """Update the plots based on current parameters"""
        # Get current parameter values
        I_input = self.I_input
        inh_amp = self.inh_amp
        inh_period = self.inh_period

        # Call the appropriate simulation functions based on the selected model
        if self.model_type == "lif":
            self.simulate_lif_neuron(self.axs[0], I_input, inh_amp, inh_period)
            self.simulate_lif_for_tts(self.axs[1], I_input, inh_amp, inh_period)
        else:  # QIF model
            self.simulate_qif_neuron(self.axs[0], I_input, inh_amp, inh_period)
            self.simulate_qif_for_tts(self.axs[1], I_input, inh_amp, inh_period)

        # Update the plot
        plt.tight_layout()
        self.canvas.draw()

    # Parameter update methods
    def update_I_input_from_slider(self):
        self.I_input = self.I_input_slider.value() / 100.0
        self.I_input_value.setText(str(self.I_input))
        self.update_plot()

    def update_I_input_from_value(self):
        try:
            self.I_input = float(self.I_input_value.text())
            self.I_input_slider.setValue(int(self.I_input * 100))
            self.update_plot()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for Input Current.")

    def update_I_input_range(self):
        try:
            min_val = float(self.I_input_min.text())
            max_val = float(self.I_input_max.text())
            self.I_input_slider.setRange(int(min_val * 100), int(max_val * 100))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for Input Current range.")

    def update_inh_amp_from_slider(self):
        self.inh_amp = self.inh_amp_slider.value() / 100.0
        self.inh_amp_value.setText(str(self.inh_amp))
        self.update_plot()

    def update_inh_amp_from_value(self):
        try:
            self.inh_amp = float(self.inh_amp_value.text())
            self.inh_amp_slider.setValue(int(self.inh_amp * 100))
            self.update_plot()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for Inh Amp.")

    def update_inh_amp_range(self):
        try:
            min_val = float(self.inh_amp_min.text())
            max_val = float(self.inh_amp_max.text())
            self.inh_amp_slider.setRange(int(min_val * 100), int(max_val * 100))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for Inh Amp range.")

    def update_inh_period(self):
        try:
            self.inh_period = float(self.inh_period_input.text())
            self.update_plot()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for Inh Period.")

    def update_ylim(self):
        try:
            self.ylim_min = float(self.ylim_min_input.text())
            self.ylim_max = float(self.ylim_max_input.text())
            if self.ylim_min >= self.ylim_max:
                raise ValueError("Min must be less than Max.")
            self.update_plot()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Invalid Y-Limits: {e}")


# Main application
if __name__ == "__main__":

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())