# Model Features Determine Whether Coupled Basal Ganglia Subnetworks Synergize or Suppress Oscillations

This repository contains the code accompanying the paper:

**Tse, K.N., Ermentrout, G.B., & Rubin, J.E.** "Model Features Determine Whether Coupled Basal Ganglia Subnetworks Synergize or Suppress Oscillations"

Department of Mathematics, University of Pittsburgh

## Overview

This work addresses contradictory findings in computational studies of Parkinson's disease regarding the role of the subthalamic nucleus (STN) in pathological beta oscillations. We demonstrate that the choice between leaky integrate-and-fire (LIF) and quadratic integrate-and-fire (QIF) neurons fundamentally impacts the phase relationship between STN and external globus pallidus prototypical (Proto) neuron populations:

- **QIF STN neurons** establish **in-phase** coupling with Proto neurons, which **enhances** beta oscillation amplitude
- **LIF STN neurons** develop **anti-phase** relationships, which **suppresses** beta power

These findings reveal that the mathematical structure underlying spike generation, rather than other biophysical details, determines whether the subthalamopallidal loop acts as a beta amplifier or suppressor.

## Repository Structure

```
.
├── 01_rate_model/
│   └── figures_in_paper.ipynb      # Rate model simulations (Figures 2-3)
├── 02_tts_and_phase/
│   ├── figures_in_paper.ipynb      # Time-to-spike analysis (Figures 5, 7-9)
│   ├── interactive_tts.py          # Interactive TTS visualization
│   └── interactive_phase_plane_qif_lif.py  # Phase plane visualization
├── 03_spiking_networks/
│   ├── figures_in_paper.ipynb      # Spiking network simulations (Figures 10-13)
│   ├── neuron_population.py        # Neuron population classes
│   ├── network_simulation.py       # Network simulation framework
│   └── utils.py                    # Utility functions
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/Model-Features-Determine-Whether-Coupled-Basal-Ganglia-Subnetworks-Synergize-or-Suppress-Oscillations.git
cd Model-Features-Determine-Whether-Coupled-Basal-Ganglia-Subnetworks-Synergize-or-Suppress-Oscillations

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Each directory contains Jupyter notebooks (`figures_in_paper.ipynb`) that reproduce the figures from the paper:

### 1. Rate Model Analysis (`01_rate_model/`)
Demonstrates how the phase relationship between STN and Proto populations directly influences beta oscillation amplitude using a simplified firing rate model.

### 2. Time-to-Spike and Phase Analysis (`02_tts_and_phase/`)
Contains single neuron analysis under periodic inhibition, using time-to-spike (TTS) functions to prove that LIF and QIF neurons exhibit fundamentally different phase preferences.

Interactive scripts are also provided:
```bash
python 02_tts_and_phase/interactive_tts.py
python 02_tts_and_phase/interactive_phase_plane_qif_lif.py
```

### 3. Spiking Network Simulations (`03_spiking_networks/`)
Full multi-population basal ganglia network simulations incorporating both pallidostriatal and subthalamopallidal circuits.

## Key Findings

1. **Phase relationships control oscillation amplitude**: In-phase STN-Proto coupling enhances beta power, while anti-phase coupling suppresses it.

2. **Neuron model determines phase**: QIF neurons (and EIF neurons) naturally establish in-phase relationships with periodic inhibition, while LIF neurons prefer anti-phase locking.

3. **Reconciliation of contradictory literature**: The divergent predictions from recent computational studies likely stem from different neuronal model implementations rather than fundamental disagreements about circuit mechanisms.

## Model Details

The network architecture includes four neuronal populations:
- **STN** (Subthalamic Nucleus) - excitatory
- **Proto** (GPe Prototypical neurons) - inhibitory
- **FSI** (Fast-Spiking Interneurons) - inhibitory
- **D2** (D2-type Spiny Projection Neurons) - inhibitory

Two feedback loops are modeled:
- **Subthalamopallidal loop**: STN ↔ Proto
- **Pallidostriatal loop**: D2 → Proto → FSI → D2

## Dependencies

- Python 3.8+
- jitcdde
- matplotlib
- numpy
- pandas
- scipy
- sympy/symengine
- PyQt5 (for interactive visualizations)
- tqdm

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tse2025model,
  title={Model Features Determine Whether Coupled Basal Ganglia Subnetworks Synergize or Suppress Oscillations},
  author={Tse, Ka Nap and Ermentrout, G. Bard and Rubin, Jonathan E.},
  journal={},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
- Ka Nap Tse (corresponding author): kat154@pitt.edu
- G. Bard Ermentrout: phase@pitt.edu
- Jonathan E. Rubin: jonrubin@pitt.edu

## License

[Specify license here]
