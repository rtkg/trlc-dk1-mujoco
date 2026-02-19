# trlc-dk1-mujoco

MuJoCo simulation of the TRLC DK-1 Follower robot from its URDF description.

## Installation

Create and activate a local virtual environment, then install the package in editable mode:

```bash
python -m venv .trlc-mujoco
source .trlc-mujoco/bin/activate
pip install -e .
```

## Usage

```bash
python simulate_urdf.py [steps]
```

`steps` is optional and defaults to 10000.
