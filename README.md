# FragDockRL

Fragment-based ligand design with reinforcement learning and tethered docking.

**FragDockRL: A Reinforcement Learning Framework for Fragment-Based Ligand Design via Building Block Assembly and Tethered Docking**
Seung Hwan Hong et al.
Preprint available on bioRxiv.

---

## Overview

FragDockRL is a reinforcement learning framework for fragment-based ligand design.
It combines building block assembly with tethered docking to explore chemical space and optimize ligand binding.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-repo>/FragDockRL.git
cd FragDockRL
```

---

### 2. Create the conda environment

Create the base environment:

```bash
conda env create -f environment.yml
conda activate fragdock
```

This installs core dependencies such as RDKit, pandas, and docking tools.

---

### 3. Install PyTorch (separately)

PyTorch is **not included** in `environment.yml` because installation depends on your system (CPU vs GPU, CUDA version).

The following commands are **tested with PyTorch 2.10.0**.

#### CPU-only

```bash
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### NVIDIA GPU (CUDA 12.8 example)

```bash
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

If your system uses a different CUDA version, refer to the official PyTorch installation guide:
https://pytorch.org/get-started/locally/

---

### 4. Install FragDockRL

```bash
pip install .
```

For development (editable mode):

```bash
pip install -e .
```

---

### 5. Additional dependencies

If not already installed:

```bash
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git
```

---

## Usage

After installation, the following commands are available:

```bash
prepare_core.py
run_fragdock_random.py
run_fragdockrl.py
run_tdock.py
```

---

## Example

Example configurations and test cases are available in:

```
examples/
```

Each target (e.g., CSF1R, VEGFR2) contains:

* prepared receptor structures
* configuration files
* example scripts
* result visualization notebooks

---

## Data

See:

```
data/README.md
```

for information on building blocks and reaction templates.

---

## Notes

* Tested environment:

  * Python 3.12
  * RDKit 2023.09.6
  * PyTorch 2.10.0
* Other PyTorch versions may work but are not officially validated.
* `smina` is installed via conda and is required for docking.
* PyTorch must be installed separately depending on CPU/GPU setup.

---

## License

Licensed under a Custom Non-Commercial License.
Commercial use requires permission.

---

