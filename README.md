# Bosonic and Supersymmetric BMN models solved with QISKIT

Use the Variational Quantum Eigensolver in [`QISKIT`](www.qiskit.org) to find the ground state of the quantum 2-matrix model with gauge group SU(2) at different 't Hooft couplings.
We consider a purely bosonic model and a supersymmetric model (*minimal BMN*).
Results are reported in the publication [Rinaldi et al. (2021)](www.arxiv.org/abs/2108.00000).
Consider the citation in [Cite](#cite).

# Code

## Installation

Install the [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment manager for python and create a new environment for this project:
```bash
conda env create -f environment.yml
```

This will install a new python environment with the dependencies needed to run `QISKIT` and the scripts and notebooks of this repository.
Check that the environment exists
```bash
conda env list
```
and then activate it
```bash
conda activate qiskit-env
```

## Scripts

For making plots and datafiles, you can also use the python scripts in the [`scripts`](./scripts) folder.

The file `qiskit.ini` contains `QISKIT` setting. You can copy it to your default location usually found in `${HOME}/.qiskit/settings.conf`.

Running main scripts with `-h` will let you see the command line options, e.g.
```bash
python script/01_bmn2_bosonic_VQE.py -h
```
or
```bash
python script/02_bmn2_mini_VQE.py -h
```

The data produced is saved in the `data` folder using the binary `HDF5` protocol (with one command line flag you can save in `pickle` compressed format).

**Note**: You can run on multiple threads by using i.e.
```bash
export OMP_NUM_THREADS=6; python scripts/02_bmn2_mini_VQE.py --L=2 --N=2 --g2N=0.2 --optimizer='COBYLA' --varform=['ry','rz'] --depth=3 --nrep=10
```

The [`scripts/hokusai`](./scripts/hokusai) folder is for scripts managing the submission of jobs on the RIKEN Hokusai cluster in Wako, Japan.

There are also utility scripts for making plots.

## Notebooks

The notebooks in [`notebooks`](./notebooks) can be used as a starting point to understand the code.

* The notebook [QISKIT_HarmonicOscillator_VQE.ipynb](./notebooks/QISKIT_HarmonicOscillator_VQE.ipynb) gives an introduction to the VQE for a simple harmonic oscillator in the coordinate basis. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jk1cHXoSllGggh5oLxJnpelNgHlO_qNh?usp=sharing)
* The notebook [QISKIT_bosonic_matrices_VQE.ipynb](./notebooks/QISKIT_bosonic_matrices_VQE.ipynb) gives an introduction to the VQE for the bosonic quantum matrix model. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zOAc1BZb90KcKPtiIJ20q-HhbCGx3Drs?usp=sharing)
* The notebook [QISKIT_susy_matrices_VQE.ipynb](./notebooks/QISKIT_susy_matrices_VQE.ipynb) gives an introduction to the VQE for the supersymmetric quantum matrix model. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q81_9anzZGBK80qdYT0sUxDoEGsW6Qj9?usp=sharing)

# Cite

If you use this code (or parts of it), please consider citing our paper:
```
BIBTEX CITATION HERE
```