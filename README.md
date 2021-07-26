# Bosonic and Supersymmetric Minimal BMN solved with QISKIT

Use the Variational Quantum Eigensolver in [`QISKIT`](www.qiskit.org) to find the ground state of the quantum BMN matrix model with gauge group SU(2) at different 't Hooft couplings.

## Setup

Install the [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment manager for python and create a new environment for this project:
```shell
conda env create -f environment.yml
```

This will install a new python environment with the dependencies needed to run `QISKIT` and the scripts and notebooks of this repository.
Check that the environment exists
```bash
conda env list
```
and then activate it
```shell
conda activate qiskit-env
```

## Code

The notebooks in [`notebooks`](./notebooks) can be used as a starting point to understand the code.

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

The data produced is saved in the `data` folder using the binary `HDF5` protocol.

*Note* You can run on multiple threads by using i.e.
```bash
export OMP_NUM_THREADS=6; python scripts/02_bmn2_mini_VQE.py --L=2 --N=2 --g2N=0.2 --optimizer='COBYLA' --varform=['ry','rz'] --depth=3 --nrep=10
```
