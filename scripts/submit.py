import os
import subprocess
import numpy as np

DO_SUBMIT = False
INTERVALS = 10
BLOCK = """#!/bin/bash
#PJM -L rscgrp=batch
#PJM -L vnode=1
#PJM -L vnode-core=4
#PJM -L elapse=10:00:00
#PJM -g Q21550
#PJM -o output.log
#PJM -j
source $HOME/.bashrc
conda activate qiskit-env
export OMP_NUM_THREADS=4
# copy script in case you want to modify it
cp $HOME/Code/bmn2-qiskit/scripts/01_bmn2_bosonic_VQE.py ${PJM_O_WORKDIR}/.
echo ${PJM_O_WORKDIR}
"""

def main():
    # loop over parameters
    Chi = 1.6
    delta_omega = 2.3
    chi_values = np.linspace(Chi - 0.6, Chi + 0.4, INTERVALS)
    do_values = np.linspace(delta_omega - 1.0, delta_omega + 0.5, INTERVALS)

    for x in chi_values:
        for y in do_values:
            # create data folder
            folder_name = f"run_{x:.2f}-{y:.2f}"
            os.makedirs(folder_name, exist_ok=False)
            # move into it
            os.chdir(folder_name)
            print(
                f"Moving into folder for Chi={x:.2f} and domega={y:.2f} ... {os.path.basename(os.getcwd())}"
            )
            # create bash submit script
            script_name = f"pjrun.sh"
            with open(script_name, "w") as f:
                f.write(BLOCK)
                f.write(
                    f"python 01_bmn2_bosonic_VQE.py 100 0 {x:.2f} 0.04 {y:.2f} 50 1 15 15000 1000 50\n"
                )

            if DO_SUBMIT:
                # submit bash submit script
                print(subprocess.run(["pjsub", script_name], capture_output=True))

            # move back out
            os.chdir("../")
            print(f"... moving back to {os.path.basename(os.getcwd())}")


if __name__ == "__main__":
    main()