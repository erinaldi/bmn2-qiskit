import os
import subprocess
import numpy as np

DO_SUBMIT = False
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
cp $HOME/Code/bmn2-qiskit/scripts/02_bmn2_mini_VQE.py ${PJM_O_WORKDIR}/.
echo ${PJM_O_WORKDIR}
"""

def main():
    # loop over parameters
    o_values = ["COBYLA","L-BFGS-B","SLSQP"]#,"NELDER-MEAD"]
    L_values = np.array([2], dtype=int)
    g2N_values = np.array([0.2, 0.5, 1.0, 2.0], dtype=float)
    gate_values = [['ry'],['ry','rz']]
    depth_values = np.arange(1,11,1)
    nr = 100  # total number of repetitions
    maxit = 10000 # max number of iterations

    for L in L_values:
        for g2N in g2N_values:
            for gate in gate_values:
                for depth in depth_values:
                    for o in o_values:
                        # create data folder
                        l = str(g2N).replace(".","")
                        v = "-".join(gate)
                        folder_name = f"{o}_L{L}_l{l}_{v}_d{depth}_nr{nr}_max{maxit}"
                        os.makedirs(folder_name, exist_ok=False)
                        # move into it
                        os.chdir(folder_name)
                        print(
                            f"Moving into folder ... {os.path.basename(os.getcwd())}"
                        )
                        # make data folder needed for the python script
                        os.makedirs("data", exist_ok=True)
                        # create bash submit script
                        script_name = f"pjrun.sh"
                        with open(script_name, "w") as f:
                            f.write(BLOCK)
                            f.write(
                                f"python 02_bmn2_mini_VQE.py --L={L} --N=2 --g2N={g2N} --optimizer={o} --varform={gate} --depth={depth} --nrep={nr} --maxit={maxit} --rngseed={depth}\n"
                            )

                        if DO_SUBMIT:
                            # submit bash submit script
                            print(subprocess.run(["pjsub", script_name], capture_output=True))

                        # move back out
                        os.chdir("../")
                        print(f"... moving back to {os.path.basename(os.getcwd())}")

if __name__ == "__main__":
    main()
