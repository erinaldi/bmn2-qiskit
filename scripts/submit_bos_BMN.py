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
cp $HOME/Code/bmn2-qiskit/scripts/01_bmn2_bosonic_VQE.py ${PJM_O_WORKDIR}/.
echo ${PJM_O_WORKDIR}
"""

def main():
    # loop over parameters
    o_values = ["COBYLA","L-BFGS-B","SLSQP","NELDER-MEAD"]
    L_values = np.array([2, 4], dtype=int)
    g2N_values = np.array([0.2, 0.5, 1.0, 2.0], dtype=float)
    gate_values = [['ry'],['ry','rz']]
    depth_values = np.arange(1,11,1)
    nr = 100  # total number of repetitions
    maxit = 10000 # max number of iterations
    i=0
    for L in L_values:
        for g2N in g2N_values:
            for gate in gate_values:
                for depth in depth_values:
                    for o in o_values:
                        # create data folder
                        l = str(g2N).replace(".","")
                        v = "-".join(gate)
                        folder_name = f"{o}L{L}_l{l}_{v}_d{depth}_nr{nr}_max{maxit}"
                        print(folder_name)
                        i+=1
                        # os.makedirs(folder_name, exist_ok=False)
                        # # move into it
                        # os.chdir(folder_name)
                        # print(
                        #     f"Moving into folder ... {os.path.basename(os.getcwd())}"
                        # )
                        # # create bash submit script
                        # script_name = f"pjrun.sh"
                        # with open(script_name, "w") as f:
                        #     f.write(BLOCK)
                        #     f.write(
                        #         f"python 01_bmn2_bosonic_VQE.py 100 0 {x:.2f} 0.04 {y:.2f} 50 1 15 15000 1000 50\n"
                        #     )

                        # if DO_SUBMIT:
                        #     # submit bash submit script
                        #     print(subprocess.run(["pjsub", script_name], capture_output=True))

                        # # move back out
                        # os.chdir("../")
                        # print(f"... moving back to {os.path.basename(os.getcwd())}")
    print(i)

if __name__ == "__main__":
    main()

    # --L=4 --N=2 --g2N=0.2 --optimizer='NELDER-MEAD' --varform=['ry','rz'] --depth=3 --nrep=10 --maxit=10000