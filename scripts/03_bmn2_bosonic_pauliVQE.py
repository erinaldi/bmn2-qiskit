# %%
import time
import fire
import os
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse import identity
from qiskit import Aer
from qiskit.opflow import MatrixOp, ListOp, TensoredOp, SummedOp, I
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms import NumPyEigensolver, VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B, NELDER_MEAD
from qiskit.utils import algorithm_globals, QuantumInstance


# %%
def build_operators(L: int, N: int) -> list:
    """Generate all the annihilation operators needed to build the hamiltonian

    Args:
        L (int): the cutoff of the single site Fock space
        N (int): the number of colors of gauge group SU(N) *it can not be different from 2 right now*

    Returns:
        list: a list of annihilation operators, length=N_bos, using PauliOp representation.
    """
    # These are low-D (single boson) so we can use the MatrixOp for convenience
    # The annihilation operator for the single boson
    a_b = MatrixOp(
        diags(np.sqrt(np.linspace(1, L - 1, L - 1)), offsets=1)
    ).to_pauli_op()
    # The identity operator of the Fock space of a single boson
    i_b = MatrixOp(identity(L)).to_pauli_op()
    # Now the memory starts growing! Use the PauliOp structure
    # Bosonic Hilbert space
    N_bos = int(2 * (N ** 2 - 1))  # number of boson sites -> fixed for mini-BMN 2
    a_b_list = []  # this will contain a1...a6 as a list of ListOp
    for i in np.arange(0, N_bos):  # loop over all operators
        operator_list = [i_b] * N_bos  # only the identity repeated Nmat times
        operator_list[
            i
        ] = a_b  # the i^th element is now the annihilation operator for a single boson
        a_b_list.append(ListOp(operator_list))
    return [TensoredOp(a) for a in a_b_list]


# %%
def bmn2_hamiltonian(L: int = 2, N: int = 2, g2N: float = 0.2) -> SummedOp:
    """Construct the Hamiltonian of the bosonic BMN model as a sparse matrix.
    The cutoff for each boson is L while the 't Hooft coupling in g2N for a gauge group SU(N).
    The limited number of qubits only let us simulate N=2 and L=4 => for 6 bosons this is a 12 qubits problem.

    Args:
        L (int, optional): The cutoff of the bosonic modes (the annihilation operators will be LxL matrices). Defaults to 2.
        N (int, optional): The number of colors of a SU(N) gauge group. The degrees of freedom of one matrix will be N^2-1. Defaults to 2.
        g2N (float, optional): The 't Hooft coupling. Defaults to 0.2.

    Returns:
        SummedOp: the Hamiltonian in PauliOp form
    """
    print(
        f"Building bosonic BMN Hamiltonian for SU({N}) with cutoff={L} and coupling={g2N}\n"
    )
    # annihilation operators for bosons in full Hilbert space
    a_tensor = build_operators(L, N)
    N_bos = int(2 * (N ** 2 - 1))  # number of boson sites -> FIXED for mini-BMN 2
    assert len(a_tensor) == N_bos
    # identity in full Hilbert space
    i_b = MatrixOp(identity(L)).to_pauli_op()
    i_tensor = TensoredOp(ListOp([i_b] * N_bos))
    # Build the Hamiltonian
    # Start piece by piece
    # for each boson they are constructed using a and adag
    x_tensor = [1 / np.sqrt(2) * (~a + a) for a in a_tensor]
    # Free Hamiltonian
    # vacuum energy
    H_zero = 0.5 * N_bos * i_tensor
    # oscillators
    H_list = [H_zero]
    for a in a_tensor:
        H_list.append((~a @ a))
    H_osc = SummedOp(H_list)
    # Interaction among bosons
    quartic1 = SummedOp(
        [
            x_tensor[2] @ x_tensor[2] @ x_tensor[3] @ x_tensor[3],
            x_tensor[2] @ x_tensor[2] @ x_tensor[4] @ x_tensor[4],
            x_tensor[1] @ x_tensor[1] @ x_tensor[3] @ x_tensor[3],
            x_tensor[1] @ x_tensor[1] @ x_tensor[5] @ x_tensor[5],
            x_tensor[0] @ x_tensor[0] @ x_tensor[4] @ x_tensor[4],
            x_tensor[0] @ x_tensor[0] @ x_tensor[5] @ x_tensor[5],
        ]
    )
    quartic2 = SummedOp(
        [
            x_tensor[0] @ x_tensor[2] @ x_tensor[3] @ x_tensor[5],
            x_tensor[0] @ x_tensor[1] @ x_tensor[3] @ x_tensor[4],
            x_tensor[1] @ x_tensor[2] @ x_tensor[4] @ x_tensor[5],
        ],
        coeff=-2.0,
    )
    ### Quartic Interaction
    V = quartic1 + quartic2
    # full hamiltonian
    H = H_osc + g2N / N * V
    return H.to_pauli_op()


def eigenvalues_qiskit(qOp: SummedOp, k: int = 10):
    """Compute the lowest k eigenvalues of a quantum operator in matrix form qOp.
    Internally it uses numpy.

    Args:
        qOp (SummedOp): The quantum operator build from a PauliOp.
        k (int, optional): The number of lowest eigenvalues. Defaults to 10.
    """
    algorithm_globals.massive=True
    mes = NumPyEigensolver(k)  # k is the number of eigenvalues to compute
    result = mes.compute_eigenvalues(qOp)
    return np.real(result.eigenvalues)


# %%
def run_vqe(
    L: int = 2,
    N: int = 2,
    g2N: float = 0.2,
    optimizer: str = "COBYLA",
    maxit: int = 5000,
    varform: list = ["ry"],
    depth: int = 3,
    nrep: int = 10,
    rngseed: int = 0,
    h5: bool = True,
):
    """Run the main VQE solver for a bosonic BMN Hamiltonian where bosons are LxL matrices and the 't Hooft coupling is g2N for a SU(N) gauge group.
    The VQE is initialized with a specific optimizer and a specific variational quantum circuit based on EfficientSU2.

    Args:
        L (int, optional): Cutoff of each bosonic degree of freedom. Defaults to 2.
        N (int, optional): Colors for the SU(N) gauge group. Defaults to 2.
        g2N (float, optional): 't Hooft coupling. Defaults to 0.2.
        optimizer (str, optional): VQE classical optimizer. Defaults to "COBYLA".
        maxit (int, optional): Max number of iterations for the optimizer. Defaults to 5000.
        varform (str, optional): EfficientSU2 rotation gates. Defaults to 'ry'.
        depth (int, optional): Depth of the variational form. Defaults to 3.
        nrep (int, optional): Number of different random initializations of parameters. Defaults to 1.
        rngseed (int, optional): The random seed. Defaults to 0.
        h5 (bool, optional): The flag to save in HDF5 format. Defaults to True.
    """
    assert N == 2  # code only works for SU(2) :-(
    # Create the matrix Hamiltonian in PauliOp form
    qubitOp = bmn2_hamiltonian(L, N, g2N)
    # check the exact eigenvalues
    #print(
    #    f"Exact Result of discrete hamiltonian (qubit): {eigenvalues_qiskit(qubitOp)}"
    #)

    # Next, we create the variational form.
    var_form = EfficientSU2(
        qubitOp.num_qubits, su2_gates=varform, entanglement="full", reps=depth
    )

    # start a quantum instance
    # fix the random seed of the simulator to make values reproducible
    rng = np.random.default_rng(seed=rngseed)
    algorithm_globals.random_seed = rngseed
    algorithm_globals.massive=True
#    backend = Aer.get_backend(
#        "statevector_simulator", max_parallel_threads=6, max_parallel_experiments=0
#    )
    backend = Aer.get_backend('aer_simulator')
    q_instance = QuantumInstance(
        backend, seed_transpiler=rngseed, seed_simulator=rngseed
    )

    # initialize optimizers' parameters: number of iterations
    optimizers = {
        "COBYLA": COBYLA(maxiter=maxit),
        "L-BFGS-B": L_BFGS_B(maxfun=maxit),
        "SLSQP": SLSQP(maxiter=maxit),
        "NELDER-MEAD": NELDER_MEAD(maxfev=maxit),
    }

    print(f"\nRunning VQE main loop ...")
    start_time = time.time()
    try:
        optim = optimizers[optimizer]
        print(f"{optimizer} settings: {optim.settings}")
    except KeyError:
        print(
            f"Optimizer {optimizer} not found in our list. Try one of {[x for x in optimizers.keys()]}"
        )
        return

    results = {"counts": [], "energy": [], "casimir": []}
    casimir_result = (
        "NaN"  # initialize to NaN since it will not be defined if we do not measure it
    )

    # callback functions to store the counts from each iteration of the VQE
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    # run multiple random initial points
    for i in np.arange(nrep):
        counts = []
        values = []
        # initital points for the angles of the rotation gates
        random_init = rng.uniform(-2 * np.pi, 2 * np.pi, var_form.num_parameters)
        # Setup the VQE algorithm
        vqe = VQE(
            ansatz=var_form,
            optimizer=optim,
            initial_point=random_init,
            quantum_instance=q_instance,
            callback=store_intermediate_result,
            include_custom=True,
        )
        # run the VQE with our Hamiltonian operator
        result = vqe.compute_minimum_eigenvalue(qubitOp)
        vqe_result = np.real(result.eigenvalue)
        print(f"[{i}] - {varform} - [{optimizer}]: VQE gs energy: {vqe_result}")
        # collect results
        results["counts"].append(counts)
        results["energy"].append(values)
        results["casimir"].append(casimir_result)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Program runtime: {runtime} s")
    # make a dataframe from the results
    df = pd.DataFrame.from_dict(results)
    data_types_dict = {"counts": int, "energy": float}
    df = df.explode(["counts", "energy"]).astype(data_types_dict).rename_axis("rep")
    # report summary of energy across reps
    converged = df["energy"].groupby("rep").apply(min).values
    print(f"Statistics across {nrep} repetitions:\n-------------------")
    print(
        f"Least upper bound: {np.min(converged)}\nWorst upper bound: {np.max(converged)}\nMean bound: {np.mean(converged)}\nStd bound: {np.std(converged)}"
    )
    # save results on disk
    varname = "-".join(varform)
    g2Nstr = str(g2N).replace(".", "")
    os.makedirs("data", exist_ok=True)
    if h5:
        outfile = f"data/bosBMN_L{L}_l{g2Nstr}_convergence_{optimizer}_{varname}_depth{depth}_reps{nrep}_max{maxit}.h5"
        print(f"Save results on disk: {outfile}")
        df.to_hdf(outfile, "vqe")
    else:
        outfile = f"data/bosBMN_L{L}_l{g2Nstr}_convergence_{optimizer}_{varname}_depth{depth}_reps{nrep}_max{maxit}.gz"
        print(f"Save results on disk: {outfile}")
        df.to_pickle(outfile)
    return


# %%
if __name__ == "__main__":
    fire.Fire(run_vqe)
