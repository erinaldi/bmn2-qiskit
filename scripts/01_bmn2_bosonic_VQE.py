# %%
import time
import fire
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse import kron
from scipy.sparse.linalg import eigsh
from qiskit import Aer
from qiskit.opflow import MatrixOp
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms import NumPyEigensolver, VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B, NELDER_MEAD
from qiskit.utils import algorithm_globals, QuantumInstance


# %%
def build_operators(L: int, N: int) -> list:
    """Generate all the annihilation operators needed to build the hamiltonian

    Args:
        L (int): the cutoff of the single site Fock space
        N (int): the number of colors of gauge group SU(N)

    Returns:
        list: a list of annihilation operators, length=N_bos
    """
    # The annihilation operator for the single boson
    a_b = diags(np.sqrt(np.linspace(1, L - 1, L - 1)), offsets=1)
    # The identity operator of the Fock space of a single boson
    i_b = identity(L)
    # Bosonic Hilbert space
    N_bos = int(2 * (N ** 2 - 1))  # number of boson sites -> fixed for mini-BMN 2
    product_list = [i_b] * N_bos  # only the identity for bosons repeated N_bos times
    a_b_list = []  # this will contain a1...a6
    for i in np.arange(0, N_bos):  # loop over all bosonic operators
        operator_list = product_list.copy()  # all elements are the identity operator
        operator_list[
            i
        ] = a_b  # the i^th element is now the annihilation operator for a single boson
        a_b_list.append(
            operator_list[0]
        )  # start taking tensor products from first element
        for a in operator_list[1:]:
            a_b_list[i] = kron(
                a_b_list[i], a
            )  # do the outer product between each operator_list element
    return a_b_list


# %%
def build_gauge_casimir(L: int, N: int):
    """Generate the gauge generators operators

    Args:
        L (int): the single site cutoff of the Fock space
        N (int): the number of colors in the gauge group SU(N)

    Returns:
        scipy.sparse : The sparse matrix for \sum_i G_i^2
    """
    # generate the annihilation operators
    bosons = build_operators(L, N)
    # define the generator list for SU(2)
    g_list = [0] * 3
    g_list[0] = 1j * (
        bosons[1].conjugate().transpose() * bosons[2]
        - bosons[2].conjugate().transpose() * bosons[1]
        + bosons[4].conjugate().transpose() * bosons[5]
        - bosons[5].conjugate().transpose() * bosons[4]
    )
    g_list[1] = 1j * (
        bosons[2].conjugate().transpose() * bosons[0]
        - bosons[0].conjugate().transpose() * bosons[2]
        + bosons[5].conjugate().transpose() * bosons[3]
        - bosons[3].conjugate().transpose() * bosons[5]
    )
    g_list[2] = 1j * (
        bosons[0].conjugate().transpose() * bosons[1]
        - bosons[1].conjugate().transpose() * bosons[0]
        + bosons[3].conjugate().transpose() * bosons[4]
        - bosons[4].conjugate().transpose() * bosons[3]
    )

    return g_list[0] * g_list[0] + g_list[1] * g_list[1] + g_list[2] * g_list[2]


# %%
def bmn2_hamiltonian(L: int = 2, N: int = 2, g2N: float = 0.2):
    """Construct the Hamiltonian of the bosonic BMN model as a sparse matrix.
    The cutoff for each boson is L while the 't Hooft coupling in g2N for a gauge group SU(N).
    The limited number of qubits only let us simulate N=2 and L=4 => for 6 bosons this is a 12 qubits problem.

    Args:
        L (int, optional): The cutoff of the bosonic modes (the annihilation operators will be LxL matrices). Defaults to 2.
        N (int, optional): The number of colors of a SU(N) gauge group. The degrees of freedom of one matrix will be N^2-1. Defaults to 2.
        g2N (float, optional): The 't Hooft coupling. Defaults to 0.2.
    """
    print(
        f"Building bosonic BMN Hamiltonian for SU({N}) with cutoff={L} and coupling={g2N}\n"
    )
    a_b_list = build_operators(L,N)
    N_bos = int(2 * (N ** 2 - 1))  # number of boson sites -> FIXED for mini-BMN 2
    # Build the Hamiltonian
    # Start piece by piece
    x_list = []
    # only use the bosonic operators
    for op in a_b_list:
        x_list.append(1 / np.sqrt(2) * (op.conjugate().transpose() + op))
    # Free Hamiltonian
    H_k = 0
    for a in a_b_list:
        H_k = H_k + a.conjugate().transpose() * a
    # vacuum energy
    H_k = H_k + 0.5 * N_bos * identity(L ** N_bos)
    # Interaction among bosons
    V_b = (
        x_list[2] * x_list[2] * x_list[3] * x_list[3]
        + x_list[2] * x_list[2] * x_list[4] * x_list[4]
        + x_list[1] * x_list[1] * x_list[3] * x_list[3]
        + x_list[1] * x_list[1] * x_list[5] * x_list[5]
        + x_list[0] * x_list[0] * x_list[4] * x_list[4]
        + x_list[0] * x_list[0] * x_list[5] * x_list[5]
        - 2 * x_list[0] * x_list[2] * x_list[3] * x_list[5]
        - 2 * x_list[0] * x_list[1] * x_list[3] * x_list[4]
        - 2 * x_list[1] * x_list[2] * x_list[4] * x_list[5]
    )
    # full hamiltonian
    return H_k + g2N / N * V_b


def eigenvalues_scipy(H, k: int = 10):
    """Compute the lowest k eigenvalues of a sparse symmetric matrix H.

    Args:
        H (scipy.sparse matrix): The Hamiltonian in the form of a sparse matrix
        k (int): The number of lowest eigenvalues to compute. Defaults to 10.
    """
    eigv = eigsh(H, k, which="SA", return_eigenvectors=False, tol=0)
    return np.real(eigv[::-1])


def eigenvalues_qiskit(qOp: MatrixOp, k: int = 10):
    """Compute the lowest k eigenvalues of a quantum operator in matrix form qOp.
    Internally it uses numpy.

    Args:
        qOp (MatrixOp): The quantum operator build from a matrix.
        k (int, optional): The number of lowest eigenvalues. Defaults to 10.
    """
    mes = NumPyEigensolver(k)  # k is the number of eigenvalues to compute
    result = mes.compute_eigenvalues(qOp)
    return np.real(result.eigenvalues)


# %%
def check_expectation(H, O, k: int = 1):
    """Compute the lowest k eigenstates of a sparse symmetric matrix H
    and then compute the expectation value of O.

    Args:
        H (scipy.sparse matrix): The Hamiltonian in the form of a sparse matrix
        O (scipy.sparse matrix): The operator to "measure" in the form of a sparse matrix
        k (int): The number of lowest eigenstates to compute. Defaults to 1.
    """
    _, eigk = eigsh(H, k, which="SA", return_eigenvectors=True, tol=0)
    expect = []
    for i in np.arange(k):
        bra = eigk[:,i].conjugate().transpose()
        ket = eigk[:,i]
        expect.append(bra.dot(O.dot(ket)))
    return np.real(np.array(expect))


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
    G2: bool = False,
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
        G2 (bool, optional): The flag to compute the expectation value of the gauge Casimir. Defaults to False.
        h5 (bool, optional): The flag to save in HDF5 format. Defaults to True.
    """
    # Create the matrix Hamiltonian
    H = bmn2_hamiltonian(L, N, g2N)
    # Now, we take the Hamiltonian matrix and map it onto a qubit operator.
    qubitOp = MatrixOp(primitive=H)
    # check the exact eigenvalues
    print(f"Exact Result of discrete hamiltonian (matrix): {eigenvalues_scipy(H)}")
    print(
        f"Exact Result of discrete hamiltonian (qubit): {eigenvalues_qiskit(qubitOp)}"
    )
    # create the gauge casimir and check it if requested
    if G2:
        G = build_gauge_casimir(L,N)
        print(f"Exact Result of Casimir on GS (matrix): {check_expectation(H,G)}")
        gOp = MatrixOp(primitive=G)

    # Next, we create the variational form.
    var_form = EfficientSU2(
        qubitOp.num_qubits, su2_gates=varform, entanglement="full", reps=depth
    )

    # start a quantum instance
    # fix the random seed of the simulator to make values reproducible
    rng = np.random.default_rng(seed=rngseed)
    algorithm_globals.random_seed = rngseed
    backend = Aer.get_backend(
        "statevector_simulator", max_parallel_threads=6, max_parallel_experiments=0
    )
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
    except KeyError:
        print(
            f"Optimizer {optimizer} not found in our list. Try one of {[x for x in optimizers.keys()]}"
        )
        return

    results = {"counts": [], "energy": [], "casimir": []}
    casimir_result = 'NaN'  # initialize to NaN since it will not be defined if we do not measure it

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
        )
        # run the VQE with our Hamiltonian operator
        if G2:
            result = vqe.compute_minimum_eigenvalue(qubitOp,aux_operators=[gOp])
            vqe_result = np.real(result.eigenvalue)
            casimir_result = np.real(result.aux_operator_eigenvalues[0,0])
            print(f"[{i}] - {varform} - [{optimizer}]: VQE gs energy: {vqe_result} | VQE gs gauge casimir: {casimir_result}")
        else:
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
    if h5:
        outfile = f"data/bosBMN_L{L}_l{g2Nstr}_convergence_{optimizer}_{varname}_depth{depth}_reps{nrep}_max{maxit}.h5"
        print(f"Save results on disk: {outfile}")
        df.to_hdf(outfile, "vqe")
    else:
        outfile = f"data/bosBMN_L{L}_l{g2Nstr}_convergence_{optimizer}_{varname}_depth{depth}_reps{nrep}_max{maxit}.gz"
        print(f"Save results on disk: {outfile}")
        df.to_pickle(outfile)
    return df


# %%
if __name__ == "__main__":
    fire.Fire(run_vqe)
