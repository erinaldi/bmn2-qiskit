# %%
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("../figures/paper.mplstyle")
import os, fire

# %%
def read_data(
    optimizer: str,
    p: dict,
):
    """Read the VQE convergence data for the mini BMN model from disk

    Args:
        optimizer (str): The optimizer used.
        p (dict): The dictionary with the parameters for the filename.
    Returns:
        pandas.DataFrame: The dataframe collecting the results of the convergence
    """
    filename = f"{p['f']}/miniBMN_l{p['l']}_convergence_{optimizer}_{p['v']}_depth{p['d']}_reps{p['n']}_max{p['m']}.h5"
    assert os.path.isfile(filename)
    return pd.read_hdf(filename, "vqe")


# %%
def collect_data(
    optimizers: list,
    p: dict,
):
    """Read the VQE convergence data for the mini BMN model from disk
    Args:
        optimizer (str): The optimizer used.
        p (dict): The dictionary with the parameters for the filename.
    Returns:
        pandas.DataFrame: The dataframe collecting the all the results of the convergence
    """
    # concatenate the results from all files
    frames = [read_data(o, p) for o in optimizers]
    return pd.concat(frames, keys=optimizers, names=["Optimizer"])


# %%
def plot_convergence(
    optimizers: list = ["COBYLA", "SLSQP", "L-BFGS-B", "NELDER-MEAD"],
    g2N: float = 0.2,
    maxit: int = 10000,
    varform: list = ["ry"],
    depth: int = 3,
    nrep: int = 10,
    datafolder: str = "../data",
    figfolder: str = "../figures",
    ht: float = 0.00328726,
    up: int = 1000,
):
    """Read the VQE convergence data for the mini BMN model from disk

    Args:
        optimizers (list, optional): The optimizers to collect in the plot. Defaults to ["COBYLA","SLSQP","L-BFGS-B","NELDER-MEAD"].
        g2N (float, optional): The 't Hooft coupling. Defaults to 0.2.
        maxit (int, optional): The maximun number of iterations of the optimizer. Defaults to 5000.
        varform (list, optional): The prametric gates of the variational form. Defaults to ["ry"].
        depth (int, optional): The depth of the variational form. Defaults to 3.
        nrep (int, optional): The number of random initial points used. Defaults to 1.
        datafolder (str, optional): The data folder of the file. Defaults to "../data".
    """
    # setup parameters
    params = dict()
    params["l"] = str(g2N).replace(".", "")
    params["d"] = depth
    params["v"] = "-".join(varform)
    params["m"] = maxit
    params["n"] = nrep
    params["f"] = datafolder
    # collect data
    result = collect_data(optimizers, params)
    # get best runs
    gs = dict()
    for r in optimizers:
        gs[r] = result.loc[r].groupby("rep").apply(min).energy
    gsdf = pd.DataFrame.from_dict(gs, dtype=float)
    # Plot
    # select the best runs for each optimizer
    fig, ax = plt.subplots(figsize=(20, 8))
    for o in optimizers:
        result.loc[o, gsdf[o].idxmin()].plot(
            x="counts", y="energy", xlim=[0, up], label=o, ax=ax
        )
    ax.axhline(ht, c="k", ls="--", lw="2", label="HT")
    ax.set_xlabel("iterations")
    ax.set_ylabel("VQE energy")
    ax.legend(loc="upper right")
    filename = f"{figfolder}/miniBMN_l{params['l']}_convergence_{params['v']}_depth{params['d']}_nr{params['n']}_max{params['m']}"
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png")
    plt.close()


# %%
if __name__ == "__main__":
    fire.Fire(plot_convergence)
