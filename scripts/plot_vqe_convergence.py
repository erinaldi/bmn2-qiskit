# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import fire
matplotlib.use("Agg")
plt.style.use("figures/paper.mplstyle")

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
    filename = f"{p['f']}_l{p['l']}_convergence_{optimizer}_{p['v']}_depth{p['d']}_reps{p['n']}_max{p['m']}.{p['s']}"
    if not os.path.isfile(filename):
        print(f"{filename} does not exist.")
        sys.exit()
    if p['s'] == 'h5':
        df = pd.read_hdf(filename, "vqe")
    if p['s'] == 'gz':
        df = pd.read_pickle(filename)

    return df[df.counts<=p['m']]


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
    dataprefix: str = "data/miniBMN",
    datasuffix: str = "h5",
    figprefix: str = "figures/miniBMN",
    ht: float = 0.0,
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
        dataprefix (str, optional): The data folder of the file. Defaults to "data/miniBMN".
        datasuffix (str, optional): The data file extension. Defaults to "h5".
        figprefix (str, optional): The output folder and name prefix for the figure. Defaults to "figures/miniBMN".
        ht (float, optional): The exact truncation data. Defaults to 0.0.
        up (int, optional): The upper limit of iterations to plot. Defaults to 1000.
    """
    # setup parameters
    params = dict()
    params["l"] = str(g2N).replace(".", "")
    params["d"] = depth
    params["v"] = "-".join(varform)
    params["m"] = maxit
    params["n"] = nrep
    params["f"] = dataprefix
    params["s"] = datasuffix
    assert type(optimizers).__name__ == "list"
    # collect data
    result = collect_data(optimizers, params)
    # get best runs
    gs = dict()
    for r in optimizers:
        gs[r] = result.loc[r].groupby("rep").apply(min).energy
    gsdf = pd.DataFrame.from_dict(gs, dtype=float)
    print(gsdf.describe().T[["min", "max", "mean", "std"]])
    # Plot
    # select the best runs for each optimizer
    fig, ax = plt.subplots()
    for o in optimizers:
        result.loc[o, gsdf[o].idxmin()].plot(
            x="counts", y="energy", xlim=[0, up], label=o, ax=ax
        )
    ax.axhline(ht, c="k", ls="--", lw="2", label="HT")
    ax.set_xlabel("iterations")
    ax.set_ylabel("VQE energy")
    ax.legend(loc="upper right")
    filename = f"{figprefix}_l{params['l']}_convergence_{params['v']}_depth{params['d']}_nr{params['n']}_max{params['m']}_xlim{up}"
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png")
    plt.savefig(f"{filename}.svg")
    plt.close()


# %%
if __name__ == "__main__":
    fire.Fire(plot_convergence)
