# %%
import scienceplots
from omnixas.scripts.plots.scripts import AllTunedMetrics
from omnixas.model.trained_model import ModelTag, MeanModel

from omnixas.utils.spectra_outliers import OutlierDetector
from scipy.interpolate import interp1d
from omnixas.data import MLData
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from omnixas.data import MLSplits
from omnixas.scripts.plots.constants import FEFFSplits, VASPSplits

# %%

FEFF_data = {k.value: v for k, v in FEFFSplits().items()}
VASP_data = {k.value + "_VASP": v for k, v in VASPSplits().items()}

# %%

idxs = (
    {
        "Ti": {
            "FEFF": {
                "mp-390": ("test", 275),  # anatase
                "mp-1840": ("test", 366),  # brookite
                "mp-2657": None,  # rutile
            },
            "VASP": {
                "mp-390": ("train", 1641),
                "mp-1840": ("val", 264),
                "mp-2657": ("train", 1857),
            },
        }
    },
    {
        "Cu": {
            "FEFF": {
                "mp-361": ("train", 948),  #
            },
            "VASP": {
                "mp-361": None,
            },
        }
    },
)

# %%

idx = 366
split = FEFF_data["Ti"].test
simulation = split.y[idx]
plt.plot(simulation)

# %%

experiment = np.loadtxt(
    "/Users/bnl/Downloads/response/experiment/Ti/brookite_mp-1840.dat"
)
plt.plot(experiment[:, 0], experiment[:, 1], label="full_experiment")
plt.legend()
plt.show()


# %%

# interpolate experimental to same as simulation grid


experiment_energy = experiment[:, 0] + 1.25
# experiment_energy += 1
experiment_spectra = experiment[:, 1]
f = interp1d(
    experiment_energy,
    experiment_spectra,
    kind="cubic",
    bounds_error=False,
    fill_value=0,
)


simulation_grid = np.linspace(4964.50, 4964.50 + 35, 141)
experiment_spectra_interp = f(simulation_grid)
# scale experimental_spectra to be same max as simulation
experiment_spectra_interp -= np.min(experiment_spectra_interp)
experiment_spectra_interp *= np.max(simulation) / np.max(experiment_spectra_interp)


# %%

metrics = AllTunedMetrics()

# %%

tag = ModelTag(element="Ti", type="FEFF", feature="m3gnet", name="tunedUniversalXAS")
prediction = metrics[tag].predictions[idx]


# %%
plt.style.use(["default", "science"])
plt.s
plt.plot(simulation_grid, experiment_spectra_interp, label="experiment")
plt.plot(simulation_grid, simulation, label="simulation")
plt.legend()
plt.show()

# %%


def apply_plot_theme(FONTSIZE=18):
    plt.style.use(["default", "science"])
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "xtick.labelsize": FONTSIZE * 0.8,
            "ytick.labelsize": FONTSIZE * 0.8,
            "legend.fontsize": FONTSIZE * 0.75,
            "axes.labelsize": FONTSIZE,
        }
    )


plt.style.use(["default", "science"])
apply_plot_theme(20)
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.02})
ax[0].plot(prediction, label="prediction", color="blue")
ax[0].plot(simulation, label="simulation", color="green", linestyle="--")
ax[0].legend(loc="upper left")
ax[1].plot(prediction, label="prediction", color="blue")
ax[1].plot(experiment_spectra_interp, label="experiment", color="red", linestyle="--")
ax[1].legend(loc="upper left")
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels(simulation_grid[::10].round(2))
ax[1].set_xlabel(r"$E$ (eV)")
# mp-1840": ("test", 366),  # brookite
plt.suptitle(
    r"TiO$_2$ mp-1840 (brookite)",
    # location has to be bit closer to to
    x=0.5,
    y=0.925,
)
plt.savefig("comparision_TiO2_mp-1840.pdf", bbox_inches="tight", dpi=300)
plt.tight_layout()
