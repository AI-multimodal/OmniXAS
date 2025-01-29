# %%
import os
import scienceplots
from omnixas.scripts.plots.scripts import AllTunedMetrics
from omnixas.model.trained_model import ModelTag, MeanModel

from omnixas.scripts.plots.scripts import AllTunedModels

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
# tuned_metrics = AllTunedMetrics()
all_tuned_models = AllTunedModels()

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


simulation_data = [
    # ("Ti", "FEFF", "mp-390", "test", 275, -1, 1, 1),  # anatase
    ("Ti", "FEFF", "mp-1840", "test", 366, -1, 1, 1),  # brookite
    # ("Ti", "FEFF", "mp-2657", None, None),  # rutile
    ("Ti", "VASP", "mp-390", "train", 1641, 0, 1, 1),  # anatase
    # ("Ti", "VASP", "mp-1840", "val", 264, 0, 1, 1),  # brookite
    ("Ti", "VASP", "mp-2657", "train", 1857, 0, 1, 1),  # rutile
    ("Cu", "FEFF", "mp-361", "train", 948, -3, 1, 1),  #
    # ("Cu", "VASP", "mp-361", None, None),
]

simulation_energy_grid = {
    "Ti": np.linspace(4964.50, 4964.50 + 35, 141),
    "Cu": np.linspace(8983.173, 8983.173 + 35, 141),
}


plt.style.use(["default", "science"])
apply_plot_theme(18)
fig, axs = plt.subplots(len(simulation_data), 1, figsize=(8, 12), sharex=True)
for ax, data in zip(axs.flatten(), simulation_data):
    (
        element,
        sim_type,
        material_id,
        split_name,
        idx,
        offset,
        max_factor,
        e_scale_factor,
    ) = data

    dataset = FEFF_data if sim_type == "FEFF" else VASP_data
    element_name = element if sim_type == "FEFF" else element + "_VASP"
    simulated_spectra = getattr(dataset[element_name], split_name).y[idx]

    exp_directory = (
        os.path.expanduser("~/Downloads/response/experiment") + f"/{element}"
    )
    files = [
        file
        for file in os.listdir(exp_directory)
        if material_id in file and (".dat" in file or ".csv" in file)
    ]
    assert len(files) == 1
    file = files[0]
    if ".dat" in file:
        experimental_data = np.loadtxt(os.path.join(exp_directory, file))
    elif ".csv" in file:
        import pandas as pd

        experimental_data = pd.read_csv(os.path.join(exp_directory, file)).values

    interpolation = interp1d(
        experimental_data[:, 0],
        experimental_data[:, 1],
        kind="cubic",
    )
    experimental_spectra = interpolation(
        (simulation_energy_grid[element] + offset) * e_scale_factor
    )

    tag = ModelTag(
        element=element,
        type=sim_type,
        feature="m3gnet",
        name="tunedUniversalXAS",
    )

    data = dataset[element_name]
    feature = getattr(data, split_name).X[idx]
    model = all_tuned_models[tag]
    prediction = model.predict([feature * 1000])[0] / 1000

    # # scale preditction to be same as experiment level
    # prediction -= np.min(prediction)
    # prediction *= np.max(simulated_spectra) / np.max(prediction)

    # ax.twinx().plot(
    #     experimental_spectra, label="experiment", color="red", linestyle="--"
    # )

    experimental_spectra -= np.min(experimental_spectra)
    experimental_spectra *= np.max(simulated_spectra) / np.max(experimental_spectra)
    experimental_spectra *= max_factor

    mse_simulation = np.mean((simulated_spectra - prediction) ** 2)
    mse_experiment = np.mean((experimental_spectra - prediction) ** 2)
    ax.plot(
        prediction,
        label="prediction",
        color="blue",
        linestyle="-",
    )
    ax.plot(
        experimental_spectra,
        label=f"experiment, MSE vs pred={mse_experiment:.1e}",
        color="red",
        linestyle="--",
    )
    ax.plot(
        simulated_spectra,
        label=f"simulation, MSE vs pred={mse_simulation:.1e}",
        color="green",
    )
    ax.set_title(
        # f"{element} {material_id} {split_name} {sim_type}",
        f"{element} {material_id} ",
    )
    ax.legend()
    ax.set_xlim(0, 140)
    xticks = ax.get_xticks()
    xtick_labels = xticks * 0.25
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(r"$\Delta$E (eV)")
plt.tight_layout()
plt.savefig("comparison_with_experiment.pdf", bbox_inches="tight", dpi=300)

# %%
