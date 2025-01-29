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
from omnixas.scripts.plots.scripts import AllTunedMetrics, AllExpertMetrics
# %%

metrics = AllTunedMetrics()
metrics = {**metrics, **AllExpertMetrics()}

# %%


def apply_plot_theme(FONTSIZE=12):
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


tag = list(metrics.keys())[0]
y_pred = metrics[tag].predictions
y = metrics[tag].targets
pct_err = (y_pred - y) / y * 100


FONTSIZE = 20
fig, axs = plt.subplots(
    4,
    2,
    figsize=(16, 12),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.1, "hspace": 0.2},
)
ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]

quantiles = [0.05, 0.25, 0.75, 0.95]
colors = ["C0", "C0"]

for ax, element in zip(axs.ravel(), ordered_elements):
    tag = ModelTag(
        element=element,
        type="FEFF",
        feature="m3gnet",
        name="tunedUniversalXAS",
        # name="expertXAS",
    )
    y_pred = metrics[tag].predictions
    y = metrics[tag].targets
    pct_err = (y_pred - y) / y * 100
    for i in [0, 1]:
        ax.plot(np.quantile(pct_err, 0.5, axis=0), color="black")
        ax.fill_between(
            np.arange(pct_err.shape[1]),
            np.quantile(pct_err, quantiles[i], axis=0),
            np.quantile(pct_err, quantiles[-i - 1], axis=0),
            alpha=0.4 if quantiles[i] in [0.05, 0.095] else 0.9,
            color="gray",
        )
        ax.set_xlim(0, 140)
    ax.set_ylim(-55, 55)
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_title(element)

# %%

FONTSIZE = 20
fig, axs = plt.subplots(
    3,
    3,
    figsize=(6, 6),
    sharex=True,
    sharey=True,
    # gridspec_kw={"wspace": 0.1, "hspace": 0.1},
)
ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
for ax, element in zip(axs.ravel(), ordered_elements):
    tag = ModelTag(
        element=element,
        type="FEFF",
        feature="m3gnet",
        name="tunedUniversalXAS",
        # name="expertXAS",
    )
    y_pred = metrics[tag].predictions
    y = metrics[tag].targets
    ax.scatter(y, y_pred, s=0.1, alpha=0.1)
    ax.set_aspect("equal")
    ax.set_title(element)
axs[-1, -1].axis("off")
plt.tight_layout()

# %%

FONTSIZE = 20
apply_plot_theme(FONTSIZE)
fig, axs = plt.subplots(
    3,
    3,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
    # gridspec_kw={"wspace": 0.1, "hspace": 0.1},
)
ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
for ax, element in zip(axs.ravel(), ordered_elements):
    tag = ModelTag(
        element=element,
        type="FEFF",
        feature="m3gnet",
        name="tunedUniversalXAS",
        # name="expertXAS",
    )
    y_pred = metrics[tag].predictions
    y = metrics[tag].targets
    median_pct_err = np.median(np.abs(y_pred - y) / y * 100, axis=1)
    ax.hist(
        median_pct_err,
        bins=np.arange(0, 30, 0.5),
        density=True,
        # no fill color only edge color with bars
        edgecolor="black",
        color="white",
        # histtype="step",
    )
    ax.set_title(element)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.35)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks(np.arange(0, 0.40, 0.05))
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
for i in [0, 3, 6]:
    ax_ = axs.ravel()[i]
    ax_.set_ylabel("Density")
for i in range(6, 9):
    ax_ = axs.ravel()[i]
    ax_.set_xlabel(r"Median Abs. Error $\%$")

axs[-1, -1].axis("off")
plt.tight_layout()

# %%

# BOOTSTRAPPED
FONTSIZE = 20
apply_plot_theme(FONTSIZE)
fig, axs = plt.subplots(
    3,
    3,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
)

n_bootstraps = 10000  # Number of bootstrap iterations
sample_size = 250  # Size of each bootstrap sample
bins = np.arange(0, 30, 0.5)

ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
for ax, element in zip(axs.ravel(), ordered_elements):
    tag = ModelTag(
        element=element,
        type="FEFF",
        feature="m3gnet",
        name="tunedUniversalXAS",
    )
    y_pred = metrics[tag].predictions
    y = metrics[tag].targets
    median_pct_err = np.median(np.abs(y_pred - y) / y * 100, axis=1)

    # Store histogram counts for averaging
    all_counts = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(median_pct_err, size=sample_size)
        counts, _ = np.histogram(bootstrap_sample, bins=bins)
        all_counts.append(counts)

    # Calculate mean counts across bootstrap samples
    mean_counts = np.mean(all_counts, axis=0)

    # Plot mean histogram
    ax.bar(
        bins[:-1],
        mean_counts,
        width=0.5,
        edgecolor="black",
        color="white",
        align="edge",
    )

    ax.set_title(element)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 35)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
fig.suptitle("10,000x bootstrapped (250 samples) Error Distribution")
axs[-1, -1].axis("off")
plt.tight_layout()

# %%

FONTSIZE = 20
apply_plot_theme(FONTSIZE)
fig, axs = plt.subplots(
    4,
    2,
    figsize=(16, 12),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.1, "hspace": 0.2},
)

n_bootstraps = 500  # Number of bootstrap iterations
sample_size = 250  # Size of each bootstrap sample
# quantiles = [0.05, 0.25, 0.75, 0.95]
quantiles = [0.05, 0.25, 0.75, 0.95]
colors = ["C0", "C0"]

ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
for ax, element in zip(axs.ravel(), ordered_elements):
    tag = ModelTag(
        element=element,
        type="FEFF",
        feature="m3gnet",
        name="tunedUniversalXAS",
    )
    y_pred = metrics[tag].predictions
    y = metrics[tag].targets
    pct_err = (y_pred - y) / y * 100

    # Store quantiles for averaging
    bootstrap_quantiles = []
    for _ in range(n_bootstraps):
        # Random sampling with replacement
        indices = np.random.choice(pct_err.shape[0], size=sample_size)
        bootstrap_sample = pct_err[indices]

        # Calculate quantiles for this bootstrap sample
        sample_quantiles = [np.quantile(bootstrap_sample, q, axis=0) for q in quantiles]
        bootstrap_quantiles.append(sample_quantiles)

    # Calculate mean quantiles across bootstrap samples
    mean_quantiles = np.mean(bootstrap_quantiles, axis=0)

    # Plot mean quantiles
    ax.plot(np.median(pct_err, axis=0), color="black")
    for i in [0, 1]:
        ax.fill_between(
            np.arange(pct_err.shape[1]),
            mean_quantiles[i],
            mean_quantiles[-i - 1],
            alpha=0.4 if quantiles[i] in [0.05, 0.95] else 0.9,
            color="gray",
        )

    ax.set_xlim(0, 140)
    ax.set_ylim(-55, 55)
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_title(element)

fig.suptitle("10,000x bootstrapped (250 samples) Quantile Plot")
plt.tight_layout()

# %%
