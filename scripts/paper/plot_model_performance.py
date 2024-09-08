# %%

from functools import cached_property
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib import colors as mcolors

from config.defaults import cfg
from scripts.paper.universal_TL_mses import compute_universal_model_metrics
from src.data.ml_data import DataQuery
from src.models.trained_models import MeanModel, Trained_FCModel


def generate_performance_comparison_plot(
    model_names=["universal_feff", "per_compound_tl", "ft_tl"],
    metric: Literal[
        "mse",
        "geometric_mean_of_mse_per_spectra",
    ] = "mse",
    use_relative: bool = True,
    include_vasp: bool = True,
    ax=None,
    FONTSIZE=20,
    y_label=r"Performance ($\eta$)",
):

    if ax is None:
        plt.style.use(["default", "science"])
        fig, ax = plt.subplots(figsize=(10, 7))

    sims = list(zip(cfg.compounds, ["FEFF"] * len(cfg.compounds)))
    if include_vasp:
        sims += [("Ti", "VASP"), ("Cu", "VASP")]

    # BARS PROPERTIES
    BAR_CENTER_FACTOR = 1.5
    bar_width = 0.95 / len(model_names)
    compound_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(cfg.compounds) + 2))
    hatches = {
        "universal_feff": "..",
        "per_compound_tl": ".....",
        "ft_tl": "",
    }
    bar_fill_dict = {k: v for k, v in zip(model_names, [False, False, True])}
    _bar_loc_dict = {k: i for i, k in enumerate(model_names)}

    universal_model_metrics = compute_universal_model_metrics(
        metric,
        compute_relative=use_relative,
    )

    models_metrics = {}
    models_metrics["UniversalXAS"] = universal_model_metrics["per_compound"]

    ax.bar(
        np.arange(len(cfg.compounds)) * BAR_CENTER_FACTOR
        + _bar_loc_dict["universal_feff"] * bar_width,
        universal_model_metrics["per_compound"].values(),
        bar_width,
        label="universal-feff-tl",
        edgecolor=compound_colors,
        fill=bar_fill_dict["universal_feff"],
        color=compound_colors,
        zorder=3,
        hatch=hatches["universal_feff"],
    )

    for model_name in ["per_compound_tl", "ft_tl"]:
        bar_positions = (
            np.arange(len(sims)) * BAR_CENTER_FACTOR
            + _bar_loc_dict[model_name] * bar_width
        )
        bar_positions[-2:] = bar_positions[-2:] + bar_width

        if not use_relative:
            metrics = {
                (c, sim_type): getattr(
                    Trained_FCModel(DataQuery(c, sim_type), name=model_name), metric
                )
                for c, sim_type in sims
            }
        else:
            metrics = {
                (c, sim_type): getattr(MeanModel(DataQuery(c, sim_type)), metric)
                / getattr(
                    Trained_FCModel(DataQuery(c, sim_type), name=model_name), metric
                )
                for c, sim_type in sims
            }

        models_metrics[model_name] = metrics

        ax.bar(
            bar_positions,
            metrics.values(),
            bar_width,
            color=compound_colors,
            label=model_name,
            fill=bar_fill_dict[model_name],
            edgecolor=compound_colors,
            hatch=hatches[model_name],
            zorder=3,
        )

    ax.set_ylabel(y_label, fontsize=FONTSIZE * 1.2)
    ax.set_xlabel("Element", fontsize=FONTSIZE * 1.2, labelpad=-10)
    xticks = np.arange(len(sims)) * BAR_CENTER_FACTOR + bar_width
    if include_vasp:
        # add vertical line before vap
        # ax.axvline(x=xticks[-2] - bar_width * 1.5, color="grey", linestyle="--")
        # add background color just for the vasp region
        ax.axvspan(
            xticks[-2] - bar_width * 1.5,
            ax.get_xlim()[1],
            alpha=0.1,
            color="grey",
        )
        # mentsion is text that that region is VASP at center of that region on middle of y axis
        ax.text(
            xticks[-1] - bar_width / 2,
            ax.get_ylim()[1] * 0.75,
            "VASP",
            fontsize=FONTSIZE * 1.2,
            ha="center",
            va="center",
            color="grey",
        )
        # set the tick position of VASPs by one bar to right
        xticks[-2:] = xticks[-2:] + bar_width * 1.5
    ax.set_xticks(xticks)
    ax.set_xlim(-bar_width * 1.5, ax.get_xlim()[1] - 1.5 * bar_width)
    yticks = ax.get_yticks()
    assert np.all([x % 1 == 0 for x in yticks]), "Y ticks assumed to be integers"
    ax.set_yticklabels([f"{int(y)}" for y in yticks], fontsize=FONTSIZE)
    ax.set_xticklabels(
        cfg.compounds
        + (
            [
                "Ti\n" + r"{\Large VASP}",
                "Cu\n" + r"{\Large VASP}",
            ]
            if include_vasp
            else []
        ),
        fontsize=FONTSIZE,
    )
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            # edgecolor="gray",
            hatch=hatches[model_name],
            fill=fill,
            color="gray",
        )
        for model_name, fill in bar_fill_dict.items()
    ]
    handle_names_dict = {
        "universal_feff": "UniversalXAS",
        "per_compound_tl": "ExpertXAS",
        "ft_tl": "Tuned-UniversalXAS",
    }
    legend = ax.legend(
        handles,
        [handle_names_dict[model_name] for model_name in model_names],
        fontsize=FONTSIZE * 0.8,
        handlelength=2,
        handleheight=1,
        # x and y
        loc=(0.41, 0.75),
    )
    for text in legend.get_texts():
        text.set_rotation(0)  # Ensure the text is horizontal
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()

    file_name = f"model_perf_{metric}.pdf"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    # return ax
    return models_metrics


ASPECT_RATIO = 4 / 3
HEIGHT = 6
WEIGHT = HEIGHT * ASPECT_RATIO
DPI = 300
plt.style.use(["default", "science", "tableau-colorblind10"])
fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=DPI)

models_metrics = generate_performance_comparison_plot(
    ax=ax,
    metric="median_of_mse_per_spectra",
    use_relative=True,
)

# %%

# =============================================================================
# GENERATE LATEx TABLE
# =============================================================================
for model_name in ["per_compound_tl", "ft_tl"]:
    dict_feff = {
        k[0]: v for k, v in models_metrics[model_name].items() if k[1] == "FEFF"
    }
    dict_vasp = {
        str(k[0]) + "_VASP": v
        for k, v in models_metrics[model_name].items()
        if k[1] == "VASP"
    }
    model_labels = {
        "per_compound_tl": "ExpertXAS",
        "ft_tl": "Tuned-UniversalXAS",
        "universal": "UniversalXAS",
    }
    models_metrics[model_labels[model_name]] = {**dict_feff, **dict_vasp}
models_metrics.pop("per_compound_tl")
models_metrics.pop("ft_tl")
df = pd.DataFrame(models_metrics)
df = df.fillna("N/A")
df.index = df.index.str.replace("_", " ")
df = df.reset_index().rename(columns={"index": "Element"})
df = df[["Element", "ExpertXAS", "UniversalXAS", "Tuned-UniversalXAS"]]
df = df.set_index("Element")


# Function to format numbers
def format_number(x):
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return x


df = df.applymap(format_number)
# Generate LaTeX table
latex_table = df.to_latex(
    index=True,
    column_format="|l|c|c|c|",
    bold_rows=False,
    caption="Model Metrics Comparison",
    label="tab:model-metrics",
)
# Add additional LaTeX formatting
latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[h!]\n\\centering")
latex_table = latex_table.replace(
    "\\begin{tabular}", "\\begin{tabular}{|l|c|c|c|}\n\\hline"
)
latex_table = latex_table.replace("\\end{tabular}", "\\hline\n\\end{tabular}")
print(latex_table)
# ==============================================================================