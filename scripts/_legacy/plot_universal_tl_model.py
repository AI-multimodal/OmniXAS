# %%
# =============================================================================
# Plots related to Universal-TL-MLP model
# =============================================================================

from matplotlib import colors as mcolors
import scienceplots
from universal_TL_mses import compute_universal_model_metrics
from src.models.trained_models import MeanModel
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery
from config.defaults import cfg
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property
from src.models.trained_models import LinReg


def plot_universal_tl_vs_per_compound_tl(
    # model_names=None,  # ["per_compound_tl", "ft_tl"],
    model_names=["linreg", "universal_feff", "per_compound_tl", "ft_tl"],
    relative_to_per_compound_mean_model=False,
    include_vasp: bool = False,
    ax=None,
    add_weighted=False,
    FONTSIZE=18,
    # include_linreg=False,
):
    plt.style.use(["default", "science"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # bar_width = 0.15
    bar_width = 0.95 / len(model_names)

    # compound_colors = [ # Tableau colorblind 10
    #     "#006BA4",
    #     "#FF800E",
    #     "#ABABAB",
    #     "#595959",
    #     "#5F9ED1",
    #     "#C85200",
    #     "#898989",
    #     "#A2C8EC",
    #     "#FFBC79",
    #     "#CFCFCF",
    # ]

    cmap = "tab10"

    compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))

    colors = {
        "universal_feff": compound_colors[0],
        "per_compound_tl": compound_colors[1],
        "ft_tl": compound_colors[2],
        "linreg": compound_colors[3],
    }

    hatches = {
        "universal_feff": "..",
        "per_compound_tl": ".....",
        "ft_tl": "",
        "linreg": "////",
        "acsf": "xxx",
        "soap": "ooo",
    }

    # # NO HATCHES
    # hatches = {k: v for k, v in zip(model_names, [""] * len(model_names))}

    # bar_alpha_dict = {
    #     k: v for k, v in zip(model_names, np.linspace(0.4, 1, len(model_names)))
    # }
    bar_alpha_dict = {k: v for k, v in zip(model_names, [0, 0, 1])}

    if len(model_names) == 3:
        bar_fill_dict = {k: v for k, v in zip(model_names, [False, False, True])}

    sims = list(zip(cfg.compounds, ["FEFF"] * len(cfg.compounds)))
    if include_vasp:
        sims += [("Ti", "VASP"), ("Cu", "VASP")]

    univ_mses = compute_universal_model_metrics(relative_to_per_compound_mean_model)

    _bar_loc_dict = {k: i for i, k in enumerate(model_names)}

    # def add_alpha(color, alpha=0.1):
    #     rgba = mcolors.to_rgba(color)
    #     return (rgba[0], rgba[1], rgba[2], alpha)

    BAR_CENTER_FACTOR = 1.5
    bars = ax.bar(
        # index,
        np.arange(len(cfg.compounds)) * BAR_CENTER_FACTOR
        + _bar_loc_dict["universal_feff"] * bar_width,
        univ_mses["per_compound"].values(),
        bar_width,
        label="universal-feff-tl",
        # alpha=bar_alpha_dict["universal_feff"],
        # edgecolor="black",
        edgecolor=compound_colors,
        fill=bar_fill_dict["universal_feff"],
        color=compound_colors,
        zorder=3,
        hatch=hatches["universal_feff"],
    )

    # for bar in bars:
    #     bar.set_facecolor(add_alpha(bar.get_facecolor()))

    _model_dict = {
        "per_compound_tl": lambda c, sim_type: Trained_FCModel(
            DataQuery(c, sim_type), name="per_compound_tl"
        ),
        "ft_tl": lambda c, sim_type: Trained_FCModel(
            DataQuery(c, sim_type), name="ft_tl"
        ),
        "universal_feff": lambda c, _: Trained_FCModel(
            DataQuery("ALL", "FEFF"), name="universal_feff"
        ),
        "linreg": lambda c, sim_type: LinReg(DataQuery(c, sim_type)),
        "acsf": lambda c, _: Trained_FCModel(
            DataQuery(c, "ACSF"), name="per_compound_tl"
        ),
        "soap": lambda c, _: Trained_FCModel(
            DataQuery(c, "SOAP"), name="per_compound_tl"
        ),
    }

    for model_name in model_names:

        if model_name == "universal_feff":
            continue
        elif model_name in ["per_compound_tl", "ft_tl"]:
            bar_positions = (
                np.arange(len(sims)) * BAR_CENTER_FACTOR
                + _bar_loc_dict[model_name] * bar_width
            )
            bar_positions[-2:] = bar_positions[-2:] + bar_width
            fc_residues = [
                (
                    _model_dict[model_name](c, sim_type).mse_relative_to_mean_model
                    if relative_to_per_compound_mean_model
                    else _model_dict[model_name](c, sim_type).mse
                )
                for c, sim_type in sims
            ]
        elif model_name in ["linreg"]:
            bar_positions = (
                np.arange(len(sims)) * BAR_CENTER_FACTOR
                + _bar_loc_dict[model_name] * bar_width
            )
            bar_positions[-2:] = bar_positions[-2:] + bar_width
            fc_residues = [
                (
                    _model_dict[model_name](c, sim_type).mse
                    if not relative_to_per_compound_mean_model
                    else MeanModel(DataQuery(c, sim_type)).mse
                    / _model_dict[model_name](c, sim_type).mse
                )
                for c, sim_type in sims
            ]
        elif model_name in ["soap", "acsf"]:
            bar_positions = (
                np.arange(len(cfg.compounds)) * BAR_CENTER_FACTOR
                + _bar_loc_dict[model_name] * bar_width
            )
            fc_residues = [
                (
                    _model_dict[model_name](c, sim_type).mse_relative_to_mean_model
                    if relative_to_per_compound_mean_model
                    else _model_dict[model_name](c, sim_type).mse
                )
                for c, sim_type in zip(
                    cfg.compounds, [model_name.upper()] * len(cfg.compounds)
                )
            ]
        else:
            raise ValueError(f"model_name {model_name} not recognized")

        bars = ax.bar(
            bar_positions,
            fc_residues,
            bar_width,
            color=compound_colors,
            label=model_name,
            fill=bar_fill_dict[model_name],
            # alpha=bar_alpha_dict[model_name],
            # edgecolor="black",
            edgecolor=compound_colors,
            hatch=hatches[model_name],
            zorder=3,
        )

    # for bar in bars:
    #     bar.set_facecolor(add_alpha(bar.get_facecolor()))

    if add_weighted:
        ax.axhline(
            univ_mses["global"],
            color=colors["universal_feff"],
            linestyle="--",
            label="Universal_FEFF_global_MSE",
        )

    if (
        not relative_to_per_compound_mean_model and add_weighted
    ):  # coz weighte mean has no meaning in relative case
        for model_name in model_names:
            fc_models = [
                Trained_FCModel(DataQuery(c, "FEFF"), name=model_name)
                for c in cfg.compounds
            ]
            data_sizes = [len(model.data.test.y) for model in fc_models]
            fc_mse_weighted_mse = sum(
                [model.mse * size for model, size in zip(fc_models, data_sizes)]
            ) / sum(data_sizes)

            ax.axhline(
                fc_mse_weighted_mse,
                color=colors[model_name],
                linestyle="--",
                label=f"{model_name}_weighted_MSE",
            )

    if relative_to_per_compound_mean_model:
        y_label = r"Performance over baseline ($\eta$)"
    else:
        y_label = "MSE"

    ax.set_ylabel(y_label, fontsize=FONTSIZE * 1.2)

    file_name = (
        "per_compound_tl_vs_universal_tl_mlp"
        if not relative_to_per_compound_mean_model
        else "per_compound_tl_vs_universal_tl_relative"
    ) + f"_{len(model_names)}.pdf"

    ax.set_xlabel("Compound", fontsize=FONTSIZE * 1.2, labelpad=-10)
    xticks = np.arange(len(sims)) * BAR_CENTER_FACTOR + bar_width
    if include_vasp:
        # add vertical line before vap
        # ax.axvline(x=xticks[-2] - bar_width * 1.5, color="grey", linestyle="--")
        # add background color just for the vasp region
        ax.axvspan(
            xticks[-2] - bar_width * 1.5,
            # xticks[-1] + bar_width * 1.5,
            # end of fig
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
            # alpha=0.5,
            color="grey",
            # rotation=90,
        )

        # set the tick position of VASPs by one bar to right
        xticks[-2:] = xticks[-2:] + bar_width * 1.5
    ax.set_xticks(xticks)
    # set xlim
    ax.set_xlim(-bar_width * 2, ax.get_xlim()[1] - 3 * bar_width)
    # ax.set_xticks(np.arange(len(sims)) + bar_width)

    ax.set_xticklabels(
        cfg.compounds
        + (
            [
                "Ti\n" + r"{\large VASP}",
                "Cu\n" + r"{\large VASP}",
            ]
            if include_vasp
            else []
        ),
        fontsize=FONTSIZE,
    )

    # # add legends with hatches in gray
    # def hatch_fn(hatch, alpha):
    #     return plt.Rectangle(
    #         (0, 0),
    #         1,
    #         1,
    #         edgecolor="gray",
    #         fc=(0, 0, 0, alpha),
    #         hatch=hatch,
    #     )
    # handles = (
    #     [
    #         hatch_fn(
    #             hatches[model_name],
    #             # bar_alpha_dict[model_name],
    #             0,
    #         )
    #         for model_name in model_names
    #     ],
    # )

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

    name_dict = {
        "universal_feff": "Universal",
        "per_compound_tl": "Expert",
        "ft_tl": "Tuned-Universal",
    }

    legend = ax.legend(
        handles,
        [name_dict[model_name] for model_name in model_names],
        fontsize=FONTSIZE,
        # handletextpad=0.1,
        # labelspacing=0,  # Remove space between legend entries
        # loc="upper right",  # Position the legend above the plot
        # bbox_to_anchor=(0.5, 1.10),  # Adjust the exact position
        # ncol=len(model_names),  # Arrange all labels in one row
        # borderaxespad=0,  # Remove space around the legend
        handlelength=2,
        handleheight=1,
    )
    for text in legend.get_texts():
        text.set_rotation(0)  # Ensure the text is horizontal

    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.savefig(file_name[:-4] + ".png", bbox_inches="tight", dpi=300)
    plt.show()
    return ax


# if __name__ == "__main__":

# plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=True)

import matplotlib as mpl

ASPECT_RATIO = 4 / 3
HEIGHT = 6
WEIGHT = HEIGHT * ASPECT_RATIO
DPI = 300
FONTSIZE = 14
plt.style.use(["default", "science", "tableau-colorblind10"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["axes.labelsize"] = FONTSIZE
mpl.rcParams["xtick.labelsize"] = FONTSIZE
mpl.rcParams["ytick.labelsize"] = FONTSIZE
mpl.rcParams["legend.fontsize"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"

fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=DPI)
plot_universal_tl_vs_per_compound_tl(
    model_names=[
        # "acsf",
        # "soap",
        # "linreg",
        "universal_feff",
        "per_compound_tl",
        "ft_tl",
    ],
    relative_to_per_compound_mean_model=True,
    include_vasp=True,
    ax=ax,
    add_weighted=False,
    FONTSIZE=FONTSIZE,
    # include_linreg=True,
)


# plot_deciles_of_top_predictions(
#     model_name="per_compound_tl",
#     fixed_model=None,
# )

# plot_deciles_of_top_predictions(
#     model_name="ft_tl",
#     fixed_model=None,
# )

# plot_deciles_of_top_predictions(
#     model_name="universal_tl",
#     fixed_model=Trained_FCModel(
#         DataQuery("ALL", "FEFF"), name="universal_tl"
#     ),  # uses same model for all predictions
# )

# plot_deciles_of_top_predictions(
#     model_name="ft_tl",
#     fixed_model=None,
#     compounds=["Ti", "Cu"],
#     simulation_type="VASP",
# )

# %%