# %%
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from omnixas.model.trained_model import MeanModel, ModelTag
from omnixas.scripts.plots.scripts import (
    AllExpertMetrics,
    AllTunedMetrics,
)

# %%

all_expert_metrics = AllExpertMetrics()
all_tuned_metrics = AllTunedMetrics()

# %%

directory = "/Users/bnl/Downloads/dev/aimm/ai_for_xas/dataset/material_id_and_site"
df_all = pd.read_csv(
    "/Users/bnl/Downloads/dev/aimm/ai_for_xas/dataset/omnixas_v1/metrics/umap_data/umap_descriptors.csv"
)

# %%

df_combined = pd.DataFrame()
for tag in all_expert_metrics.keys():
    for split in ["train", "test", "val"]:
        file_name = f"{directory}/{tag.element}_{tag.type}_{split}.txt"
        df = pd.read_csv(file_name, header=None)
        df.rename(columns={0: "id_site"}, inplace=True)
        df["element"] = tag.element
        df["ids"] = df["id_site"].str.split("_").str[0]
        df["sites"] = df["id_site"].str.split("_").str[1]
        df["type"] = tag.type
        tunedUnviersal_tag = ModelTag(
            element=tag.element, type=tag.type, name="tunedUniversalXAS"
        )
        df["split"] = split
        if split == "test":
            df["expert_mse"] = all_expert_metrics[tag].mse_per_spectra
            df["tuned_mse"] = all_tuned_metrics[tunedUnviersal_tag].mse_per_spectra

            tag = ModelTag(
                element=tag.element, type=tag.type, feature="m3gnet", name="expertXAS"
            )
            mean_model = MeanModel(tag=tag)
            normalizer = mean_model.metrics.median_of_mse_per_spectra

            diff = df["expert_mse"] - df["tuned_mse"]
            df["improvement"] = diff / normalizer * 100

        else:
            df["expert_mse"] = 0
            df["tuned_mse"] = 0
            df["improvement"] = 0

        df["sites"] = df["sites"].astype(int)
        df = df.merge(df_all, left_on=["ids", "sites"], right_on=["ids", "sites"])
        df["spectras"] = df["spectras"].apply(lambda x: np.array(literal_eval(x)))
        df_combined = pd.concat([df_combined, df])
df = df_combined
df = df.dropna()
df.loc[:, "OS"] = df["OS"].astype(int)
df.loc[:, "CN"] = df["CN"].astype(int)

df = df.drop(
    columns=[
        "features",
        # "spectras",
        "umap_s0",
        "umap_s1",
        "umap_f0",
        "umap_f1",
    ],
)
df["OCN_binned"] = df["OCN"].apply(lambda x: round(x * 2) / 2)
df.to_csv("data.csv", index=False)
# df = pd.read_csv("data.csv")


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


for DESC in ["OCN_binned", "OS", "CN"]:
    MIN_COUNT = 10
    FONTSIZE = 20

    apply_plot_theme(FONTSIZE)
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 6),
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )
    axes = axes.ravel()

    cmap = "tab10"

    ordered_elements = {
        "Ti": 0,
        "V": 1,
        "Cr": 2,
        "Mn": 3,
        "Fe": 5,
        "Co": 6,
        "Ni": 7,
        "Cu": 8,
        "Ti_VASP": 4,
        "Cu_VASP": 9,
    }

    compound_colors = plt.get_cmap(cmap)(
        # np.linspace(0, 1, len(df["element"].unique()) + 2)
        np.linspace(0, 1, len(ordered_elements))
    )
    compound_colors = {c: compound_colors[i] for i, c in enumerate(ordered_elements)}

    for element, idx in ordered_elements.items():
        ax = axes[idx]
        sim_type = "VASP" if "VASP" in element else "FEFF"
        element_name = element if "VASP" not in element else element.split("_")[0]

        tag = ModelTag(
            element=element_name,
            type=sim_type,
            feature="m3gnet",
            name="expertXAS",
        )
        mean_model = MeanModel(tag=tag)
        normalizer = mean_model.metrics.median_of_mse_per_spectra

        delta_etas = []
        valid_desc = []

        element_df = df[
            (df["element"] == element_name)
            & (df["type"] == sim_type)
            & (df["split"] == "test")
        ]
        for desc_value in element_df[DESC].unique():
            mask = element_df[DESC] == desc_value
            group = element_df[mask]

            if len(group) > MIN_COUNT:
                expert_median = group["expert_mse"].median()
                tuned_median = group["tuned_mse"].median()

                # baseline should be based on train mean in same categoy
                train_group = df[
                    (df["type"] == sim_type)
                    & (df["element"] == element_name)
                    & (df["split"] == "train")
                ]
                train_group = train_group[train_group[DESC] == desc_value]
                spectras = train_group["spectras"].to_list()
                group_mean = np.mean(spectras, axis=0)
                grouped_residue = group_mean - group["spectras"].to_list()
                grouped_mses = np.mean(grouped_residue**2, axis=1)
                group_normalizer = np.median(grouped_mses)

                eta_expert = group_normalizer / expert_median
                eta_tuned = group_normalizer / tuned_median

                # delta_eta = eta_tuned - eta_expert

                delta_eta = (eta_tuned - eta_expert) / eta_expert * 100

                # delta_eta = eta_expert

                delta_etas.append(delta_eta)
                valid_desc.append(desc_value)

        if valid_desc:
            bars = ax.bar(valid_desc, delta_etas, width=0.5, alpha=0.875)

            # Color bars based on positive/negative values
            for bar, value in zip(bars, delta_etas):
                if value >= 0:
                    bar.set_color("royalblue")
                else:
                    bar.set_color("lightcoral")

            ax.text(
                0.06,
                0.95,
                element_name,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=FONTSIZE * 1,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    facecolor=compound_colors[element], alpha=0.5, edgecolor="white"
                ),
            )
            # if vasp add small text below the text above writing "VASP"
            if sim_type == "VASP":
                ax.text(
                    0.045,
                    0.8,
                    "VASP",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=FONTSIZE * 0.5,
                    verticalalignment="top",
                    horizontalalignment="left",
                    # bbox=dict(
                    #     facecolor=compound_colors[element], alpha=0.5, edgecolor="white"
                    # ),
                )

            desc_label = DESC if DESC != "OCN_binned" else "OCN"

            ax.tick_params(
                axis="both",
                which="both",
                right=False,
                top=False,
                left=True,
                bottom=True,
            )

            if idx >= 5:
                ax.set_xlabel(desc_label + r"$(x)$")
                ax.set_xticks([2, 3, 4, 5, 6])
                ax.set_xticklabels(["2", "3", "4", "5", "6"])
                plt.setp(ax.get_xticklabels(), visible=True)  # Corrected method name
            else:
                plt.setp(
                    ax.get_xticklabels(), visible=False
                )  # Optionally hide labels for other subplots

            if idx in [0, 5]:
                ax.set_ylabel(
                    # r"$\tilde{\eta}^{(\scriptstyle " + desc_label + ")}\,[\%]$"
                    r"$\tilde{\eta}^{(x)}\,[\%]$"
                )

            ax.grid(
                which="major",
                axis="both",
                color="gray",
                linestyle="-",
                linewidth=0.5,
                alpha=0.2,
            )

            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"delta_eta_vs_{DESC}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# %%
