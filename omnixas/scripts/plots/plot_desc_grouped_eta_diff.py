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
    if tag.type == "VASP":
        continue
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


# DESC = "OCN_binned"
# DESC = "OS"
DESC = "CN"
for DESC in ["OCN_binned", "OS", "CN"]:
    MIN_COUNT = 5
    FONTSIZE = 20

    filtered_df = df[(df["type"] == "FEFF") & (df["split"] == "test")]

    # Get unique elements
    elements = sorted(filtered_df["element"].unique())
    desc_values = sorted(filtered_df[DESC].unique())

    apply_plot_theme(FONTSIZE)
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(14, 12),
        # sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.2, "hspace": 0.1},
    )
    axes = axes.ravel()

    cmap = "tab10"
    ordered_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
    compound_colors = plt.get_cmap(cmap)(
        np.linspace(0, 1, len(df["element"].unique()) + 2)
    )
    compound_colors = {c: compound_colors[i] for i, c in enumerate(ordered_elements)}

    # Create subplot for each element
    for idx, element in enumerate(ordered_elements):
        ax = axes[idx]
        element_df = filtered_df[filtered_df["element"] == element]

        tag = ModelTag(element=element, type="FEFF", feature="m3gnet", name="expertXAS")
        mean_model = MeanModel(tag=tag)
        normalizer = mean_model.metrics.median_of_mse_per_spectra

        delta_etas = []
        valid_desc = []

        for desc_value in desc_values:
            mask = element_df[DESC] == desc_value
            group = element_df[mask]

            if len(group) > MIN_COUNT:
                expert_median = group["expert_mse"].median()
                tuned_median = group["tuned_mse"].median()

                # baseline should be based on train mean in same categoy
                train_group = df[df["type"] == "FEFF"][
                    (df["element"] == element) & (df["split"] == "train")
                ]
                train_group = train_group[train_group[DESC] == desc_value]
                spectras = train_group["spectras"].to_list()
                group_mean = np.mean(spectras, axis=0)
                grouped_residue = group_mean - group["spectras"].to_list()
                grouped_mses = np.mean(grouped_residue**2, axis=1)
                group_normalizer = np.median(grouped_mses)

                eta_expert = group_normalizer / expert_median
                eta_tuned = group_normalizer / tuned_median
                delta_eta = eta_tuned - eta_expert

                # eta_expert = normalizer / expert_median
                # eta_tuned = normalizer / tuned_median
                # delta_eta = eta_tuned - eta_expert

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
                0.05,
                0.95,
                element,
                transform=ax.transAxes,
                fontsize=FONTSIZE,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    facecolor=compound_colors[element], alpha=0.5, edgecolor="white"
                ),
            )

            desc_label = DESC if DESC != "OCN_binned" else "OCN"
            if idx >= 4:
                ax.set_xlabel(desc_label)
                ax.tick_params(axis="x", labelbottom=True)
            if idx in [0, 4]:
                # ax.set_ylabel(r"$\eta_T - \eta_E$")
                ax.set_ylabel(
                    r"$\eta^{(" + desc_label + ")}_T - \eta^{(" + desc_label + ")}_E$"
                )
            ax.grid(True, alpha=0.2)

            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.set_ylim(None, ax.get_ylim()[1] * 1.35)

            max_abs = max(abs(min(delta_etas)), abs(max(delta_etas)))

    for idx in range(len(elements), 16):
        axes[idx].remove()

    plt.tight_layout()
    plt.savefig(f"delta_eta_vs_{DESC}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# %%
