# %%
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from omnixas.data import SpectrumType
from omnixas.model.trained_model import ModelMetrics, ModelTag
from omnixas.scripts.plots.scripts import (
    AllExpertMetrics,
    AllTunedMetrics,
    AllUniversalMetrics,
    ExpertEtas,
    TunedEtas,
    UniversalModelEtas,
)


class DecilePlotter:
    @staticmethod
    def plot(
        metrics: Dict[ModelTag, ModelMetrics],
        etas: Dict[ModelTag, float] = None,
        fig=None,
        axs=None,
        fontsize=16,
    ):
        """Plot deciles of predictions for each element/model."""
        plt.style.use(["default", "science"])
        n_deciles = 9
        n_elements = len(metrics)

        if fig is None or axs is None:
            # Set figure DPI higher for PDF export
            fig, axs = plt.subplots(
                n_deciles,
                n_elements,
                figsize=(n_elements * 2, n_deciles * 1.5),
                sharex=True,
                gridspec_kw={"hspace": 0, "wspace": 0},
                dpi=300,  # Higher DPI
            )

            # Set facecolor to white to prevent transparency issues
            fig.patch.set_facecolor("white")

        # Create color map for elements
        cmap = "tab10"
        color_keys = [tag.element + "_" + tag.type for tag in metrics.keys()]
        colors = {e: plt.get_cmap(cmap)(i) for i, e in enumerate(color_keys)}

        # Plot each column (element)
        for tag, ax_col in zip(metrics.keys(), axs.T):
            color = colors[tag.element + "_" + tag.type]
            deciles = metrics[tag].deciles

            # Plot each decile
            for decile_idx, (ax, decile_data) in enumerate(
                zip(ax_col, deciles), start=1
            ):
                # Set background color explicitly
                ax.set_facecolor("white")

                target = decile_data[0]
                pred = decile_data[1]

                # Plot with slightly thicker lines for PDF
                ax.plot(
                    target,
                    linestyle=":",
                    color=color,
                    linewidth=1.0,  # Increased from 0.8
                    label="Ground Truth",
                    solid_capstyle="round",  # Prevent pixel artifacts
                    solid_joinstyle="round",
                )
                ax.plot(
                    pred,
                    linestyle="-",
                    color=color,
                    linewidth=0.5,  # Increased from 0.4
                    label="Predictions",
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

                # Fill between with slightly higher alpha
                ax.fill_between(
                    np.arange(len(target)),
                    target,
                    pred,
                    alpha=0.4,
                    color=color,
                    linewidth=0,
                )

                # Remove spines completely instead of setting visible to False
                ax.spines["top"].set_linewidth(0)
                ax.spines["right"].set_linewidth(0)
                ax.spines["left"].set_linewidth(0)
                ax.spines["bottom"].set_linewidth(0)

                ax.set_xticks([])
                ax.set_yticks([])

                # Set transparent background for the subplot
                ax.patch.set_alpha(0.0)

                # Add decile label on leftmost plots
                if tag == list(metrics.keys())[0]:
                    ax.set_ylabel(
                        r"$\bf{D}_{" + f"{decile_idx}0" + r"}$",
                        rotation=0,
                        fontsize=fontsize * 0.8,
                        labelpad=-6,
                        loc="center",
                        alpha=0.5,
                        color="black",
                    )

            # Add title with eta value if provided
            title = f"{tag.element}"
            if tag.type == "VASP":
                title += " (VASP)"
            if etas is not None:
                title += "\n" + r"$\bf{\eta=}$" + f"{etas[tag]:.2f}"

            ax_col[0].set_title(
                title,
                loc="center",
                fontsize=fontsize,
                color=color,
                y=1.025,
                x=0.5,
            )

        return fig, axs


def main(**kwargs):
    from omnixas.scripts.plots.scripts import (
        AllExpertMetrics,
        AllTunedMetrics,
        AllUniversalMetrics,
        ExpertEtas,
        TunedEtas,
        UniversalModelEtas,
    )

    fig, ax = DecilePlotter.plot(AllExpertMetrics(**kwargs), ExpertEtas(**kwargs))
    fig.savefig("deciles_expert.pdf", dpi=300, bbox_inches="tight")

    fig, ax = DecilePlotter.plot(AllTunedMetrics(**kwargs), TunedEtas(**kwargs))
    fig.savefig("deciles_tuned.pdf", dpi=300, bbox_inches="tight")

    fig, ax = DecilePlotter.plot(
        AllUniversalMetrics(**kwargs), UniversalModelEtas(**kwargs)
    )
    fig.savefig("deciles_universal.pdf", dpi=300, bbox_inches="tight")


def plot_three():
    metrics = AllTunedMetrics()
    etas = TunedEtas()
    min_element, median_element, max_element = "Mn", "V", "Cu"
    metrics = {
        k: v
        for k, v in metrics.items()
        if k.element in [min_element, median_element, max_element]
        and k.type == SpectrumType.FEFF
    }
    etas = {
        k: v
        for k, v in etas.items()
        if k.element in [min_element, median_element, max_element]
        and k.type == SpectrumType.FEFF
    }
    # sort by etas
    sort_order = sorted(etas, key=etas.get, reverse=True)
    metrics = {k: metrics[k] for k in sort_order}
    etas = {k: etas[k] for k in sort_order}
    fig, axs = plt.subplots(
        9,
        3,
        figsize=(3 * 2.5, 9 * 1),
        sharex=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
        dpi=300,
    )
    fig, ax = DecilePlotter.plot(metrics, etas, fig, axs, fontsize=12)
    fig.savefig("three_deciles_tuned.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # plot_three()

# %%
