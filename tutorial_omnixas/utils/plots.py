import warnings
from typing import Dict, Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from omnixas.model.metrics import ModelMetrics, ModelTag
from omnixas.model.trained_model import MeanModel


def plot_line_heatmap(
    data: np.ndarray,
    ax=None,
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    # norm: matplotlib.colors.Normalize = LogNorm(),
    norm: matplotlib.colors.Normalize = LogNorm(
        vmin=1, vmax=100
    ),  # Adjust normalization
    aspect=0.618,  # golden ratio
    x_ticks=None,
    y_ticks=None,
    interpolate: Union[None, int] = None,
):
    """
    Generate a heatmap from multiple lines of data.
    """

    if ax is None:
        ax = plt.gca()

    if interpolate is not None:
        from scipy.interpolate import interp1d

        x = np.arange(data.shape[1])
        x_new = np.linspace(0, data.shape[1] - 1, interpolate)
        f_x = interp1d(x, data)
        data = f_x(x_new)

    # initialize heatmap to zeros
    width = data.shape[1]
    height = width if height == "same" else height

    heatmap = np.zeros((width, height))
    max_val = data.max()
    max_val *= 1.1  # add some padding
    for line in data:
        for x_idx, y_val in enumerate(line):
            y_idx = y_val / max_val * height
            y_idx = y_idx.astype(int)
            y_idx = np.clip(y_idx, 0, height - 1)
            heatmap[y_idx, x_idx] += 1

    colorbar = ax.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="nearest",
    )

    if x_ticks is not None:
        x_ticks_pos = np.linspace(0, width, len(x_ticks))
        colorbar.axes.xaxis.set_ticks(x_ticks_pos, x_ticks)
    if y_ticks is not None:
        y_ticks_pos = np.linspace(0, height, len(y_ticks))
        colorbar.axes.yaxis.set_ticks(y_ticks_pos, y_ticks)

    return colorbar


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
                title += "\n" + r"$\bf{\eta=}$" + f"{etas[tag]:.1f}"

            ax_col[0].set_title(
                title,
                loc="center",
                fontsize=fontsize,
                color=color,
                y=1.025,
                x=0.5,
            )

        return fig, axs


class MSEHistogramPlotter:
    @staticmethod
    def plot_mse_histograms(
        metrics: Dict[ModelTag, ModelMetrics],
        fig=None,
        axs=None,
    ):
        """Plot MSE histograms for each model/element."""

        ASPECT_RATIO = 4 / 3
        HEIGHT = 10
        WIDTH = HEIGHT / ASPECT_RATIO
        DPI = 300
        FONTSIZE = 22

        plt.style.use(["default", "science"])

        cmap = "tab10"

        n_elements = len(metrics)
        COLS = 2
        ROWS = (n_elements + 1) // COLS  # ceiling division

        if fig is None or axs is None:
            fig, axs = plt.subplots(
                ROWS,
                COLS,
                figsize=(WIDTH, HEIGHT),
                dpi=DPI,
                sharex=True,
                sharey=True,
                gridspec_kw={"hspace": 0.05, "wspace": 0.025},
            )

        color_keys = [f"{tag.element}_{tag.type}" for tag in metrics.keys()]
        colormap = plt.get_cmap(cmap)
        colors = {k: colormap(i) for i, k in enumerate(color_keys)}

        all_log_mses = np.concatenate(
            [np.log10(m.mse_per_spectra) for m in metrics.values()]
        )

        bin_min = np.floor(np.quantile(all_log_mses, 0.01))
        bin_max = np.ceil(np.quantile(all_log_mses, 0.99))
        bins = np.linspace(bin_min, bin_max, 30)

        # Plot each histogram
        for i, (tag, ax) in enumerate(zip(metrics.keys(), axs.flatten())):
            element = tag.element
            color = colors[f"{element}_{tag.type}"]

            # Calculate MSE per spectra
            mse_per_spectra = metrics[tag].mse_per_spectra

            # Plot histogram
            ax.hist(
                np.log10(mse_per_spectra),
                density=True,
                # bins=np.linspace(-4, 0, 30),
                bins=bins,
                color=color,
                edgecolor="black",
                zorder=1,
                alpha=0.6,
            )

            # Add element label
            label_text = f"{element}"
            if tag.type == "VASP":
                label_text = f"{element}"

            ax.text(
                0.95,
                0.92,
                label_text,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=FONTSIZE * 0.8,
                color="black",
                fontweight="bold",
                bbox=dict(
                    facecolor=color,
                    alpha=0.2,
                    edgecolor=color,
                ),
            )

            # Add VASP label if needed
            if tag.type == "VASP":
                ax.text(
                    0.91,
                    0.66,
                    "VASP",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=FONTSIZE * 0.45,
                    color="black",
                )

            x_min = -10.2
            x_max = -5.8
            # x_min = bin_min - 0.2
            # x_max = bin_max + 0.2

            warnings.warn("Hardcoded x_min and x_max")

            xticks = np.arange(np.ceil(x_min), np.ceil(x_max), 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.0f}" for x in xticks], fontsize=FONTSIZE * 0.7)
            ax.set_xlim(x_min, x_max)

            warnings.warn("Hardcoded yticks")
            yticks = [0.3, 0.6, 0.9, 1.2]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontsize=FONTSIZE * 0.7)
            ax.set_ylim(None, 1.32)

            ax.tick_params(
                axis="both", which="both", direction="in", top=False, right=False
            )

            ax.axvline(
                np.log10(MeanModel(tag=tag).metrics.mse),
                color="black",
                linestyle=":",
                alpha=0.5,
                label="baseline",
                zorder=0,
            )

        # Clean up empty subplots if any
        for ax in axs.flatten()[len(metrics) :]:
            ax.set_visible(False)

        # Add labels
        axs[2, 0].set_ylabel("Density", fontsize=FONTSIZE)
        for ax in axs[-1, :]:
            ax.set_xlabel(r"$\log_{10}(\text{MSE})$", fontsize=FONTSIZE)

        # Add legend
        axs[-1, 0].legend(fontsize=FONTSIZE * 0.8, loc="center left")

        plt.tight_layout()
        return fig, axs
