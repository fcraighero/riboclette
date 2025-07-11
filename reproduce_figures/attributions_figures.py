import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools
    import os

    import config
    import marimo as mo
    import marsilea as ma
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots
    import utils
    from adjustText import adjust_text
    from pyhere import here
    from scipy.stats import pearsonr, spearmanr
    return (
        adjust_text,
        config,
        here,
        itertools,
        ma,
        mo,
        mpl,
        np,
        os,
        pd,
        pearsonr,
        plt,
        scienceplots,
        spearmanr,
        utils,
    )


@app.cell
def _(here, mo):
    ALLOW_OVERWRITE = mo.cli_args().get("overwrite") or True
    OUT_DIRPATH = mo.cli_args().get("output_dirpath")
    PLOTTING_DIRPATH = mo.cli_args().get("plotting_dirpath") or here(
        "data", "results", "plotting"
    )
    return ALLOW_OVERWRITE, OUT_DIRPATH, PLOTTING_DIRPATH


@app.cell
def _(np):
    def global_attr_plot(
        ax,
        data,
        title: str,
        xlim=(-10, 10),
        xlabel=False,
        ylabel=False,
        yticks=False,
        xticks=False,
        ylogscale=False,
    ):
        data = np.array(data)
        data = data[(data > xlim[0]) & (data < xlim[1])]
        unique_values, counts = np.unique(data, return_counts=True)
        ax.bar(unique_values, counts / np.sum(counts), color="#6291B0")
        ax.axvspan(-5.4, 4.4, color="#6291B0", alpha=0.2)
        if ylogscale:
            ax.set_yscale("log")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        if yticks:
            ax.set_yticks([0, 0.05, 0.1])
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel("Codon Distance from A-site")
        if ylabel:
            ax.set_ylabel("Frequency")
    return (global_attr_plot,)


@app.cell
def _(adjust_text, config, itertools, np, pearsonr):
    def plot_condition(
        ax,
        condition,
        stall_mean_sorted,
        attr_mean_sorted,
        codons_to_depr,
        ctrl_tagged_codons,
        colors_depr,
        texts,
        annotation_kwargs=dict(fontsize=5),
    ):
        for codon in stall_mean_sorted[condition]:
            x = stall_mean_sorted[condition][codon]
            y = attr_mean_sorted[condition][codon]

            # Determine the color and label for the codon
            if len(codons_to_depr[codon]) != 0:
                ax.scatter(
                    x, y, label=codon, color=colors_depr[codons_to_depr[codon][0]]
                )
                if condition in ["ILE", "LEU_ILE"] and codon in ["ATC", "ATT"]:
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
                elif condition in ["VAL", "LEU_ILE_VAL"] and codon in [
                    "GTC",
                    "GTT",
                    "GTA",
                    "GTG",
                ]:
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
            elif codon in ctrl_tagged_codons:
                ax.scatter(x, y, label=codon, color=colors_depr["CTRL"], alpha=1)
                if condition == "CTRL":
                    texts.append(ax.text(x, y, codon, **annotation_kwargs))
            else:
                ax.scatter(x, y, label=codon, color="black", linewidth=0, alpha=0.5)

            ax.set_ylim(-0.3, +0.6)
        return texts

    def global_stalling(
        axs,
        genetic_code,
        condition_codon_stall_mean_sorted,
        condition_codon_attr_peaks_mean_sorted,
        condition_codon_attr_full_mean_sorted,
        mode="peaks",
        conditions=["CTRL", "ILE", "VAL"],
    ):

        # Get the codons for each deprivation condition
        deprivation_conditions = ["Ile", "Leu", "Val", "CTRL"]
        depr_codons = {}

        for condition in deprivation_conditions:
            amino_acids = condition.split("_")
            codons_dep_cond = []
            for amino_acid in amino_acids:
                df_aa = genetic_code[genetic_code["AminoAcid"] == amino_acid]
                codons_dep_cond += df_aa["Codon"].tolist()

            depr_codons[condition.upper()] = codons_dep_cond
        depr_codons["LEU_ILE"] = []
        depr_codons["LEU_ILE_VAL"] = []

        # Ensure CTRL is the first key
        depr_codons = {
            k: depr_codons[k]
            for k in ["CTRL", "ILE", "LEU", "VAL", "LEU_ILE", "LEU_ILE_VAL"]
        }

        id_to_codon = {
            idx: "".join(el)
            for idx, el in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
        }
        codons_to_depr = {
            codon: [depr for depr, codons in depr_codons.items() if codon in codons]
            for codon in id_to_codon.values()
        }
        ctrl_tagged_codons = ["GAC", "GAA", "GAT"]

        attr_mean_sorted = (
            condition_codon_attr_peaks_mean_sorted
            if mode == "peaks"
            else condition_codon_attr_full_mean_sorted
        )

        for i, condition in enumerate(conditions):
            texts = []
            texts = plot_condition(
                axs[i],
                condition,
                condition_codon_stall_mean_sorted,
                attr_mean_sorted,
                codons_to_depr,
                ctrl_tagged_codons,
                config.COND_COL,
                texts,
            )

            # Adjust the text
            adjust_text(
                texts,
                ax=axs[i],
                expand_points=(1.2, 1.2),
                expand_text=(1.2, 1.2),
                force_text=(0.5, 0.5),
            )

            # Calculate PCC
            x = [
                condition_codon_stall_mean_sorted[condition][codon]
                for codon in condition_codon_stall_mean_sorted[condition]
            ]
            y = [
                attr_mean_sorted[condition][codon]
                for codon in condition_codon_stall_mean_sorted[condition]
            ]
            corr, _ = pearsonr(x, y)

            # Fit a line to the data
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs[i].plot(x, p(x), "r--", color="black", alpha=0.5)

            # axs[i].set_xlabel('Mean Ribosome Counts')
            if i == 0:
                axs[i].set_ylabel("Mean Attribution", labelpad=-2)
            c_text = config.CONDITIONS_FIXNAME[condition]
            axs[i].set_title(f"{c_text} (PCC: {corr:.2f})")
            axs[i].tick_params(axis="both", which="major")
    return global_stalling, plot_condition


@app.cell
def _(config, ma):
    def topk_attributions(
        data, genetic_code, width: float, height=float, fontsizes: list[int] = [5, 6, 7]
    ):

        data = data.rename(columns=config.CONDITIONS_FIXNAME)[
            config.CONDITIONS_FIXNAME.values()
        ]
        h = ma.Heatmap(
            data.T,
            width=width,
            height=height,
            cmap="RdBu_r",
            cbar_kws=dict(
                title="Normalized Mean Attribution",
                orientation="horizontal",
                height=1.5,
                fontsize=config.FSS,
                width=12,
                title_fontproperties=dict(weight="normal", size=config.FSM),
            ),
        )
        AA = config.AMINO_ACID_MAP.keys()
        h.group_cols(genetic_code.AminoAcid, spacing=0.002, order=AA)
        h.add_top(
            ma.plotter.Chunk(
                [config.AMINO_ACID_MAP[a] for a in AA],
                [config.AMINO_ACID_COLORS[a] for a in AA],
            ),
            pad=0.025,
        )
        h.add_bottom(ma.plotter.Labels(data.index, rotation=45), name="Codon")
        h.add_left(ma.plotter.Labels(data.columns, align="center"))
        h.group_rows([1, 0, 0, 0, 0, 0], spacing=0.03, order=[1, 0])
        h.add_legends("bottom")
        h.add_title(
            "Codon-wise mean attribution in high stalling positions",
            fontsize=config.FSB,
        )
        h.render()

        return h
    return (topk_attributions,)


@app.cell
def _(mo):
    mo.md(r"""# Attributions panel""")
    return


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    PLOTTING_DIRPATH,
    config,
    global_attr_plot,
    global_stalling,
    here,
    mpl,
    np,
    os,
    pd,
    plt,
    utils,
):
    def _():
        corrected_width = config.TEXTWIDTH_CM + 3.75
        aspect_ratio = 6

        fig = plt.figure(
            figsize=(
                corrected_width * config.CM_TO_INCH,
                corrected_width / aspect_ratio * config.CM_TO_INCH,
            )
        )
        gs = fig.add_gridspec(nrows=1, ncols=5, wspace=0.6, hspace=3)
        sub_gs = gs[0, :2].subgridspec(1, 2, wspace=0.1)
        ax = fig.add_subplot(sub_gs[0, 0])
        ax.text(-0.3, 1.1, "a", transform=ax.transAxes, fontsize=8)
        ctrl_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_True_True.npz"))[
            "CTRL"
        ]
        dd_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_False_True.npz"))
        global_attr_plot(
            ax, ctrl_data, title="Control", ylabel=True, yticks=True, xticks=True
        )
        ax = fig.add_subplot(sub_gs[0, 1], sharex=ax, sharey=ax)
        global_attr_plot(
            ax,
            np.concatenate([dd_data[k] for k in dd_data.keys() if k != "CTRL"]),
            title="Deprivation Difference",
            yticks=False,
            xticks=True,
        )
        plt.tick_params("y", labelleft=False)
        subfig_coords = sub_gs.get_grid_positions(fig)
        ax.text(
            subfig_coords[2][1] - 0.02,
            subfig_coords[0][0] - 0.22,
            "Codon Distance from A-site",
            transform=fig.transFigure,
            ha="center",
            fontsize=config.FSM,
        )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#6291B0", alpha=0.2, label="Ribosome protected fragment")
        ]
        ax.legend(
            loc="center",
            bbox_to_anchor=(subfig_coords[2][1] - 0.02, -0.2),
            handles=legend_elements,
            frameon=False,
            bbox_transform=fig.transFigure,
        )

        sub_gs = gs[0, 2:].subgridspec(1, 3, wspace=0.1)
        axs = [
            fig.add_subplot(sub_gs[row, col]) for row in range(1) for col in range(0, 3)
        ]
        _genetic_code = pd.read_csv(here("data", "data", "genetic_code.csv"))
        condition_codon_attr_full_mean_sorted = pd.read_csv(
            os.path.join(PLOTTING_DIRPATH, "condition_codon_attr_full_mean_sorted.zip"),
            index_col=0,
        ).to_dict()
        condition_codon_attr_peaks_mean_sorted = pd.read_csv(
            os.path.join(
                PLOTTING_DIRPATH, "condition_codon_attr_peaks_mean_sorted.zip"
            ),
            index_col=0,
        ).to_dict()
        condition_codon_stall_mean_sorted = pd.read_csv(
            os.path.join(PLOTTING_DIRPATH, "condition_codon_stall_mean_sorted.zip"),
            index_col=0,
        ).to_dict()
        global_stalling(
            axs,
            genetic_code=_genetic_code,
            condition_codon_attr_full_mean_sorted=condition_codon_attr_full_mean_sorted,
            condition_codon_attr_peaks_mean_sorted=condition_codon_attr_peaks_mean_sorted,
            condition_codon_stall_mean_sorted=condition_codon_stall_mean_sorted,
        )
        axs[0].text(-0.3, 1.1, "b", transform=axs[0].transAxes, fontsize=8)
        axs[0].set_xlabel(r"Mean RPF Counts")
        axs[1].set_xlabel("Mean \u0394RPF Counts")
        axs[2].set_xlabel("Mean \u0394RPF Counts")
        for ax in axs[1:]:
            ax.set_yticklabels([])
        legend_elements = [
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="CTRL",
                markerfacecolor=config.COND_COL["CTRL"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="ILE",
                markerfacecolor=config.COND_COL["ILE"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="LEU",
                markerfacecolor=config.COND_COL["LEU"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="VAL",
                markerfacecolor=config.COND_COL["VAL"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Other",
                markerfacecolor="black",
                markersize=5,
            ),
        ]
        axs[1].legend(
            handles=legend_elements,
            loc="center",
            ncols=5,
            bbox_to_anchor=(0.5, -0.4),
            frameon=False,
        )

        fig.align_xlabels()

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "attr_hist.svg")
        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)

        plt.show()

    with plt.style.context(
        ["grid", "nature", "no-latex"]
    ), utils.journal_plotting_ctx():
        _()
    return


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    PLOTTING_DIRPATH,
    config,
    here,
    os,
    pd,
    plt,
    topk_attributions,
    utils,
):
    def _():
        data = pd.read_csv(
            os.path.join(PLOTTING_DIRPATH, "topk_attr_cond_wise.zip"),
            index_col=0,
        )
        genetic_code = pd.read_csv(config.GENCODE_FPATH, index_col=0)
        genetic_code = genetic_code.set_index("Codon").drop(index=["TAA", "TAG", "TGA"])

        print("Correlation between salient codons by condition")
        print(pd.DataFrame(data.loc[genetic_code.index].corr()))
        f = topk_attributions(
            data=data.loc[genetic_code.index],
            genetic_code=genetic_code,
            width=(config.TEXTWIDTH_CM - 1) * config.CM_TO_INCH,
            height=1.8 * config.CM_TO_INCH,
        )
        plt.text(0.05, 0.875, "c", transform=f.figure.transFigure, fontsize=8)
        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "corr_attr_stall.svg")

        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            f.figure.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(
        ["grid", "nature", "no-latex"]
    ), utils.journal_plotting_ctx():
        _()
    return


@app.cell
def _(mo):
    mo.md(r"""# Supplementary Figures""")
    return


@app.cell
def _(OUT_DIRPATH, PLOTTING_DIRPATH, global_attr_plot, here, np, os, plt):
    def _():
        _, axs = plt.subplots(2,2, figsize=(10, 8))
        axs = np.array(axs).flatten()
    
        ctrl_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_True_True.npz"))[
            "CTRL"
        ]
        global_attr_plot(
            axs[0], ctrl_data, title="Control", ylabel=True, yticks=True, xticks=True
        )
        dd_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_False_True.npz"))
        global_attr_plot(
            axs[1],
            np.concatenate([dd_data[k] for k in dd_data.keys() if k != "CTRL"]),
            title="Deprivation Difference",
            yticks=False,
            xticks=True,
        )
        axs[1].twinx().set_ylabel('A-site = peak', rotation=-90, labelpad=18)
    
        ctrl_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_True_False.npz"))[
            "CTRL"
        ]
        global_attr_plot(
            axs[2], ctrl_data, title="Control", ylabel=True, yticks=True, xticks=True
        )
    
        dd_data = np.load(os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_False_False.npz"))
        global_attr_plot(
            axs[3],
            np.concatenate([dd_data[k] for k in dd_data.keys() if k != "CTRL"]),
            title="Deprivation Difference",
            yticks=False,
            xticks=True,
        )
        axs[3].twinx().set_ylabel('A-site = fast codon position', rotation=-90, labelpad=18)

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "hist_attr_peak_notpeak.png")
    
        plt.savefig(output_fpath, dpi=400, bbox_inches='tight')
    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Attribution vs RPF (supp)""")
    return


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    PLOTTING_DIRPATH,
    config,
    global_stalling,
    here,
    mpl,
    os,
    pd,
    plt,
    utils,
):
    def _():
        corrected_width = config.TEXTWIDTH_CM * config.CM_TO_INCH * 3 / 5
        aspect_ratio = 2.5
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(corrected_width, corrected_width / aspect_ratio),
            constrained_layout=True,
        )
        _genetic_code = pd.read_csv(here("data", "data", "genetic_code.csv"))
        condition_codon_attr_full_mean_sorted = pd.read_csv(
            os.path.join(PLOTTING_DIRPATH, "condition_codon_attr_full_mean_sorted.zip"),
            index_col=0,
        ).to_dict()
        condition_codon_attr_peaks_mean_sorted = pd.read_csv(
            os.path.join(
                PLOTTING_DIRPATH, "condition_codon_attr_peaks_mean_sorted.zip"
            ),
            index_col=0,
        ).to_dict()
        condition_codon_stall_mean_sorted = pd.read_csv(
            os.path.join(PLOTTING_DIRPATH, "condition_codon_stall_mean_sorted.zip"),
            index_col=0,
        ).to_dict()
        global_stalling(
            axs,
            genetic_code=_genetic_code,
            condition_codon_attr_full_mean_sorted=condition_codon_attr_full_mean_sorted,
            condition_codon_attr_peaks_mean_sorted=condition_codon_attr_peaks_mean_sorted,
            condition_codon_stall_mean_sorted=condition_codon_stall_mean_sorted,
            conditions=["LEU", "LEU_ILE", "LEU_ILE_VAL"],
        )
        axs[0].set_xlabel(r"Mean RPF Counts")
        axs[1].set_xlabel("Mean \u0394RPF Counts")
        axs[2].set_xlabel("Mean \u0394RPF Counts")
        for ax in axs[1:]:
            ax.set_yticklabels([])
        legend_elements = [
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="CTRL",
                markerfacecolor=config.COND_COL["CTRL"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="ILE",
                markerfacecolor=config.COND_COL["ILE"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="LEU",
                markerfacecolor=config.COND_COL["LEU"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="VAL",
                markerfacecolor=config.COND_COL["VAL"],
                markersize=5,
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Other",
                markerfacecolor="black",
                markersize=5,
            ),
        ]
        axs[1].legend(
            handles=legend_elements,
            loc="center",
            ncols=5,
            bbox_to_anchor=(0.5, -0.4),
            frameon=False,
        )
        output_dirpath = OUT_DIRPATH or here(
            "data",
            "results",
            "figures",
        )
        output_fpath = os.path.join(output_dirpath, "supp_corr_attr_stall.svg")
        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(
        ["grid", "nature", "no-latex"]
    ), utils.journal_plotting_ctx():
        _()
    return


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    PLOTTING_DIRPATH,
    config,
    global_attr_plot,
    here,
    np,
    os,
    plt,
    utils,
):
    def _():
        corrected_width = config.TEXTWIDTH_CM - 5
        aspect_ratio = 4

        fig, axes = plt.subplots(
            ncols=5,
            sharey=True,
            figsize=(
                corrected_width * config.CM_TO_INCH,
                corrected_width / aspect_ratio * config.CM_TO_INCH,
            ),
            constrained_layout=True,
        )
        cond_dd_data = np.load(
            os.path.join(PLOTTING_DIRPATH, "globl_attr_plot_False_True.npz")
        )
        cond_iter = config.CONDITIONS_FIXNAME.copy()
        cond_iter.pop("CTRL", None)
        for i, (cond, cond_fix) in enumerate(cond_iter.items()):
            ax = fig.add_subplot(axes[i])
            global_attr_plot(
                ax, cond_dd_data[cond], title=cond_fix, yticks=False, xticks=True
            )

        axes[2].set_xlabel("Codon Distance from A-site")
        axes[0].set_ylabel("Frequency")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#6291B0", alpha=0.2, label="Ribosome protected fragment"),
        ]
        fig.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.05),
            handles=legend_elements,
            frameon=False,
        )

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "supp_attr_hist.pdf")
        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()

    with plt.style.context(
        ["grid", "nature", "no-latex"]
    ), utils.journal_plotting_ctx():
        _()
    return


if __name__ == "__main__":
    app.run()
