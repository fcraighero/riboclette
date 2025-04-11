import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    import config
    import h5py
    import marimo as mo
    import matplotlib.font_manager as fm
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots
    import seaborn as sns
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyhere import here

    import utils

    return (
        AnchoredSizeBar,
        config,
        fm,
        gridspec,
        h5py,
        here,
        mo,
        np,
        os,
        pd,
        plt,
        scienceplots,
        sns,
        utils,
    )


@app.cell
def _(mo):
    ALLOW_OVERWRITE = mo.cli_args().get("overwrite") or True
    OUT_DIRPATH = mo.cli_args().get("output_dirpath")
    return ALLOW_OVERWRITE, OUT_DIRPATH


@app.cell
def _(config, here):
    H5PY_FPATH = here("data", "results", "interpretability", config.ATTR_FNAMES[2])
    return (H5PY_FPATH,)


@app.cell
def _(config, utils):
    ensembl_df = utils.read_ensembl(config.ENSEMBL_FPATH)
    return (ensembl_df,)


@app.cell
def _(ensembl_df):
    ensembl_df.query('symbol == "Col1a2"')
    return


@app.cell
def _(H5PY_FPATH, h5py, np):
    with h5py.File(H5PY_FPATH, "r") as _f:
        transcripts = _f["transcript"][:].astype(str)
        idxs = np.where(
            np.isin(transcripts, ["ENSMUST00000141483.7", "ENSMUST00000031668.9"])
        )[0]
        print(idxs)
        print(_f["condition"][idxs])
        print(idxs[3])
    return idxs, transcripts


@app.cell
def _(H5PY_FPATH, h5py):
    seq_id = 7841
    with h5py.File(H5PY_FPATH, "r") as _f:
        print(_f["condition"][seq_id])
        y_true_ctrl = _f["y_true_ctrl"][seq_id]
        n_codons = len(y_true_ctrl)
        y_true_ctrl = y_true_ctrl[136 : 136 + 60]
        y_true_ctrl = np.where(y_true_ctrl == 999, np.nan, y_true_ctrl)
        y_pred_ctrl = _f["y_pred_ctrl"][seq_id][136 : 136 + 60]
        y_true_depr = _f["y_true_depr"][seq_id][136 : 136 + 60]
        y_pred_depr = _f["y_pred_depr"][seq_id][136 : 136 + 60]
        y_true_full = y_true_depr + y_true_ctrl
        y_pred_full = y_pred_depr + y_pred_ctrl
        attr_ctrl = _f["attr_ctrl"][seq_id]
        attr_depr = _f["attr_depr"][seq_id]
    return (
        attr_ctrl,
        attr_depr,
        n_codons,
        seq_id,
        y_pred_ctrl,
        y_pred_depr,
        y_pred_full,
        y_true_ctrl,
        y_true_depr,
        y_true_full,
    )


@app.cell
def _(attr_ctrl, attr_depr, n_codons):
    codon_id = 166
    sel_attr_ctrl = attr_ctrl.reshape(n_codons, -1)[codon_id][136 : 136 + 60]
    sel_attr_depr = attr_depr.reshape(n_codons, -1)[codon_id][136 : 136 + 60]
    return codon_id, sel_attr_ctrl, sel_attr_depr


@app.cell
def _():
    n_codons_1 = 60
    return (n_codons_1,)


@app.cell
def _():
    from matplotlib import transforms
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        AuxTransformBox,
        DrawingArea,
        TextArea,
        VPacker,
    )
    from matplotlib.patches import ArrowStyle, FancyArrowPatch, PathPatch, Rectangle
    from matplotlib.text import TextPath

    class CustomAnchoredSizeBar(AnchoredOffsetbox):
        def __init__(
            self,
            transform,
            size,
            label,
            loc,
            pad=0.1,
            borderpad=0.1,
            sep=2,
            frameon=True,
            size_vertical=0,
            color="black",
            label_top=False,
            fontproperties=None,
            fill_bar=None,
            **kwargs,
        ):

            if fill_bar is None:
                fill_bar = size_vertical > 0

            self.size_bar = AuxTransformBox(transform)
            self.size_bar.add_artist(
                Rectangle(
                    (0, 0),
                    size,
                    size_vertical,
                    fill=fill_bar,
                    facecolor=color,
                    edgecolor=color,
                )
            )

            if fontproperties is None and "prop" in kwargs:
                fontproperties = kwargs.pop("prop")

            if fontproperties is None:
                textprops = {"color": color}
            else:
                textprops = {"color": color, "fontproperties": fontproperties}
            textprops["ha"] = "center"

            self.txt_label = TextArea(label, textprops=textprops)

            if label_top:
                _box_children = [self.txt_label, self.size_bar]
            else:
                _box_children = [self.size_bar, self.txt_label]

            self._box = VPacker(children=_box_children, align="center", pad=0, sep=sep)

            super().__init__(
                loc,
                pad=pad,
                borderpad=borderpad,
                child=self._box,
                prop=fontproperties,
                frameon=frameon,
                **kwargs,
            )

    return (
        AnchoredOffsetbox,
        ArrowStyle,
        AuxTransformBox,
        CustomAnchoredSizeBar,
        DrawingArea,
        FancyArrowPatch,
        PathPatch,
        Rectangle,
        TextArea,
        TextPath,
        VPacker,
        transforms,
    )


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    config,
    gridspec,
    here,
    n_codons_1,
    np,
    os,
    plt,
    sel_attr_ctrl,
    sel_attr_depr,
    sns,
    utils,
    y_pred_ctrl,
    y_pred_depr,
    y_pred_full,
    y_true_ctrl,
    y_true_depr,
    y_true_full,
):
    with utils.journal_plotting_ctx():
        fig = plt.figure(
            figsize=(9 * config.CM_TO_INCH, 3.9 * config.CM_TO_INCH),
            constrained_layout=True,
        )
        out_gs = fig.add_gridspec(1, 1, bottom=0.85, top=1)

        def apply_axis_transform(ax):
            ax.set_xticks([n_codons_1 // 2, n_codons_1], [166, 196])
            for axis in ["left", "bottom"]:
                ax.spines[axis].set_linewidth(0.5)
            ax.spines["bottom"].set_position(("data", 0))

        attr_max = np.nanmax(np.concatenate((sel_attr_ctrl, sel_attr_depr))) + 0.05
        attr_min = np.nanmin(np.concatenate((sel_attr_ctrl, sel_attr_depr))) + 0.05
        xlim = (-1, n_codons_1 + 1)
        y_max = max(np.nanmax(y_true_full), np.nanmax(y_pred_full)) + 0.2
        out_gs = fig.add_gridspec(3, 3, bottom=0, top=0.75)
        ax = fig.add_subplot(out_gs[0, 0])
        apply_axis_transform(ax)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim((-1, y_max))
        ax.set_xlim(xlim)
        ax.set_ylabel("True")
        ax.set_title("Control", fontsize=config.FSS)
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_true_ctrl, 0), color=config.TRUE_COL
        )
        ax = fig.add_subplot(out_gs[0, 1])
        apply_axis_transform(ax)
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_true_depr, 0), color=config.TRUE_COL
        )
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim((-1, y_max))
        ax.set_xlim(xlim)
        ax.set_title("Depr. Difference", fontsize=config.FSS)
        ax.set_yticklabels([])
        ax = fig.add_subplot(out_gs[0, 2])
        apply_axis_transform(ax)
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_true_full, 0), color=config.TRUE_COL
        )
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1, y_max)
        ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.set_title("Depr. Counts", fontsize=config.FSS)
        y_max = 1.1
        ax = fig.add_subplot(out_gs[1, 0])
        apply_axis_transform(ax)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim((-1, y_max))
        ax.set_xlim(xlim)
        ax.set_ylabel("Predicted")
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_pred_ctrl, 0), color=config.PRED_COL
        )
        ax.annotate(
            "explained A-site",
            xy=(n_codons_1 // 2, y_pred_ctrl[n_codons_1 // 2]),
            xycoords="data",
            xytext=(-5, 12.5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", facecolor="none", edgecolor="none", pad=0),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.5,
                connectionstyle="angle,angleA=0,angleB=90,rad=5",
            ),
        )
        ax = fig.add_subplot(out_gs[1, 1])
        apply_axis_transform(ax)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim((-1, y_max))
        ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_pred_depr, 0), color=config.PRED_COL
        )
        ax.annotate(
            "explained A-site",
            xy=(n_codons_1 // 2, y_pred_depr[n_codons_1 // 2]),
            xycoords="data",
            xytext=(-5, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", facecolor="none", edgecolor="none", pad=0),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.5,
                connectionstyle="angle,angleA=0,angleB=90,rad=5",
            ),
        )
        ax = fig.add_subplot(out_gs[1, 2])
        apply_axis_transform(ax)
        ax.fill_between(
            np.arange(n_codons_1), np.nan_to_num(y_pred_full, 0), color=config.PRED_COL
        )
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_ylim(-1, y_max)
        ax.set_xlim(xlim)
        ax.set_yticklabels([])
        gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=out_gs[2], wspace=0.15)
        ax = fig.add_subplot(out_gs[2, 0])
        apply_axis_transform(ax)
        ax.set_ylabel("Attributions")
        ax.set_ylim((attr_min, attr_max))
        ax.set_xlim(xlim)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.fill_between(
            np.arange(n_codons_1),
            np.nan_to_num(sel_attr_ctrl, 0),
            color=config.ATTR_COL,
        )
        ax = fig.add_subplot(out_gs[2, 1])
        apply_axis_transform(ax)
        ax.set_xlim(xlim)
        ax.set_ylim((attr_min, attr_max))
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_yticklabels([])
        ax.fill_between(
            np.arange(n_codons_1),
            np.nan_to_num(sel_attr_depr, 0),
            color=config.ATTR_COL,
        )
        sns.despine()
        fig.align_ylabels()
        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, f"main_panel_heads.svg")
        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            plt.savefig(output_fpath, **config.SAVEFIG_KWARGS)
        plt.show()
    return (
        apply_axis_transform,
        attr_max,
        attr_min,
        ax,
        fig,
        gs,
        out_gs,
        output_dirpath,
        output_fpath,
        xlim,
        y_max,
    )


if __name__ == "__main__":
    app.run()
