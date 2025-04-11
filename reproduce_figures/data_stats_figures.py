import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import config
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots
    import seaborn as sns
    from pyhere import here
    from scipy.stats import pearsonr, skew

    import utils

    return (
        config,
        h5py,
        here,
        mo,
        np,
        os,
        pd,
        pearsonr,
        plt,
        scienceplots,
        skew,
        sns,
        utils,
    )


@app.cell
def _(mo):
    ALLOW_OVERWRITE = mo.cli_args().get("overwrite") or True
    OUT_DIRPATH = mo.cli_args().get("output_dirpath")
    return ALLOW_OVERWRITE, OUT_DIRPATH


@app.cell
def _(ALLOW_OVERWRITE, OUT_DIRPATH, config, h5py, here, np, os, plt, utils):
    def _():
        fig, axs = plt.subplots(
            3, 4, figsize=(config.TEXTWIDTH_INCH, 3), constrained_layout=True
        )

        split_to_data = {}
        for split, fnames in config.SPLIT_TO_FNAMES.items():
            split_data = []
            conditions = []
            for f in fnames:
                f = h5py.File(here("data", "results", "interpretability", f), "r")
                y_true_ctrl = f["y_true_ctrl"][:]
                y_true_ctrl = [np.where(x == 999, np.nan, x) for x in y_true_ctrl]
                y_true_depr = f["y_true_depr"][:]
                y_true_depr = [np.where(x == 999, np.nan, x) for x in y_true_depr]
                y_true_full = [x + y for x, y in zip(y_true_ctrl, y_true_depr)]
                split_data.extend(y_true_full)
                conditions.append([c.decode("utf8") for c in f["condition"][:]])

            split_to_data[split] = dict(
                rpf=split_data, conditions=np.concatenate(conditions)
            )

        for col, split in enumerate(config.SPLIT_TO_FNAMES.keys()):

            y_true_full = split_to_data[split]["rpf"]
            coverage = [
                (((~np.isnan(x)) & (x != 0)).sum()) / ((~np.isnan(x)).sum())
                for x in y_true_full
            ]
            length = [len(x) for x in y_true_full]
            axs[0][col].hist(coverage)
            axs[1][col].hist(length)

            axs[0][0].set_ylabel("Coverage")
            axs[1][0].set_ylabel("Sequence Length")
            axs[0][col].set_xlim(0, 1.1)

            axs[0][col].set_title(split)

            unique_values, counts = np.unique(
                split_to_data[split]["conditions"], return_counts=True
            )
            ordered_values = [
                unique_values[list(unique_values).index(cat)]
                for cat in config.CONDITIONS_FIXNAME.keys()
                if cat in unique_values
            ]

            axs[2][col].bar(
                [config.CONDITIONS_FIXNAME[c] for c in unique_values], counts
            )
            axs[2][0].set_ylabel("Condition frequency")

        fig.suptitle("Split statistics")
        fig.align_ylabels()

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(OUT_DIRPATH, "supp_dataset_statistics.svg")
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
def _(ALLOW_OVERWRITE, OUT_DIRPATH, config, h5py, here, np, os, plt, utils):
    def _():

        y_diff_pred = []
        y_diff_true = []
        condition = []
        for split, fnames in config.SPLIT_TO_FNAMES.items():
            for f in fnames:
                f = h5py.File(here("data", "results", "interpretability", f), "r")
                y_diff_pred.extend(f["y_pred_depr"][:])
                y_true_depr = f["y_true_depr"][:]
                y_true_depr = [np.where(x == 999, np.nan, x) for x in y_true_depr]
                y_diff_true.extend(y_true_depr)
                condition.extend([c.decode("utf8") for c in f["condition"][:]])

        condition = np.array(condition)

        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(config.TEXTWIDTH_INCH * 0.6, 1.5), constrained_layout=True
        )

        conds_dict = config.CONDITIONS_FIXNAME.copy()
        conds_dict.pop("CTRL")

        positive_mean = [
            [
                np.nanmean(y_diff_pred[idx][np.where(y_diff_pred[idx] > 0)])
                for idx in np.where(condition == cond)[0]
            ]
            for cond in conds_dict.keys()
            if cond != "CTRL"
        ]
        ax1.violinplot(positive_mean)
        ax1.set_xticks(np.arange(1, 6), conds_dict.values())
        ax1.set_title("Predicted Values")

        positive_mean = [
            [
                np.nanmean(y_diff_true[idx][np.where(y_diff_true[idx] > 0)])
                for idx in np.where(condition == cond)[0]
            ]
            for cond in conds_dict.keys()
            if cond != "CTRL"
        ]
        ax0.violinplot(positive_mean)
        ax0.set_xticks(np.arange(1, 6), conds_dict.values())
        ax0.set_title("True Values")

        fig.suptitle("Average positive \u0394RPF gene-wise")

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "supp_positive_rpf.svg")
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
def _(mo):
    mo.md(r"""# Distance between deprived codons plot""")
    return


@app.cell
def _():
    def distances_bn_depr_codons(seq, depr_codon_cond_ids):

        seq = [1 if k in depr_codon_cond_ids else 0 for k in seq]

        # find the distances between the 1s, and move to the next 1
        distances = []
        for i in range(len(seq)):
            if seq[i] == 1:
                for j in range(i + 1, len(seq)):
                    if seq[j] == 1:
                        distances.append(j - i)
                        i = j
                        break

        return distances

    return (distances_bn_depr_codons,)


@app.cell
def _(config, distances_bn_depr_codons, h5py, here, mo, pd, utils):
    import re

    from tqdm.auto import tqdm

    genetic_code = pd.read_csv(config.GENCODE_FPATH, index_col=0).assign(
        AminoAcid=lambda df: df.AminoAcid.str.upper()
    )
    ensembl_df = utils.read_ensembl(config.ENSEMBL_FPATH)

    cond_distances = {"LEU": [], "ILE": [], "VAL": []}

    for split, fnames in mo.status.progress_bar(config.SPLIT_TO_FNAMES.items()):
        for f in fnames:
            with h5py.File(here("data", "results", "interpretability", f), "r") as h5:
                seqs = (
                    ensembl_df.set_index("transcript")
                    .loc[h5["transcript"][:].astype(str)]
                    .sequence.values
                )
                conditions = h5["condition"][:].astype(str)
                for s, cond in zip(tqdm(seqs), conditions):
                    if cond not in cond_distances.keys():
                        continue
                    cond_distances[cond].extend(
                        distances_bn_depr_codons(
                            re.findall("...", s),
                            genetic_code.query("AminoAcid == @cond").Codon.values,
                        )
                    )
    return (
        cond,
        cond_distances,
        conditions,
        ensembl_df,
        f,
        fnames,
        genetic_code,
        h5,
        re,
        s,
        seqs,
        split,
        tqdm,
    )


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    cond_distances,
    config,
    here,
    np,
    os,
    plt,
    utils,
):
    def _():
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        labels = []

        def add_label(v, label):
            color = v["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        # make boxplots for each condition, do it log scale
        fig, ax = plt.subplots(
            1, 1, figsize=(config.TEXTWIDTH_INCH * 0.25, 1.5), constrained_layout=True
        )
        # make overlapping histograms
        for cond in cond_distances.keys():
            thresh = np.mean(cond_distances[cond]) + np.std(cond_distances[cond])
            cond_dist = [k for k in cond_distances[cond] if k < thresh]
            vp = ax.violinplot(
                cond_dist,
                vert=False,
                showmeans=True,
                showmedians=True,
                showextrema=True,
                bw_method=0.5,
                points=1000,
                widths=0.5,
                positions=[list(cond_distances.keys()).index(cond)],
            )
            for pc in vp["bodies"]:
                pc.set_facecolor(config.COND_COL[cond])
                # pc.set_edgecolor('black')
                pc.set_alpha(0.25)
            for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
                vp[partname].set_edgecolor(config.COND_COL[cond])
                vp[partname].set_linewidth(1)

        ax.set_xlabel("Codon distance between deprived codons")
        ax.set_yticks([0, 1, 2], cond_distances.keys())

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, "supp_codon_dist.svg")
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
def _(mo):
    mo.md(r"""# Control batch effect plot""")
    return


@app.cell
def _(ALLOW_OVERWRITE, config, h5py, here, mo, np, os, pearsonr, plt, utils):
    def _():
        liver_transcripts = np.load(
            here("data", "results", "interpretability", "liver_transcripts.npz")
        )["arr_0"]
        y_ctr_pred = []
        y_ctrl_true = []
        transcript = []
        splits = []
        for split, fnames in config.SPLIT_TO_FNAMES.items():
            for f in fnames:
                f = h5py.File(here("data", "results", "interpretability", f), "r")
                for idx in range(len(f["y_pred_ctrl"])):
                    if f["condition"][idx].decode("utf8") != "CTRL":
                        continue
                    else:
                        y_ctr_pred.append(f["y_pred_ctrl"][idx])
                        y_ctrl_true_tmp = f["y_true_ctrl"][idx]
                        y_ctrl_true_tmp = np.where(
                            y_ctrl_true_tmp == 999, np.nan, y_ctrl_true_tmp
                        )
                        y_ctrl_true.append(y_ctrl_true_tmp)
                        transcript.append(f["transcript"][idx].decode("utf8"))
                        splits.append(split)

        pcc = []
        std = []
        hue = []
        for ts, pr, tr in zip(transcript, y_ctr_pred, y_ctrl_true):
            hue.append("liver" if ts in liver_transcripts else "fibroblast")
            mask = ~np.isnan(tr)
            std.append(np.std(tr[mask]))
            pcc.append(pearsonr(np.array(tr)[mask], np.array(pr)[mask])[0])
        pcc = np.array(pcc)
        hue = np.array(hue)
        std = np.array(std)
        splits = np.array(splits)

        fig, axs = plt.subplots(
            1, 2, figsize=(config.TEXTWIDTH_INCH * 0.6, 1.5), constrained_layout=True
        )

        ax = axs[0]
        ax.hist(
            pcc[(hue == "liver") & (splits == "test")],
            label="liver",
            alpha=0.5,
            bins=20,
        )
        ax.hist(
            pcc[(hue != "liver") & (splits == "test")],
            label="fibroblast",
            alpha=0.5,
            bins=20,
        )
        ax.set_title("Test CTRL RFPs PCC")

        ax = axs[1]
        ax.hist(std[hue == "liver"], label="liver", alpha=0.5, bins=20)
        ax.hist(std[hue != "liver"], label="fibroblast", alpha=0.5, bins=20)
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
        ax.set_title("RFPs standard deviation")

        output_dirpath = mo.cli_args().get("output_dirpath") or here(
            "data", "results", "figures"
        )
        output_fpath = os.path.join(output_dirpath, "batch_effect.svg")
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
