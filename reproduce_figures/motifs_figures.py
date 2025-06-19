import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools

    import config
    import marimo as mo
    import marsilea as ma
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots
    import utils
    from pyhere import here
    from tqdm.auto import tqdm
    import os
    import h5py
    from collections import defaultdict
    return (
        config,
        defaultdict,
        h5py,
        here,
        itertools,
        ma,
        mo,
        np,
        os,
        pd,
        plt,
        scienceplots,
        tqdm,
        utils,
    )


@app.cell
def _(mo):
    ALLOW_OVERWRITE = mo.cli_args().get("overwrite") or True
    OUT_DIRPATH = mo.cli_args().get("output_dirpath")
    return ALLOW_OVERWRITE, OUT_DIRPATH


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(here, np):
    mutations_everything = np.load(
        here("data", "results", "motifs", "raw_bms", "motifswAF_addStall_1000.npz"),
        allow_pickle=True,
    )["mutations_everything"].item()
    keys = list(mutations_everything.keys())
    return keys, mutations_everything


@app.cell
def _(keys, mutations_everything, np, tqdm):
    window_size = 10
    motif_str = []
    motif_len = []
    condition = []
    perc_increase = []
    orig_density_list = []
    new_density_list = []
    for k in tqdm(keys):
        start = k[2]
        orig_density = k[5]
        for mot in mutations_everything[k]:
            condition.append(k[4])
            new_density = mutations_everything[k][mot]
            orig_density_list.append(orig_density)
            new_density_list.append(new_density)
            try:
                perc_increase.append(
                    np.abs((new_density - orig_density) / orig_density)
                )
            except:
                perc_increase.append(0)
            motif_len.append(int(len(mot) / 2))
            motif_sample_dict = {}
            for _i in range(0, len(mot), 2):
                motif_sample_dict[mot[_i] - (start + window_size)] = mot[_i + 1]
            motif_sample_dict = dict(sorted(motif_sample_dict.items()))
            motif_str_sample = ""
            for k1, v1 in motif_sample_dict.items():
                motif_str_sample = motif_str_sample + (str(k1) + "_" + str(v1) + "_")
            motif_str.append(motif_str_sample)
    return (
        condition,
        k,
        k1,
        mot,
        motif_len,
        motif_sample_dict,
        motif_str,
        motif_str_sample,
        new_density,
        new_density_list,
        orig_density,
        orig_density_list,
        perc_increase,
        start,
        v1,
        window_size,
    )


@app.cell
def _(
    condition,
    motif_len,
    motif_str,
    new_density_list,
    orig_density_list,
    pd,
    perc_increase,
):
    # make a dataframe
    df = pd.DataFrame(
        {
            "motif": motif_str,
            "motif_len": motif_len,
            "perc_increase": perc_increase,
            "condition": condition,
            "orig_density": orig_density_list,
            "new_density": new_density_list,
        }
    )
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""# Create Motif Positional Heatmap""")
    return


@app.cell
def _(itertools):
    id_to_codon = {
        str(idx): "".join(el)
        for idx, el in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
    }
    codon_to_id = {v: k for k, v in id_to_codon.items()}
    id_to_codon["X"] = "[SKIP]"

    # Function to fill missing integers with zeros
    def create_exp_motif(lst):

        pos = lst["positions"]
        motif_codons = iter(lst["motif_codons"])

        # Create a range from the minimum to the maximum value in the list
        full_range = range(min(pos), max(pos) + 1)
        # Replace missing integers with 0
        return [next(motif_codons) if i in pos else "X" for i in full_range]
    return codon_to_id, create_exp_motif, id_to_codon


@app.cell
def _(mo):
    mo.md(r"""## Define filled motifs and their starting positions""")
    return


@app.cell
def _(create_exp_motif, df, np):
    filled_motifs_df = (
        df.assign(motif_exp=lambda df: df.motif.str[:-1].str.split("_"))
        .assign(
            positions=lambda df: df.motif_exp.apply(
                lambda lst: list(map(int, lst[0::2]))
            )
        )
        .assign(motif_codons=lambda df: df.motif_exp.apply(lambda lst: lst[1::2]))
        .assign(filled_motif_codons=lambda df: df.apply(create_exp_motif, axis=1))
        .assign(start_pos=lambda df: df.positions.apply(lambda lst: int(lst[0])))
        .assign(
            MAAPE=lambda df: df.apply(
                lambda row: 100
                * np.arctan(
                    np.abs(
                        (row.orig_density - row.new_density) / (row.orig_density + 1e-9)
                    )
                ),
                axis=1,
            )
        )
        .drop(
            columns=[
                "motif",
                "motif_len",
                "perc_increase",
                "orig_density",
                "new_density",
                "motif_exp",
                "positions",
                "motif_codons",
            ]
        )
        .assign(motif_key=lambda df: df.filled_motif_codons.astype(str))
        .groupby(["condition", "motif_key"])
        .agg(
            dict(start_pos=lambda x: list(x), MAAPE="mean", filled_motif_codons="first")
        )
        .reset_index()
        .drop(columns=["motif_key"])
    )
    filled_motifs_df
    return (filled_motifs_df,)


@app.cell
def _(mo):
    mo.md(r"""## Aggregate filled motifs and compute motif significance""")
    return


@app.cell
def _(codon_to_id, config, defaultdict, h5py, here, np):
    def count_substring_pattern_matches(data, patterns, wildcard='[SKIP]'):
        pattern_count = defaultdict(lambda: 0)

        for pattern_str in patterns:
            pattern = pattern_str.split()
            pattern_len = len(pattern)

            for row in data:
                row = np.array(row).astype(int)

                if len(row) < pattern_len: 
                    # Too few positions
                    continue

                # Sliding windows for this row
                windows = np.lib.stride_tricks.sliding_window_view(row, window_shape=pattern_len)

                # Build boolean mask for matching
                mask = np.ones(windows.shape[0], dtype=bool)
                for i, val in enumerate(pattern):
                    if val != wildcard:
                        mask &= (windows[:, i] == int(codon_to_id[val]))

                pattern_count[f"hit_{pattern_str}"] += np.sum(mask)
                pattern_count[f'miss_{pattern_str}'] += windows.shape[0] - np.sum(mask)

        return pattern_count

    def _extract_windows(arr, indices, window_size):
        arr_idxs = np.arange(len(arr))
        indices = np.asarray(indices)
        windows = []

        for i in indices:
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(arr))
            windows.append(arr_idxs[start:end])

        if len(windows) > 0:
            peak_idxs = np.unique(np.concatenate(windows))
            return arr[np.setdiff1d(arr_idxs, peak_idxs)], arr[peak_idxs]
        else:
            return arr, []

    def extract_windows(f, window_size=10, condition = 'LEU'):

        data_non_peak, data_peak = [], []
        for idx in range(len(f['x_input'])):

            if f["transcript"][idx] in config.DISCARDED_TRANSCRIPTS:
                continue

            if f['condition'][idx].decode('utf-8') != condition: continue

            suffix = "ctrl" if condition == "CTRL" else "depr"

            _true_rpf = np.where(f[f'y_true_{suffix}'][idx] == 999, np.nan, f[f'y_true_{suffix}'][idx])

            indices = np.where(_true_rpf >= np.nanmean(_true_rpf) + np.nanstd(_true_rpf))[0]

            _substring_non_peak, _substring_peak = _extract_windows(f['x_input'][idx], indices, window_size)

            if len(_substring_non_peak) > 0: data_non_peak.append(_substring_non_peak)
            if len(_substring_peak) > 0: data_peak.append(_substring_peak)
        return data_non_peak, data_peak

    def extract_windows_across_files(condition, window_size=10):
        """Extract subsequences in genes that are or are not around peaks."""

        data_non_peak, data_peak = [], []
        for fname in config.ATTR_FNAMES:
            with h5py.File(here("data", "results", "interpretability", fname), "r") as f:
                non_peak, peak = extract_windows(f, window_size=window_size, condition=condition)
                data_non_peak.extend(non_peak)
                data_peak.extend(peak)
        return data_non_peak, data_peak
    return (
        count_substring_pattern_matches,
        extract_windows,
        extract_windows_across_files,
    )


@app.cell
def _(count_substring_pattern_matches, extract_windows_across_files):
    from scipy.stats import fisher_exact, binomtest, poisson_means_test
    from statsmodels.stats.multitest import multipletests

    def pvalue_to_stars(p):
            if p < 0.001:
                return 'p < 0.001'
            elif p < 0.01:
                return 'p < 0.01'
            elif p < 0.05:
                return 'p < 0.05'
            else:
                return 'ns'

    def get_pvalue_starts(condition, patterns):
        data_non_peak, data_peak = extract_windows_across_files(condition=condition)
        peak_pattern_count = count_substring_pattern_matches(data_peak, patterns=patterns)
        non_peak_pattern_count = count_substring_pattern_matches(data_non_peak, patterns=patterns)

        stattest_dict = {t: [] for t in ['pvalue', 'stats', 'n_hit_peak', 'n_hit_npeak', 'n_miss_peak', 'n_miss_npeak', 'filled_motif_codons']}
        for motif in patterns:
            stat, pvalue = poisson_means_test(peak_pattern_count[f"hit_{motif}"], peak_pattern_count[f"miss_{motif}"]+peak_pattern_count[f"hit_{motif}"], non_peak_pattern_count[f"hit_{motif}"], non_peak_pattern_count[f"miss_{motif}"]+non_peak_pattern_count[f"hit_{motif}"])
            stattest_dict['filled_motif_codons'].append(motif)
            stattest_dict['pvalue'].append(pvalue)
            stattest_dict['stats'].append(stat)
            stattest_dict['n_hit_peak'].append(peak_pattern_count[f"hit_{motif}"])
            stattest_dict['n_miss_peak'].append(peak_pattern_count[f"miss_{motif}"])
            stattest_dict['n_hit_npeak'].append(non_peak_pattern_count[f"hit_{motif}"])
            stattest_dict['n_miss_npeak'].append(non_peak_pattern_count[f"miss_{motif}"])

        # Correct
        _, pvals_corrected, _, _ = multipletests(stattest_dict['pvalue'], alpha=0.05, method='fdr_bh')
        stattest_dict['pvalue_corrected'] = pvals_corrected
        stattest_dict['pvalue_significance'] = [pvalue_to_stars(pc) for pc in pvals_corrected]

        return stattest_dict
    return (
        binomtest,
        fisher_exact,
        get_pvalue_starts,
        multipletests,
        poisson_means_test,
        pvalue_to_stars,
    )


@app.cell
def _(config, filled_motifs_df, get_pvalue_starts, id_to_codon, np, pd, tqdm):
    def _():
        agg_filled_motifs_df = (
            filled_motifs_df.assign(
                freq=lambda df: df.start_pos.apply(lambda x: len(x))
            )
            .groupby("condition", group_keys=False)
            .apply(lambda x: x.nlargest(50, "freq"))
            .assign(
                filled_motif_codons=lambda df: df.filled_motif_codons.apply(
                    lambda x: " ".join([id_to_codon[key] for key in x])
                )
            )
            .assign(
                bincount=lambda df: df.start_pos.apply(
                    lambda x: np.bincount(np.array(x) + 10, minlength=16) / len(x)
                )
            )
        )

        significance_dfs = []
        for cond in tqdm(config.CONDITIONS_FIXNAME.keys()):
            motif_to_stars = get_pvalue_starts(condition=cond, patterns=agg_filled_motifs_df.query(f'condition == @cond').filled_motif_codons)
            significance_dfs.append(pd.DataFrame.from_dict(motif_to_stars).assign(condition=cond))
        return agg_filled_motifs_df.merge(pd.concat(significance_dfs), on=['filled_motif_codons', 'condition'])
    agg_filled_motifs_df = _()
    return (agg_filled_motifs_df,)


@app.cell
def _(agg_filled_motifs_df, config, here, pd, tqdm):
    def _():
        with pd.ExcelWriter(here("data", "results", "plotting", 'motifs_significance.xlsx'), mode='w') as writer:
            for cond in tqdm(config.CONDITIONS_FIXNAME.keys()):
                df_to_save = (
                    agg_filled_motifs_df.query('condition == @cond')
                    .rename(
                        columns=dict(filled_motif_codons='motif', n_hit_npeak="n_hit_not_peak", n_miss_npeak="n_miss_not_peak", stats='statistics')
                    )[['motif', 'n_hit_peak', 'n_miss_peak', 'n_hit_not_peak', 'n_miss_not_peak', 'statistics', 'pvalue', 'pvalue_corrected']]
                .set_index('motif')
                .sort_values('statistics',ascending=False))
                df_to_save.to_excel(writer, sheet_name=cond)
    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Generate MAAPE summary dataframe""")
    return


@app.cell
def _(create_exp_motif, df, np):
    result_df2 = (
        df.assign(motif_exp=lambda df: df.motif.str[:-1].str.split("_"))
        .assign(
            positions=lambda df: df.motif_exp.apply(
                lambda lst: list(map(int, lst[0::2]))
            )
        )
        .assign(motif_codons=lambda df: df.motif_exp.apply(lambda lst: lst[1::2]))
        .assign(filled_motif_codons=lambda df: df.apply(create_exp_motif, axis=1))
        .assign(start_pos=lambda df: df.positions.apply(lambda lst: int(lst[0])))
        .assign(
            MAAPE=lambda df: df.apply(
                lambda row: 100
                * np.arctan(
                    np.abs(
                        (row.orig_density - row.new_density) / (row.orig_density + 1e-9)
                    )
                ),
                axis=1,
            )
        )
        .drop(
            columns=[
                "motif",
                "motif_len",
                "perc_increase",
                "orig_density",
                "new_density",
                "motif_exp",
                "positions",
                "motif_codons",
            ]
        )
        .assign(motif_key=lambda df: df.filled_motif_codons.astype(str))
        .groupby(["condition", "motif_key", "start_pos"])
        .agg(dict(MAAPE="mean", filled_motif_codons="first"))
    )
    result_df2.head()
    return (result_df2,)


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    agg_filled_motifs_df,
    config,
    here,
    id_to_codon,
    ma,
    np,
    os,
    pd,
    plt,
    result_df2,
    utils,
):
    from matplotlib.ticker import FixedLocator, FuncFormatter

    def make_motif_pos_heat(
        agg_filled_motifs_df, height: int, fname: str
    ):

        maape = (
            result_df2
            .assign(
                filled_motif_codons=lambda df: df.filled_motif_codons.apply(
                    lambda x: " ".join([id_to_codon[key] for key in x])
            ))
            .reset_index()
            .pivot(
                index=["condition", "filled_motif_codons"], columns="start_pos", values="MAAPE"
            )
            .loc[pd.MultiIndex.from_frame(agg_filled_motifs_df[['condition',"filled_motif_codons"]])]
            .fillna(0)
            .values
        )

        pos_freq = np.stack(agg_filled_motifs_df.bincount.values)

        fmt = FuncFormatter(lambda x, _: f"{x:.0%}")
        cells_proportion = ma.plotter.SizedMesh(
            pos_freq,
            color=maape,
            sizes=(0, 25),
            linewidth=0,
            cmap="Reds",
            size_legend_kws=dict(
                title="% of occurrences",
                title_fontproperties=dict(weight="normal", size=config.FSM),
                fontsize=config.FSS,
                show_at=[0.25, 0.5, 0.75, 1],
                fmt=fmt,
                ncols=4,
            ),
            color_legend_kws=dict(
                title="MAAPE",
                orientation="horizontal",
                height=1.5,
                fontsize=config.FSS,
                title_fontproperties=dict(weight="normal", size=config.FSM),
            ),
        )
        _h = ma.base.ClusterBoard(np.zeros(pos_freq.shape), width=1.4, height=height)
        _h.add_layer(cells_proportion)
        _h.add_left(
            ma.plotter.Labels(agg_filled_motifs_df.filled_motif_codons, align="center")
        )
        cmap = plt.get_cmap("Greys")
        values = np.linspace(0, 1, 4)
        colors = [cmap(v) for v in values]
        _h.add_right(
            ma.plotter.Colors(
                agg_filled_motifs_df.pvalue_significance, 
                palette = {'p < 0.001': colors[-1], 'p < 0.01': colors[-2], 'p < 0.05': colors[-3], 'ns': colors[-4]}, 
                label="Significance",
                legend_kws=dict(title_fontproperties=dict(weight="normal", size=config.FSM))),
            size=0.05, pad=0.05
        )
        _h.group_rows(agg_filled_motifs_df.condition)
        conditions = agg_filled_motifs_df.condition.unique()
        _h.add_left(
            ma.plotter.Chunk(
                [config.CONDITIONS_FIXNAME[c] for c in conditions],
                fill_colors=[config.COND_COL[c] for c in conditions],
            ),
            pad=0.05,
        )
        _h.add_top(
            ma.plotter.Labels(
                list(range(-10, -2)) + ["E", "P", "A"] + list(range(1, 6)),
                align="center",
                rotation=0,
            )
        )
        _h.add_legends(
            "bottom",
            align_stacks="center",
            align_legends="center",
            stack_by="col",
            pad=0.2,
        )
        _h.render()

        output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
        output_fpath = os.path.join(output_dirpath, fname)
        if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
            print(f"Saving figure to: {output_fpath}")
            plt.savefig(output_fpath, dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.show()

    def _():
        with utils.journal_plotting_ctx():
            make_motif_pos_heat(
                agg_filled_motifs_df.query("condition in ['CTRL', 'ILE', 'LEU', 'VAL']").groupby("condition", group_keys=False).apply(lambda x: x.nlargest(10, "freq")), 4, "motif_pos_heatmap_sign.svg"
            )
            for c in config.CONDITIONS_FIXNAME.keys():
                make_motif_pos_heat(agg_filled_motifs_df.query(f"condition == @c"), 5, f"supp_motif_pos_heatmap_{c}_sign.svg")
    _()
    return FixedLocator, FuncFormatter, make_motif_pos_heat


@app.cell
def _(mo):
    mo.md(r"""# Create Codon Stats on Motifs""")
    return


@app.cell
def _(itertools):
    id_to_codon_1 = {
        idx: "".join(el)
        for idx, el in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
    }
    codon_to_id_1 = {v: k for k, v in id_to_codon_1.items()}
    stop_codons = ["TAA", "TAG", "TGA"]
    codonid_list = []
    for _i in range(64):
        codon = id_to_codon_1[_i]
        if codon not in stop_codons:
            codonid_list.append(_i)
    condition_dict_values = {
        64: "CTRL",
        65: "ILE",
        66: "LEU",
        67: "LEU_ILE",
        68: "LEU_ILE_VAL",
        69: "VAL",
    }
    condition_dict = {v: k for k, v in condition_dict_values.items()}
    return (
        codon,
        codon_to_id_1,
        codonid_list,
        condition_dict,
        condition_dict_values,
        id_to_codon_1,
        stop_codons,
    )


@app.cell
def _(motif_str):
    pos = []
    for _i in range(len(motif_str)):
        x = motif_str[_i].split("_")[:-1]
        x = [int(el) for el in x]
        for _j in range(0, len(x), 2):
            pos.append(x[_j])
    return pos, x


@app.cell
def _(df, id_to_codon_1, np, tqdm, window_size):
    condition_wise_dfs = {}
    for _c in df["condition"].unique():
        _df_c = df[df["condition"] == _c]
        _df_c = _df_c["motif"].value_counts(normalize=True).reset_index()
        _df_c.columns = ["motif", "perc"]
        c_list_fin = [[] for _ in range(window_size * 2 + 1)]
        for m in tqdm(_df_c["motif"]):
            m_s = m.split("_")[:-1]
            for _i in range(0, len(m_s), 2):
                c_list_fin[int(m_s[_i]) + window_size].append(
                    id_to_codon_1[int(m_s[_i + 1])]
                )
            pos_motif = [int(x) for x in m_s[::2]]
            for _i in range(-window_size, window_size + 1):
                if _i not in pos_motif:
                    c_list_fin[_i + window_size].append("-")
        for _i in range(-window_size, window_size + 1):
            _df_c["codon_" + str(_i)] = c_list_fin[_i + window_size]
        _df_c.columns = (
            ["motif", "perc_counts"]
            + [str(i) for i in np.arange(-window_size, -2)]
            + ["E", "P", "A"]
            + [str(idx) for idx in np.arange(1, window_size + 1)]
        )
        condition_wise_dfs[_c] = _df_c
    return c_list_fin, condition_wise_dfs, m, m_s, pos_motif


@app.cell
def _(
    ALLOW_OVERWRITE,
    OUT_DIRPATH,
    condition_wise_dfs,
    config,
    here,
    id_to_codon_1,
    ma,
    np,
    os,
    pd,
    plt,
    utils,
):
    def make_motif_heatmaps(filter_aa=True):
        for _c in ["CTRL", "ILE", "LEU", "VAL", "LEU_ILE", "LEU_ILE_VAL"]:
            _df_c = condition_wise_dfs[_c]
            _df_c = _df_c.drop(columns=["motif", "perc_counts"])
            df_c_mat = _df_c.to_numpy()
            df_c_mat_tot = np.zeros((64, 21))
            df_c_mat_perc = np.zeros((64, 21))
            for _i in range(21):
                codon_counts = df_c_mat[:, _i]
                num_non_dash = np.sum(codon_counts != "-")
                _idx = 0
                for _j in range(64):
                    df_c_mat_perc[_j, _i] = (
                        np.sum(codon_counts == id_to_codon_1[_j]) / num_non_dash * 100
                        if num_non_dash != 0
                        else 0
                    )
                    df_c_mat_tot[_j, _i] = np.sum(codon_counts == id_to_codon_1[_j])

            stack_data = pd.DataFrame(
                df_c_mat_tot, index=[id_to_codon_1[i] for i in range(64)]
            )
            genetic_code = pd.read_csv(config.GENCODE_FPATH, index_col=0).set_index(
                "Codon"
            )
            top_aa = (
                stack_data.merge(genetic_code, right_index=True, left_index=True)
                .groupby("AminoAcid")
                .max()
                .max(axis=1)
            )

            top_aa /= top_aa.max()
            top_aa = top_aa.loc[top_aa > 0.5]

            top_codons = (
                genetic_code.reset_index()
                .query('AminoAcid !="Stp"')
                .set_index("AminoAcid")
            )
            if filter_aa:
                top_codons = top_codons.loc[top_aa.index]
            stack_data_t = stack_data.loc[top_codons.Codon.values]
            stack_data_thresh = stack_data_t / stack_data_t.max().max()

            _h = ma.Heatmap(
                stack_data_thresh.T,
                linewidth=0.5,
                width=1.6 / 17 * len(top_codons),
                height=2,
                cmap="Blues",
                label="Rescaled occurrence count",
                vmin=0,
                vmax=1,
                cbar_kws=dict(
                    orientation="horizontal",
                    height=1.5,
                    title_fontproperties=dict(weight="normal", size=config.FSM),
                    fontsize=config.FSS,
                ),
            )

            genetic_code = genetic_code.loc[stack_data_thresh.index]
            _aa_oneletter = [config.AMINO_ACID_MAP[_a] for _a in genetic_code.AminoAcid]
            _aa_oneletter_order = np.unique(_aa_oneletter)
            _h.group_cols(
                group=_aa_oneletter,
                order=_aa_oneletter_order,
                spacing=0.007 if filter_aa else 0.002,
            )
            _colors = [
                config.AMINO_ACID_COLORS[config.AMINO_ACID_MAP_r[_a]]
                for _a in _aa_oneletter_order
            ]
            _h.add_top(
                ma.plotter.Chunk(_aa_oneletter_order, _colors, fontsize=config.FSM),
                pad=0.025,
            )
            _h.add_bottom(
                ma.plotter.Labels(
                    list(stack_data_thresh.index), fontsize=config.FSS, rotation=45
                ),
                name="Codon",
            )
            num_motifs_list = stack_data_t.sum(axis=0).values
            num_motifs_list /= np.sum(num_motifs_list)
            num_motifs_list *= 100

            cm = ma.plotter.ColorMesh(
                num_motifs_list.reshape(1, -1),
                cmap="Reds",
                vmin=0,
                vmax=30,
                cbar_kws=dict(
                    orientation="horizontal",
                    title="Position freq (%)",
                    height=1.5,
                    title_fontproperties=dict(weight="normal", size=config.FSM),
                    fontsize=config.FSS,
                ),
            )
            _h.add_right(cm, pad=0.05, size=0.075)
            c_text = _c
            if _c == "LEU_ILE":
                c_text = "(L, I)"
            if _c == "LEU_ILE_VAL":
                c_text = "(L, I, V)"
            _h.add_title(c_text, fontsize=config.FSB)
            pos_labels_list = [
                "-10",
                "-9",
                "-8",
                "-7",
                "-6",
                "-5",
                "-4",
                "-3",
                "E",
                "P",
                "A",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ]
            _h.add_left(ma.plotter.Labels(list(pos_labels_list), fontsize=config.FSS))
            if _c == "CTRL":
                _h.add_legends(pad=0.025)
            _h.render()
            if filter_aa:
                fname = f"motif_heatmap_{_c}.svg"
            else:
                fname = f"supp_motif_heatmap_{_c}.svg"

            output_dirpath = OUT_DIRPATH or here("data", "results", "figures")
            output_fpath = os.path.join(output_dirpath, fname)
            if ALLOW_OVERWRITE or not os.path.isfile(output_fpath):
                print(f"Saving figure to: {output_fpath}")
                plt.savefig(output_fpath, dpi=600, bbox_inches="tight", pad_inches=0.0)
            plt.show()

    with utils.journal_plotting_ctx():
        make_motif_heatmaps(True)
    return (make_motif_heatmaps,)


@app.cell
def _(make_motif_heatmaps, utils):
    with utils.journal_plotting_ctx():
        make_motif_heatmaps(False)
    return


if __name__ == "__main__":
    app.run()
