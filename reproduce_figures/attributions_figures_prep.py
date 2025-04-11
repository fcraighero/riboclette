import argparse
import itertools
import os
import re
from collections import defaultdict

import config
import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from config import (
    ATTR_FNAMES,
    CONDITIONS_FIXNAME,
    ENSEMBL_FPATH,
    GENCODE_FPATH,
    id_to_codon,
)
from pyhere import here
from tqdm import tqdm, trange


def _global_attr_plot(
    f, num_samples, output: dict[list], ctrl: bool = True, topk: int = 5
) -> list:

    for i in trange(num_samples):
        if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
            continue

        num_codons_sample = len(f["x_input"][i]) - 1
        dname = "attr_ctrl" if ctrl else "attr_depr"
        lig_sample_ctrl = f[dname][i].reshape(num_codons_sample, num_codons_sample)
        suffix = "ctrl" if ctrl else "depr"
        true_counts = f[f"y_true_{suffix}"][i]
        true_counts = np.where(true_counts == 999, np.nan, true_counts)
        peaks = np.nonzero(
            true_counts >= np.nanmean(true_counts) + np.nanstd(true_counts)
        )[0]
        condition = f["condition"][i].decode("utf-8")

        for j in peaks:
            lig_sample_ctrl[j] = lig_sample_ctrl[j] / np.sum(np.abs(lig_sample_ctrl[j]))
            # take absolute value of lig_sample_ctrl[j]
            lig_sample_ctrl[j] = np.abs(lig_sample_ctrl[j])
            # get top 5 codons with highest lig_sample_ctrl[j]
            top5 = np.argsort(lig_sample_ctrl[j])[-topk:]
            # get distance between top 5 codons and the codon of interest
            for k in range(topk):
                output[condition].append(top5[k] - j)
    return output


def global_attr_plot(ctrl: bool, test=False, topk: int = 5, out_dirpath=""):
    output = defaultdict(list)
    for fname in ATTR_FNAMES:
        with h5py.File(here("data", "results", "interpretability", fname), "r") as f:
            num_samples = 1 if test else len(f["x_input"])

            _global_attr_plot(f, num_samples, output, ctrl=ctrl, topk=topk)
    if test:
        print(output)
    else:
        np.savez(
            os.path.join(out_dirpath, f"globl_attr_plot_{ctrl}"),
            **output,
        )


def run_global_attr_plot(out_dirpath=""):
    global_attr_plot(ctrl=True, out_dirpath=out_dirpath)
    global_attr_plot(ctrl=False, out_dirpath=out_dirpath)


def run_global_stalling(window_size: int = 20, out_dirpath=""):

    stop_codons = ["TAA", "TAG", "TGA"]
    condition_codon_stall = {
        "CTRL": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
        "ILE": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
        "LEU": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
        "LEU_ILE": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
        "LEU_ILE_VAL": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
        "VAL": {
            codon: [] for codon in id_to_codon.values() if codon not in stop_codons
        },
    }

    for fname in ATTR_FNAMES:
        f = h5py.File(here("data", "results", "interpretability", fname), "r")

        num_samples = len(f["condition"])

        for i in tqdm(range(num_samples)):

            if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
                continue

            sample_condition = f["condition"][i].decode("utf-8")
            if sample_condition == "CTRL":
                y_true_full_sample = f["y_true_ctrl"][i]
                y_true_full_sample = np.where(
                    y_true_full_sample == 999, np.nan, y_true_full_sample
                )
            else:
                y_true_full_sample = f["y_true_depr"][i]
            y_true_full_sample = np.where(
                y_true_full_sample == 999, np.nan, y_true_full_sample
            )
            x_input_sample = f["x_input"][i][1:]
            y_true_full_sample_norm = y_true_full_sample / np.nanmax(y_true_full_sample)
            for j in range(len(y_true_full_sample_norm)):
                if (
                    np.isnan(y_true_full_sample[j]) == False
                    and id_to_codon[int(x_input_sample[j])]
                    in condition_codon_stall[sample_condition]
                    and y_true_full_sample[j] != 0.0
                ):
                    condition_codon_stall[sample_condition][
                        id_to_codon[int(x_input_sample[j])]
                    ].append(y_true_full_sample_norm[j])

        f.close()

    condition_codon_stall_mean = {
        condition: {
            codon: np.mean(condition_codon_stall[condition][codon])
            for codon in condition_codon_stall[condition]
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_stall_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_stall_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }

    pd.DataFrame.from_dict(condition_codon_stall_mean_sorted).to_csv(
        os.path.join(out_dirpath, "condition_codon_stall_mean_sorted.zip")
    )

    del condition_codon_stall_mean_sorted

    condition_codon_attr_peaks = {
        "CTRL": {codon: [] for codon in id_to_codon.values()},
        "ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE_VAL": {codon: [] for codon in id_to_codon.values()},
        "VAL": {codon: [] for codon in id_to_codon.values()},
    }
    condition_codon_attr_full = {
        "CTRL": {codon: [] for codon in id_to_codon.values()},
        "ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE_VAL": {codon: [] for codon in id_to_codon.values()},
        "VAL": {codon: [] for codon in id_to_codon.values()},
    }

    stop_codons = ["TAA", "TAG", "TGA"]
    for condition in CONDITIONS_FIXNAME.keys():
        for codon in stop_codons:
            condition_codon_attr_peaks[condition].pop(codon)
            condition_codon_attr_full[condition].pop(codon)

    for fname in ATTR_FNAMES:
        f = h5py.File(here("data", "results", "interpretability", fname), "r")

        num_samples = len(f["condition"])
        for i in tqdm(range(num_samples)):

            if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
                continue

            sample_cond = f["condition"][i].decode("utf-8")
            x_input_sample = f["x_input"][i][1:]
            num_codons = len(x_input_sample)
            if sample_cond == "CTRL":
                lig_attr_ctrl_sample = f["attr_ctrl"][i].reshape(num_codons, num_codons)
                y_true_full_sample = f["y_true_ctrl"][i]
            else:
                lig_attr_ctrl_sample = f["attr_depr"][i].reshape(num_codons, num_codons)
                y_true_full_sample = f["y_true_depr"][i]

            y_true_full_sample = np.where(
                y_true_full_sample == 999, np.nan, y_true_full_sample
            )

            # set j to be starting points, and k to be the end points
            for j in range(len(y_true_full_sample)):
                # a_site = top10_indices[j]
                a_site = j
                start = a_site - window_size
                end = a_site + window_size + 1

                lig_attr_ctrl_sample_window = lig_attr_ctrl_sample[a_site][start:end]
                if len(lig_attr_ctrl_sample_window) == (window_size * 2) + 1:
                    lig_attr_ctrl_sample_window_norm = (
                        lig_attr_ctrl_sample_window
                        / np.max(np.abs(lig_attr_ctrl_sample_window))
                    )
                    x_input_sample_window = x_input_sample[start:end]
                    for l in range(len(x_input_sample_window)):
                        if (
                            id_to_codon[int(x_input_sample_window[l])]
                            in condition_codon_attr_full[sample_cond]
                        ):
                            condition_codon_attr_full[sample_cond][
                                id_to_codon[int(x_input_sample_window[l])]
                            ].append(lig_attr_ctrl_sample_window_norm[l])

            peaks = np.nonzero(
                y_true_full_sample
                >= np.nanmean(y_true_full_sample) + np.nanstd(y_true_full_sample)
            )[0]
            for j in range(len(peaks)):
                a_site = peaks[j]
                start = a_site - window_size
                end = a_site + window_size + 1

                lig_attr_ctrl_sample_window = lig_attr_ctrl_sample[a_site][start:end]
                if len(lig_attr_ctrl_sample_window) == (window_size * 2) + 1:
                    lig_attr_ctrl_sample_window_norm = (
                        lig_attr_ctrl_sample_window
                        / np.max(np.abs(lig_attr_ctrl_sample_window))
                    )
                    x_input_sample_window = x_input_sample[start:end]
                    for l in range(len(x_input_sample_window)):
                        if (
                            id_to_codon[int(x_input_sample_window[l])]
                            in condition_codon_attr_peaks[sample_cond]
                        ):
                            condition_codon_attr_peaks[sample_cond][
                                id_to_codon[int(x_input_sample_window[l])]
                            ].append(lig_attr_ctrl_sample_window_norm[l])

        f.close()

    condition_codon_attr_full_mean = {
        condition: {
            codon: np.mean(condition_codon_attr_full[condition][codon])
            for codon in condition_codon_attr_full[condition]
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_attr_full_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_attr_full_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }

    pd.DataFrame.from_dict(condition_codon_attr_full_mean_sorted).to_csv(
        os.path.join(out_dirpath, "condition_codon_attr_full_mean_sorted.zip")
    )

    condition_codon_attr_peaks_mean = {
        condition: {
            codon: np.mean(condition_codon_attr_peaks[condition][codon])
            for codon in condition_codon_attr_peaks[condition]
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_attr_peaks_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_attr_peaks_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in CONDITIONS_FIXNAME.keys()
    }

    pd.DataFrame.from_dict(condition_codon_attr_peaks_mean_sorted).to_csv(
        os.path.join(out_dirpath, "condition_codon_attr_peaks_mean_sorted.zip")
    )


def run_topk_attr_condition_wise(wsize: int = 20, out_dirpath=""):

    # Load genetic code
    genetic_code = pd.read_csv(GENCODE_FPATH, index_col=0).set_index("Codon")
    genetic_code.head()

    # Load gene to seq dataframe
    df_trans_to_seq = []
    with open(ENSEMBL_FPATH, mode="r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            df_trans_to_seq.append(
                [
                    record.id,
                    str(record.seq),
                    record.description.split("gene_symbol:")[1].split()[0],
                ]
            )
    df_trans_to_seq = pd.DataFrame(
        df_trans_to_seq, columns=["transcript", "sequence", "symbol"]
    )

    # Define conditions
    conditions = ["ILE", "LEU", "LEU_ILE", "LEU_ILE_VAL", "VAL", "CTRL"]

    condition_attr_depr_head = {
        cond: {cod: [] for cod in genetic_code.index} for cond in conditions
    }

    for fname in ATTR_FNAMES:

        f = h5py.File(here("data", "results", "interpretability", fname), "r")
        transcripts = f["transcript"][:].astype("U")
        conditions = f["condition"][:].astype("U")

        # Loop throught transcripts
        for transc_idx in trange(transcripts.shape[0]):
            condition = conditions[transc_idx]

            suffix = "ctrl" if condition == "CTRL" else "depr"

            transcript = transcripts[transc_idx]

            if transcript in config.DISCARDED_TRANSCRIPTS:
                continue

            # Get attribution vector and reshape to matrix
            trasc_attr = f[f"attr_{suffix}"][transc_idx]
            n_codons = int(np.sqrt(trasc_attr.shape[0]))
            trasc_attr = trasc_attr.reshape(n_codons, n_codons)

            # Get sequence
            sequence = df_trans_to_seq.query(
                "transcript == @transcript"
            ).sequence.values[0]
            sequence = np.array(re.findall("...", sequence))

            depr_true = f[f"y_true_{suffix}"][transc_idx]
            depr_true = np.where(depr_true == 999, np.nan, depr_true)

            good_idxs = np.nonzero(
                depr_true > np.nanmean(depr_true) + np.nanstd(depr_true)
            )[0]
            good_idxs = good_idxs[
                (good_idxs >= wsize) & (good_idxs < n_codons - wsize - 1)
            ]

            # Loop through the sequence, excluding a prefix and suffix of wsize and wsize+1, respectively
            for idx in good_idxs:
                wattr = trasc_attr[idx, idx - wsize : idx + wsize + 1]
                wattr = wattr / np.abs(wattr).max()
                for attr_idx, codon_idx in enumerate(
                    np.arange(idx - wsize, idx + wsize + 1)
                ):
                    condition_attr_depr_head[condition][sequence[codon_idx]].append(
                        wattr[attr_idx]
                    )

        f.close()

    condition_attr_depr_head = {
        cond: {cod: np.mean(w) if len(w) > 0 else 0 for cod, w in cod_wise.items()}
        for cond, cod_wise in condition_attr_depr_head.items()
    }
    pd.DataFrame(condition_attr_depr_head).to_csv(
        os.path.join(out_dirpath, "topk_attr_cond_wise.zip")
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that takes an output directory path."
    )
    parser.add_argument(
        "-output_dirpath",
        type=str,
        required=False,
        help="Path to the output directory",
    )

    args = parser.parse_args()
    output_dirpath = args.output_dirpath or here("data", "results", "plotting")

    print("Running global_attr_plot")
    run_global_attr_plot(out_dirpath=output_dirpath)

    print("Running global_stalling")
    run_global_stalling(window_size=10, out_dirpath=output_dirpath)

    print("Running make_topk_attr_condition_wise")
    run_topk_attr_condition_wise(wsize=10, out_dirpath=output_dirpath)
