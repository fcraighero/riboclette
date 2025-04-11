import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    import os
    import pandas as pd
    from Bio import SeqIO
    from tqdm.auto import tqdm
    import itertools
    import numpy as np
    from pyhere import here
    return SeqIO, here, itertools, np, os, pd, tqdm


@app.cell
def _(here):
    # Cell tags: parameters
    DATA_FOLDER = here('data', 'Lina') # path to the Lina dataset
    LIVER_FOLDER = here('data', 'Liver') # path to the liver dataset
    fa_path = here('data', 'ensembl.cds.fa') # path to the fasta file
    return DATA_FOLDER, LIVER_FOLDER, fa_path


@app.cell
def _(SeqIO, itertools, np, os, pd, tqdm):
    # id to codon and codon to id
    id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id = {v:k for k,v in id_to_codon.items()}

    def make_dataframe(ribo_fname: str, data_path: str, df_trans_to_seq, count_norm: str = "mean"):
        '''
        inputs: path to ribosome data, path to transcript to sequence mapping dataframe
        outputs: processed dataframe with one transcript per gene, with the normalized counts
        '''
        ribo_fpath = os.path.join(data_path, ribo_fname)

        # Import dataset with ribosome data
        df_ribo = pd.read_csv(
            ribo_fpath,
            sep=" ",
            on_bad_lines="warn",
            dtype=dict(gene="category", transcript="category"),
        ).rename(columns={"count": "counts"})

        # Define count normalization function
        if count_norm == "max":
            f_norm = lambda x: x / x.max()
        elif count_norm == "mean":
            f_norm = lambda x: x / x.mean()
        elif count_norm == "sum":
            f_norm = lambda x: x / x.sum()
        else:
            raise ValueError()

        # Create final dataframe
        final_df = (
            df_ribo.merge(df_trans_to_seq).assign(fname=ribo_fname)
            # Filter spurious positions at the end of the sequence
            .query("position_A_site <= n_codons * 3")
            # Compute normalized counts
            .assign(
                norm_counts=lambda df: df.groupby("gene", observed=True).counts.transform(
                    f_norm
                )
            )
        )

        return final_df

    def make_all_dataframes(data_dirpath: str, fa_fpath: str, max_n_codons: int = 2000, count_norm: str = "mean"):
        '''
        inputs: path to ribosome data, path to ensembl fasta, maximum number of codons in the sequence, count normalization method
        outputs: merged dataframe with the ribosome read counts
        '''
        data = []
        with open(fa_fpath, mode="r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                data.append([record.id, str(record.seq)])

        # Create transcripts to sequences mapping

        df_trans_to_seq = pd.DataFrame(data, columns=["transcript", "sequence"])

        # Removes those sequences that have Ns
        sequence_has_n = df_trans_to_seq.sequence.str.contains("N", regex=False)
        df_trans_to_seq = df_trans_to_seq.loc[~sequence_has_n]

        # Number of codons in sequence
        df_trans_to_seq = df_trans_to_seq.assign(
            n_codons=lambda df: df.sequence.str.len() // 3
        )

        # Compute and merge dataframes
        dfs = [
            make_dataframe(
                f,
                df_trans_to_seq=df_trans_to_seq.drop("sequence", axis=1),
                data_path=data_dirpath,
                count_norm=count_norm,
            )
            for f in tqdm(os.listdir(data_dirpath))
            if not f.startswith("ensembl")
        ]
        dfs = pd.concat(dfs)
        for col in ["transcript", "gene", "fname"]:
            dfs[col] = dfs[col].astype("category")

        dfs = dfs.groupby(["transcript", "position_A_site"], observed=True)

        # Average replicates
        dfs = dfs.agg(dict(norm_counts="mean", gene="first")).reset_index()
    
        dfs = dfs.assign(codon_idx=lambda df: df.position_A_site // 3)
        dfs = dfs.groupby("transcript", observed=True)
        dfs = dfs.agg(
            {
                "norm_counts": lambda x: x.tolist(),
                "codon_idx": lambda x: x.tolist(),
                "gene": "first",
            }
        ).reset_index()
        dfs = dfs.merge(df_trans_to_seq)

        dfs = dfs.assign(
            n_annot=lambda df: df.norm_counts.transform(lambda x: len(x))
            / (df.sequence.str.len() // 3)
        )

        dfs = dfs.assign(perc_annot=lambda df: df.n_annot / df.n_codons)

        # Filter by max sequence lenght
        dfs = dfs.query("n_codons<@max_n_codons")

        return dfs

    def sequence2codonids(seq):
        '''
        converts nt sequence into one-hot codon ids
        '''
        codon_ids = []
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            if len(codon) == 3:
                codon_ids.append(codon_to_id[codon])

        return codon_ids

    def process_merged_df(df):
        '''
        inputs: merged dataframe with ribosome data
        outputs: dataframe with the sequences linked to the ribosome read count annotations, with sequences consisting of N being removed
        '''
        df = df[df['sequence'].str.contains('N') == False]

        codon_seqs = []
        sequences = list(df['sequence'])
        genes = list(df['gene'])
        transcripts = list(df['transcript'])
        perc_non_zero_annots = []
        norm_counts = list(df['norm_counts'])
        codon_idx = list(df["codon_idx"])
        annot_seqs = []

        for i in tqdm(range(len(sequences))):
            seq = sequences[i]
            seq = sequence2codonids(seq)
            codon_seqs.append(seq)
            codon_idx_sample = codon_idx[i]
            norm_counts_sample = norm_counts[i]
            annot_seq_sample = []
            for j in range(len(seq)):
                if j in codon_idx_sample:
                    annot_seq_sample.append(norm_counts_sample[codon_idx_sample.index(j)])
                else:
                    annot_seq_sample.append(0.0)
            annot_seqs.append(annot_seq_sample)

            # calculate percentage of non-zero annotations
            perc_non_zero_annots.append(sum([1 for i in annot_seq_sample if i != 0.0])/len(annot_seq_sample))

        final_df = pd.DataFrame(list(zip(genes, transcripts, codon_seqs, annot_seqs, perc_non_zero_annots)), columns = ['gene', 'transcript', 'codon_sequence', 'annotations', 'perc_non_zero_annots'])

        return final_df

    def checkArrayEquality(arr1, arr2):
        '''
        inputs: two arrays
        outputs: True if the arrays are equal, False otherwise
        '''
        if len(arr1) != len(arr2):
            return False
    
        for i in range(len(arr1)):
            if arr1[i] != arr2[i]:
                return False
    
        return True

    def longestZeroSeqLength(a):
        '''
        length of the longest sub-sequence of zeros
        '''
        longest = 0
        current = 0
        for i in a:
            if i == 0.0:
                current += 1
            else:
                longest = max(longest, current)
                current = 0
        longest = max(longest, current)
        return longest

    def percNans(a):
        '''
        returns the percentage of nans in the sequence
        '''
        a = np.asarray(a)
        perc = np.count_nonzero(np.isnan(a)) / len(a)

        return perc

    def coverageMod(a, window_size=30):
        '''
        returns the coverage of a sequence, defined as the percentage of (non-zero + non-nan) over (non-nan) 
        '''
        a = a[1:-1].split(',')
        a = [float(k) for k in a]
        a = np.asarray(a)
        for i in range(len(a) - window_size):
            if np.all(a[i:i+window_size] == 0.0):
                a[i:i+window_size] = np.nan

        # num non zero, non nan
        num = 0
        den = 0
        for i in a:
            if i != 0.0 and not np.isnan(i):
                num += 1
            if not np.isnan(i):
                den += 1
    
        return num / den

    def sequenceLength(a):
        '''
        returns the length of the sequence
        '''
        return len(a)

    def mergeAnnotations(annots):
        '''
        merge the annotations for the same gene
        '''
        # merge the annotations
        merged_annots = []
        for i in range(len(annots[0])):
            # get the ith annotation for all the transcripts, only non zero and non nan
            ith_annots = [a[i] for a in annots if a[i] != 0.0 and not np.isnan(a[i])]
            # take the mean of the ith annotation
            ith_mean = np.mean(ith_annots)
            merged_annots.append(ith_mean)

        return merged_annots

    def uniqueGenes(df):
        '''
        processes an input dataframe to have only one transcript per gene
        '''
        df['sequence_length'] = df['annotations'].apply(sequenceLength)

        unique_genes = list(df['gene'].unique())

        # iterate through each gene, and choose the longest transcript, for the annotation, merge the annotations
        for gene in unique_genes:
            # get the df for the gene
            df_gene = df[df['gene'] == gene]
            if len(df_gene) > 1:
                # get the transcript with the longest sequence
                df_gene = df_gene.sort_values('sequence_length', ascending=False)
                # chosen transcript
                chosen_transcript = df_gene['transcript'].values[0]
                other_transcripts = df_gene['transcript'].values[1:]
                # merge the annotations
                annotations = df_gene['annotations'].values
                merged_annotations = mergeAnnotations(annotations)
                # drop the other transcripts from the df
                df = df[~df['transcript'].isin(other_transcripts)]

                # change the annotations for the chosen transcript
                df.loc[df['transcript'] == chosen_transcript, 'annotations'] = str(merged_annotations)

        # drop sequence length column
        df = df.drop(columns=['sequence_length'])

        assert len(df['gene'].unique()) == len(df['gene'])
        assert len(df['transcript'].unique()) == len(df['transcript'])
        assert len(df['transcript']) == len(df['gene'])

        return df
    
    def slidingWindowZeroToNan(a, window_size=30):
        '''
        use a sliding window, if all the values in the window are 0, then replace them with nan.
        this is done to assign nan values to zero-counts that are presumed to be artifacts of the sequencing process
        '''
        a = [float(k) for k in a]
        a = np.asarray(a)
        for i in range(len(a) - window_size):
            if np.all(a[i:i+window_size] == 0.0):
                a[i:i+window_size] = np.nan

        return a

    def RiboDatasetGWS(df_dict, threshold: float = 0.3, longZerosThresh: int = 20, percNansThresh: float = 0.05):
        '''
        inputs: dictionary of processed dataframes, coverage threshold, longest zero sequence length threshold, percentage of nans threshold
        outputs: train, validation, and test dataframes
        '''
        # Liver data (in CTRL condition)
        df_liver = df_dict['LIVER']
        df_liver['condition'] = 'CTRL'

        # Lina: CTRL data
        df_ctrl_depr = df_dict['CTRL']
        df_ctrl_depr['condition'] = 'CTRL'

        # merge the separate ctrl datasets
        tr_liver = df_liver['transcript'].unique()
        tr_ctrl_depr = df_ctrl_depr['transcript'].unique()
        tr_to_add = [g for g in tr_liver if g not in tr_ctrl_depr]
        df_liver = df_liver[df_liver['transcript'].isin(tr_to_add)]
        df_ctrldepr_liver = pd.concat([df_liver, df_ctrl_depr], axis=0)
        df_ctrldepr_liver = uniqueGenes(df_ctrldepr_liver)
        ctrl_genes_transcripts = list(zip(df_ctrldepr_liver['gene'], df_ctrldepr_liver['transcript']))
        ctrl_genes_transcripts = [[gene, transcript] for gene, transcript in ctrl_genes_transcripts]

        # Lina: ILE data
        df_ile = df_dict['ILE']
        df_ile['condition'] = 'ILE'
        df_ile = uniqueGenes(df_ile)
        for index, row in df_ile.iterrows():
            if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
                df_ile.drop(index, inplace=True) 

        # Lina: LEU data
        df_leu = df_dict['LEU']
        df_leu['condition'] = 'LEU'
        df_leu = uniqueGenes(df_leu)
        for index, row in df_leu.iterrows():
            if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
                df_leu.drop(index, inplace=True)

        # Lina: VAL data
        df_val = df_dict['VAL']
        df_val['condition'] = 'VAL'
        df_val = uniqueGenes(df_val)
        for index, row in df_val.iterrows():
            if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
                df_val.drop(index, inplace=True)

        # Lina: LEU_ILE data
        df_leu_ile = df_dict['LEU_ILE']
        df_leu_ile['condition'] = 'LEU_ILE'
        df_leu_ile = uniqueGenes(df_leu_ile)
        for index, row in df_leu_ile.iterrows():
            if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
                df_leu_ile.drop(index, inplace=True)

        # Lina: LEU_ILE_VAL data
        df_leu_ile_val = df_dict['LEU_ILE_VAL']
        df_leu_ile_val['condition'] = 'LEU_ILE_VAL'
        df_leu_ile_val = uniqueGenes(df_leu_ile_val)
        for index, row in df_leu_ile_val.iterrows():
            if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
                df_leu_ile_val.drop(index, inplace=True)

        # concenate all the data from the different conditions
        df_full = pd.concat([df_ctrldepr_liver, df_ile, df_leu, df_val, df_leu_ile, df_leu_ile_val], axis=0) 
        df_full.columns = ['gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots', 'condition']

        # sanity check to see if the number of unique genes is equal to the number of unique transcripts
        assert len(df_full['transcript'].unique()) == len(df_full['gene'].unique())

        # apply coverage threshold
        df_full['coverage_mod'] = df_full['annotations'].apply(coverageMod)
        df_full = df_full[df_full['coverage_mod'] >= threshold]

        # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        sequences_ctrl = []
        annotations_list = list(df_full['annotations'])
        condition_df_list = list(df_full['condition'])
        genes_list = list(df_full['gene'])

        for i in range(len(condition_df_list)):
            try:
                if condition_df_list[i] != 'CTRL':
                    # find the respective CTRL sequence for the transcript
                    ctrl_sequence = df_full[(df_full['gene'] == genes_list[i]) & (df_full['condition'] == 'CTRL')]['annotations'].iloc[0]
                    sequences_ctrl.append(ctrl_sequence)
                else:
                    sequences_ctrl.append(annotations_list[i])
            except:
                sequences_ctrl.append('NA')

        # add the sequences_ctrl to the df
        df_full['ctrl_sequence'] = sequences_ctrl

        # remove those rows where the ctrl_sequence is NA
        df_full = df_full[df_full['ctrl_sequence'] != 'NA']

        # sanity check for the ctrl sequences
        # get the ds with only condition as CTRL
        df_ctrl_full = df_full[df_full['condition'] == 'CTRL']
        ctrl_sequences_san = list(df_ctrl_full['annotations'])
        ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        for i in range(len(ctrl_sequences_san)):
            assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        # add the longest zero sequence length to the df
        df_full['longest_zero_seq_length_annotation'] = df_full['annotations'].apply(longestZeroSeqLength)
        df_full['longest_zero_seq_length_ctrl_sequence'] = df_full['ctrl_sequence'].apply(longestZeroSeqLength)

        # add the number of nans to the df
        df_full['perc_nans_annotation'] = df_full['annotations'].apply(percNans)
        df_full['perc_nans_ctrl_sequence'] = df_full['ctrl_sequence'].apply(percNans)

        # apply the threshold for the longest zero sequence length
        df_full = df_full[df_full['longest_zero_seq_length_annotation'] <= longZerosThresh]
        df_full = df_full[df_full['longest_zero_seq_length_ctrl_sequence'] <= longZerosThresh]

        # apply the threshold for the number of nans
        df_full = df_full[df_full['perc_nans_annotation'] <= percNansThresh]
        df_full = df_full[df_full['perc_nans_ctrl_sequence'] <= percNansThresh]

        # Gene-Wise Split (GWS) for each condition
        genes = df_full['gene'].unique()
        gene_mean_coverage_mod = []
        for gene in genes:
            gene_mean_coverage_mod.append(df_full[df_full['gene'] == gene]['coverage_mod'].mean())

        gene_mean_coverage_mod = np.asarray(gene_mean_coverage_mod)
        genes = np.asarray(genes)

        # sort the genes by coverage_mod in descending order
        genes = genes[np.argsort(gene_mean_coverage_mod)[::-1]]

        num_test_genes = int(0.2 * len(genes))
        num_valid_genes = int(0.05 * len(genes))
    
        test_genes = []
        train_genes = []
        valid_genes = []

        for i in range(len(genes)):
            # alternating until 20% of the genes are in the test set, 5% in the val set
            # the rest are in the train set
            if i % 3 == 0 and len(test_genes) < num_test_genes:
                test_genes.append(genes[i])
            elif i % 3 == 1 and len(valid_genes) < num_valid_genes:
                valid_genes.append(genes[i])
            else:
                train_genes.append(genes[i])

        # split the dataframe
        df_train = df_full[df_full['gene'].isin(train_genes)]
        df_valid = df_full[df_full['gene'].isin(valid_genes)]
        df_test = df_full[df_full['gene'].isin(test_genes)]

        return df_train, df_valid, df_test
    return (
        RiboDatasetGWS,
        checkArrayEquality,
        codon_to_id,
        coverageMod,
        id_to_codon,
        longestZeroSeqLength,
        make_all_dataframes,
        make_dataframe,
        mergeAnnotations,
        percNans,
        process_merged_df,
        sequence2codonids,
        sequenceLength,
        slidingWindowZeroToNan,
        uniqueGenes,
    )


@app.cell
def _(
    DATA_FOLDER,
    LIVER_FOLDER,
    fa_path,
    here,
    make_all_dataframes,
    process_merged_df,
):
    conditions = ['CTRL', 'ILE', 'LEU', 'LEU_ILE', 'LEU_ILE_VAL', 'VAL', 'LIVER']
    df_dict = {}

    for cond in conditions:
        if cond == 'LIVER':
            dir_path = LIVER_FOLDER
        else:
            dir_path = f'{DATA_FOLDER}/{cond}/'
        df = make_all_dataframes(dir_path, fa_path)
        df_proc = process_merged_df(df)
        df_dict[cond] = df_proc

        print(f'{cond} done')

        # save the dataframe
        out_cond_path = here('data', 'processed', str(cond) + '.csv')
        df_proc.to_csv(out_cond_path, index=False)
    return cond, conditions, df, df_dict, df_proc, dir_path, out_cond_path


@app.cell
def _(RiboDatasetGWS, df_dict):
    df_train, df_valid, df_test = RiboDatasetGWS(df_dict)
    return df_test, df_train, df_valid


@app.cell
def _(df_test, df_train, df_valid, here):
    train_out_path = here('data', 'orig', 'train.csv')
    valid_out_path = here('data', 'orig', 'valid.csv')
    test_out_path = here('data', 'orig', 'test.csv')
    # save the dataframes
    df_train.to_csv(train_out_path, index=False)
    df_valid.to_csv(valid_out_path, index=False)
    df_test.to_csv(test_out_path, index=False)
    return test_out_path, train_out_path, valid_out_path


if __name__ == "__main__":
    app.run()
