import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    import numpy as np
    import pandas as pd 
    import torch
    from transformers import XLNetConfig, XLNetForTokenClassification
    import itertools
    from tqdm import tqdm
    from pyhere import here
    return (
        XLNetConfig,
        XLNetForTokenClassification,
        here,
        itertools,
        np,
        pd,
        torch,
        tqdm,
    )


@app.cell
def _():
    # Cell tags: parameters
    threshold = 0.3
    longZerosThresh = 20
    percNansThresh = 0.05
    return longZerosThresh, percNansThresh, threshold


@app.cell
def _(itertools):
    # conditions
    conditions_list = ['CTRL', 'LEU', 'ILE', 'VAL', 'LEU_ILE', 'LEU_ILE_VAL']
    condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}
    id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id = {v:k for k,v in id_to_codon.items()}
    return codon_to_id, condition_values, conditions_list, id_to_codon


@app.cell
def _(codon_to_id, condition_values, np):
    def pseudolabel(ground_truth, mean_preds):
        if ground_truth == 'NA':
            ground_truth = [np.nan for _j in range(len(mean_preds))]
        else:
            ground_truth = [float(k) for k in ground_truth]
        annot = []
        for _j in range(len(mean_preds)):
            if np.isnan(ground_truth[_j]) or ground_truth[_j] == 0.0:
                annot.append(np.abs(mean_preds[_j]))
            else:
                annot.append(ground_truth[_j])
        return annot

    def longestZeroSeqLength(a):
        """
        length of the longest sub-sequence of zeros
        """
        a = [float(k) for k in a]
        longest = 0
        current = 0
        for _i in a:
            if _i == 0.0:
                current = current + 1
            else:
                longest = max(longest, current)
                current = 0
        longest = max(longest, current)
        return longest

    def percNans(a):
        """
        returns the percentage of nans in the sequence
        """
        a = [float(k) for k in a]
        a = np.asarray(a)
        perc = np.count_nonzero(np.isnan(a)) / len(a)
        return perc

    def coverageMod(a, window_size=30):
        """
        returns the modified coverage function val in the sequence
        """
        a = [float(k) for k in a]
        for _i in range(len(a) - window_size):
            if np.all(a[_i:_i + window_size] == 0.0):
                a[_i:_i + window_size] = np.nan
        num = 0
        den = 0
        for _i in a:
            if _i != 0.0 and (not np.isnan(_i)):
                num = num + 1
            if not np.isnan(_i):
                den = den + 1
        if den == 0:
            return 0
        return num / den

    def ntseqtoCodonSeq(seq, condition, add_cond=True):
        """
        Convert nucleotide sequence to codon sequence
        """
        codon_seq = []
        for _i in range(0, len(seq), 3):
            if len(seq[_i:_i + 3]) == 3:
                codon_seq.append(seq[_i:_i + 3])
        codon_seq = [codon_to_id[codon] for codon in codon_seq]
        if add_cond:
            codon_seq = [condition_values[condition]] + codon_seq
        return codon_seq

    def sequenceLength(a):
        """
        returns the length of the sequence
        """
        a = [float(k) for k in a]
        return len(a)

    def mergeAnnotations(annots):
        """
        merge the annotations for the same gene
        """
        annots = [a[1:-1].split(', ') for a in annots]
        annots = [[float(k) for k in a] for a in annots]
        merged_annots = []
        for _i in range(len(annots[0])):
            ith_annots = [a[_i] for a in annots if a[_i] != 0.0 and (not np.isnan(a[_i]))]
            ith_mean = np.mean(ith_annots)
            merged_annots.append(ith_mean)
        return merged_annots

    def uniqueGenes(df):
        df['sequence_length'] = df['annotations'].apply(sequenceLength)
        unique_genes = list(df['gene'].unique())
        for gene in unique_genes:
            df_gene = df[df['gene'] == gene]
            if len(df_gene) > 1:
                df_gene = df_gene.sort_values('sequence_length', ascending=False)
                chosen_transcript = df_gene['transcript'].values[0]
                other_transcripts = df_gene['transcript'].values[1:]
                annotations = df_gene['annotations'].values
                merged_annotations = mergeAnnotations(annotations)
                df = df[~df['transcript'].isin(other_transcripts)]
                df.loc[df['transcript'] == chosen_transcript, 'annotations'] = str(merged_annotations)
        df = df.drop(columns=['sequence_length'])
        assert len(df['gene'].unique()) == len(df['gene'])
        assert len(df['transcript'].unique()) == len(df['transcript'])
        assert len(df['transcript']) == len(df['gene'])
        return df

    def seqLen(a):
        """
        returns the length of the sequence
        """
        return len(a)

    def removeFullGenes(df_mouse, df_full):
        """
        remove the genes that are already in df_full
        """
        tr_unique_full = list(df_full['transcript'].unique())
        transcripts_full_sans_version = [tr.split('.')[0] for tr in tr_unique_full]
        df_mouse_tr_sans_version = [tr.split('.')[0] for tr in df_mouse['transcript']]
        df_mouse_genes = list(df_mouse['gene'])
        mouse_tg_dict = dict(zip(df_mouse_tr_sans_version, df_mouse_genes))
        for tran in transcripts_full_sans_version:
            mouse_gene_for_full_transcript = mouse_tg_dict[tran]
            df_mouse = df_mouse[df_mouse['gene'] != mouse_gene_for_full_transcript]
        df_mouse['sequence_length'] = df_mouse['sequence'].apply(seqLen)
        df_mouse = df_mouse.sort_values('sequence_length', ascending=False).drop_duplicates('gene')
        df_mouse = df_mouse.drop(columns=['sequence_length'])
        return df_mouse
    return (
        coverageMod,
        longestZeroSeqLength,
        mergeAnnotations,
        ntseqtoCodonSeq,
        percNans,
        pseudolabel,
        removeFullGenes,
        seqLen,
        sequenceLength,
        uniqueGenes,
    )


@app.cell
def _():
    # model parameters
    annot_thresh = 0.3
    longZerosThresh_val = 20
    percNansThresh_val = 0.05
    d_model_val = 512
    n_layers_val = 3
    n_heads_val = 4
    dropout_val = 0.1
    lr_val = 1e-4
    batch_size_val = 1
    loss_fun_name = '4L' # 4L, 5L
    return (
        annot_thresh,
        batch_size_val,
        d_model_val,
        dropout_val,
        longZerosThresh_val,
        loss_fun_name,
        lr_val,
        n_heads_val,
        n_layers_val,
        percNansThresh_val,
    )


@app.cell
def _(
    XLNetConfig,
    XLNetForTokenClassification,
    d_model_val,
    dropout_val,
    here,
    n_heads_val,
    n_layers_val,
    torch,
):
    model_name1 = here('checkpoints', 'XLNet-DH_S1')
    model_name2 = here('checkpoints', 'XLNet-DH_S2')
    model_name3 = here('checkpoints', 'XLNet-DH_S3')
    model_name4 = here('checkpoints', 'XLNet-DH_S4')
    model_name42 = here('checkpoints', 'XLNet-DH_S4_2')

    class XLNetDH(XLNetForTokenClassification):

        def __init__(self, config):
            super().__init__(config)
            self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)
    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model=d_model_val, n_layer=n_layers_val, n_head=n_heads_val, d_inner=d_model_val, num_labels=1, dropout=dropout_val)
    model = XLNetDH(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model1 = model.from_pretrained(model_name1 + '/best_model')
    model2 = model.from_pretrained(model_name2 + '/best_model')
    model3 = model.from_pretrained(model_name3 + '/best_model')
    model4 = model.from_pretrained(model_name4 + '/best_model')
    model42 = model.from_pretrained(model_name42 + '/best_model')
    models_list = [model1, model2, model3, model4, model42]
    for _model_chosen in models_list:
        _model_chosen.to(device)
        _model_chosen.eval()
    print('Loaded all the models')
    return (
        XLNetDH,
        config,
        device,
        model,
        model1,
        model2,
        model3,
        model4,
        model42,
        model_name1,
        model_name2,
        model_name3,
        model_name4,
        model_name42,
        models_list,
    )


@app.cell
def _(here):
    depr_folder = here('data', 'processed')

    ctrl_depr_path = depr_folder + 'CTRL.csv'
    ile_path = depr_folder + 'ILE.csv'
    leu_path = depr_folder + 'LEU.csv'
    val_path = depr_folder + 'VAL.csv'
    leu_ile_path = depr_folder + 'LEU_ILE.csv'
    leu_ile_val_path = depr_folder + 'LEU_ILE_VAL.csv'
    liver_path = depr_folder + 'LIVER.csv'
    return (
        ctrl_depr_path,
        depr_folder,
        ile_path,
        leu_ile_path,
        leu_ile_val_path,
        leu_path,
        liver_path,
        val_path,
    )


@app.cell
def _(
    ctrl_depr_path,
    ile_path,
    leu_ile_path,
    leu_ile_val_path,
    leu_path,
    liver_path,
    pd,
    uniqueGenes,
    val_path,
):
    # load the control liver data
    df_liver = pd.read_csv(liver_path)
    df_liver['condition'] = 'CTRL'

    # load ctrl_aa data
    df_ctrl_depr = pd.read_csv(ctrl_depr_path)
    df_ctrl_depr['condition'] = 'CTRL'

    # add to the liver data the genes from ctrl depr which are not in liver
    tr_liver = df_liver['transcript'].unique()
    tr_ctrl_depr = df_ctrl_depr['transcript'].unique()
    tr_to_add = [g for g in tr_liver if g not in tr_ctrl_depr]

    df_liver = df_liver[df_liver['transcript'].isin(tr_to_add)]

    # df ctrldepr without liver intersection
    df_ctrldepr_liver = pd.concat([df_liver, df_ctrl_depr], axis=0)

    # unique genes
    df_ctrldepr_liver = uniqueGenes(df_ctrldepr_liver)

    # get ctrl gene, transcript tuple pairs from the df_ctrldepr_liver
    ctrl_genes_transcripts = list(zip(df_ctrldepr_liver['gene'], df_ctrldepr_liver['transcript']))
    # make a list of lists
    ctrl_genes_transcripts = [[gene, transcript] for gene, transcript in ctrl_genes_transcripts]

    print("CTRL Done")

    # other conditions
    df_ile = pd.read_csv(ile_path)
    df_ile['condition'] = 'ILE'
    # unique genes
    df_ile = uniqueGenes(df_ile)
    # only choose those genes+transcripts that are in ctrl_depr_liver
    # iterate through the df_ile and choose those genes that are in ctrl_genes_transcripts
    for index, row in df_ile.iterrows():
        if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
            df_ile.drop(index, inplace=True) 

    print("ILE Done")

    df_leu = pd.read_csv(leu_path)
    df_leu['condition'] = 'LEU'
    # unique genes
    df_leu = uniqueGenes(df_leu)
    # choose those transcripts that are in ctrl_depr_liver
    for index, row in df_leu.iterrows():
        if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
            df_leu.drop(index, inplace=True)

    print("LEU Done")

    df_val = pd.read_csv(val_path)
    df_val['condition'] = 'VAL'
    # unique genes
    df_val = uniqueGenes(df_val)
    # choose those transcripts that are in ctrl_depr_liver
    for index, row in df_val.iterrows():
        if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
            df_val.drop(index, inplace=True)

    print("VAL Done")

    df_leu_ile = pd.read_csv(leu_ile_path)
    df_leu_ile['condition'] = 'LEU_ILE'
    # unique genes
    df_leu_ile = uniqueGenes(df_leu_ile)
    # choose those transcripts that are in ctrl_depr_liver
    for index, row in df_leu_ile.iterrows():
        if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
            df_leu_ile.drop(index, inplace=True)

    print("LEU_ILE Done")

    df_leu_ile_val = pd.read_csv(leu_ile_val_path)
    df_leu_ile_val['condition'] = 'LEU_ILE_VAL'
    # unique genes
    df_leu_ile_val = uniqueGenes(df_leu_ile_val)
    # choose those transcripts that are in ctrl_depr_liver
    for index, row in df_leu_ile_val.iterrows():
        if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
            df_leu_ile_val.drop(index, inplace=True)

    print("LEU_ILE_VAL Done")

    df_full = pd.concat([df_ctrldepr_liver, df_ile, df_leu, df_val, df_leu_ile, df_leu_ile_val], axis=0) # liver + ctrl depr + ile + leu + val + leu ile + leu ile val

    df_full.columns = ['index_val', 'gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots', 'condition']

    # drop index_val column
    df_full = df_full.drop(columns=['index_val'])
    return (
        ctrl_genes_transcripts,
        df_ctrl_depr,
        df_ctrldepr_liver,
        df_full,
        df_ile,
        df_leu,
        df_leu_ile,
        df_leu_ile_val,
        df_liver,
        df_val,
        index,
        row,
        tr_ctrl_depr,
        tr_liver,
        tr_to_add,
    )


@app.cell
def _(conditions_list, df_full, here, pd):
    df_set1 = df_full[['gene', 'transcript', 'sequence']]
    df_set1 = df_set1.drop_duplicates()
    len_sf_set1 = len(df_set1)
    df_set1 = pd.concat([df_set1] * 6, ignore_index=True)
    cond_col = []
    for _i in range(6):
        for _j in range(len_sf_set1):
            cond_col.append(conditions_list[_i])
    df_set1['condition'] = cond_col
    df_set1_path = here('data/plabel', 'df_set1.csv')
    df_set1.to_csv(df_set1_path, index=False)
    return cond_col, df_set1, df_set1_path, len_sf_set1


@app.cell
def _(
    condition_values,
    device,
    df_full,
    df_set1,
    models_list,
    np,
    pd,
    torch,
    tqdm,
):
    final_mean_preds_list_set1 = []
    final_stds_preds_list_set1 = []
    final_conditions_list_set1 = []
    final_genes_list_set1 = []
    final_transcripts_list_set1 = []
    final_sequence_list_set1 = []
    final_annots_list_set1 = []
    sequences_df_set1 = list(df_set1['sequence'])
    genes_df_set1 = list(df_set1['gene'])
    transcripts_df_set1 = list(df_set1['transcript'])
    conditions_df_set1 = list(df_set1['condition'])
    for _j in tqdm(range(len(sequences_df_set1))):
        X = sequences_df_set1[_j]
        X = X[1:-1].split(', ')
        X = [int(k) for k in X]
        X = [condition_values[conditions_df_set1[_j]]] + X
        X = np.asarray(X)
        X = torch.from_numpy(X).long()
        preds_list_sample = []
        with torch.no_grad():
            for _model_chosen in models_list:
                y_pred = _model_chosen(X.unsqueeze(0).to(device).to(torch.int32))
                y_pred = torch.sum(y_pred['logits'], dim=2)
                y_pred = y_pred.squeeze(0)
                y_pred = y_pred[1:]
                preds_list_sample.append(y_pred.detach().cpu().numpy())
        preds_list_sample = np.asarray(preds_list_sample)
        mean_preds = np.mean(preds_list_sample, axis=0)
        stds_preds = np.std(preds_list_sample, axis=0)
        df_full_sample = df_full[df_full['condition'] == conditions_df_set1[_j]]
        df_full_sample = df_full_sample[df_full_sample['transcript'] == transcripts_df_set1[_j]]
        if len(df_full_sample) > 0:
            annots_sample = df_full_sample['annotations'].values[0]
            final_annots_list_set1.append(annots_sample)
        else:
            final_annots_list_set1.append('NA')
        final_mean_preds_list_set1.append(mean_preds)
        final_stds_preds_list_set1.append(stds_preds)
        final_conditions_list_set1.append(conditions_df_set1[_j])
        final_genes_list_set1.append(genes_df_set1[_j])
        final_transcripts_list_set1.append(transcripts_df_set1[_j])
        final_sequence_list_set1.append(sequences_df_set1[_j])
    df_final_preds = pd.DataFrame({'gene': final_genes_list_set1, 'transcript': final_transcripts_list_set1, 'sequence': final_sequence_list_set1, 'mean_preds': final_mean_preds_list_set1, 'stds_preds': final_stds_preds_list_set1, 'condition': final_conditions_list_set1, 'annotations': final_annots_list_set1})
    return (
        X,
        annots_sample,
        conditions_df_set1,
        df_final_preds,
        df_full_sample,
        final_annots_list_set1,
        final_conditions_list_set1,
        final_genes_list_set1,
        final_mean_preds_list_set1,
        final_sequence_list_set1,
        final_stds_preds_list_set1,
        final_transcripts_list_set1,
        genes_df_set1,
        mean_preds,
        preds_list_sample,
        sequences_df_set1,
        stds_preds,
        transcripts_df_set1,
        y_pred,
    )


@app.cell
def _(
    coverageMod,
    df_final_preds,
    here,
    longZerosThresh,
    longestZeroSeqLength,
    pd,
    percNans,
    percNansThresh,
    pseudolabel,
    threshold,
    tqdm,
    val_path,
):
    test_path = here('data/orig', 'test.csv')
    train_path = here('data/orig', 'train.csv')
    df_test_orig = pd.read_csv(test_path)
    df_val_orig = pd.read_csv(val_path)
    orig_test_genes = list(set(list(df_test_orig['gene'])))
    orig_test_transcripts = list(set(list(df_test_orig['transcript'])))
    orig_val_genes = list(set(list(df_val_orig['gene'])))
    orig_val_transcripts = list(set(list(df_val_orig['transcript'])))
    df_final_preds_1 = df_final_preds[~df_final_preds['gene'].isin(orig_test_genes)]
    df_final_preds_1 = df_final_preds_1[~df_final_preds_1['transcript'].isin(orig_test_transcripts)]
    df_final_preds_1 = df_final_preds_1[~df_final_preds_1['gene'].isin(orig_val_genes)]
    df_final_preds_1 = df_final_preds_1[~df_final_preds_1['transcript'].isin(orig_val_transcripts)]
    annots_imputed = []
    for _i in tqdm(range(len(df_final_preds_1))):
        ground_truth_sample = df_final_preds_1['annotations'].iloc[_i]
        mean_preds_sample = df_final_preds_1['mean_preds'].iloc[_i]
        pred_sample = pseudolabel(ground_truth_sample, mean_preds_sample)
        annots_imputed.append(pred_sample)
    df_final_preds_1['annotations'] = annots_imputed
    df_final_preds_1 = df_final_preds_1[df_final_preds_1['annotations'] != 'NA']
    df_final_preds_1['coverage_mod'] = df_final_preds_1['annotations'].apply(coverageMod)
    df_final_preds_1 = df_final_preds_1[df_final_preds_1['coverage_mod'] >= threshold]
    df_final_preds_1['longest_zero_seq_length_annotation'] = df_final_preds_1['annotations'].apply(longestZeroSeqLength)
    df_final_preds_1['perc_nans_annotation'] = df_final_preds_1['annotations'].apply(percNans)
    df_final_preds_1 = df_final_preds_1[df_final_preds_1['longest_zero_seq_length_annotation'] <= longZerosThresh]
    df_final_preds_1 = df_final_preds_1[df_final_preds_1['perc_nans_annotation'] <= percNansThresh]
    print('Added Thresholds on all the factors')
    sequences_ctrl = []
    annotations_list = list(df_final_preds_1['annotations'])
    condition_df_list = list(df_final_preds_1['condition'])
    transcripts_list = list(df_final_preds_1['transcript'])
    for _i in tqdm(range(len(condition_df_list))):
        try:
            if condition_df_list[_i] != 'CTRL':
                ctrl_sequence = df_final_preds_1[(df_final_preds_1['transcript'] == transcripts_list[_i]) & (df_final_preds_1['condition'] == 'CTRL')]['annotations'].iloc[0]
                sequences_ctrl.append(ctrl_sequence)
            else:
                sequences_ctrl.append(annotations_list[_i])
        except:
            sequences_ctrl.append('NA')
    df_final_preds_1['ctrl_sequence'] = sequences_ctrl
    df_final_preds_1 = df_final_preds_1[df_final_preds_1['ctrl_sequence'] != 'NA']
    df_ctrl_full = df_final_preds_1[df_final_preds_1['condition'] == 'CTRL']
    ctrl_sequences_san = list(df_ctrl_full['annotations'])
    ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])
    for _i in range(len(ctrl_sequences_san)):
        assert ctrl_sequences_san[_i] == ctrl_sequences_san2[_i]
    print('Sanity Checked')
    out_train_path = here('data/plabel', 'plabel_train.csv')
    df_final_preds_1.to_csv(out_train_path)
    return (
        annotations_list,
        annots_imputed,
        condition_df_list,
        ctrl_sequence,
        ctrl_sequences_san,
        ctrl_sequences_san2,
        df_ctrl_full,
        df_final_preds_1,
        df_test_orig,
        df_val_orig,
        ground_truth_sample,
        mean_preds_sample,
        orig_test_genes,
        orig_test_transcripts,
        orig_val_genes,
        orig_val_transcripts,
        out_train_path,
        pred_sample,
        sequences_ctrl,
        test_path,
        train_path,
        transcripts_list,
    )


if __name__ == "__main__":
    app.run()
