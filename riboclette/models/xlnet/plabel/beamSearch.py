import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    import torch
    from transformers import XLNetConfig, XLNetForTokenClassification
    from utils import GWSDatasetFromPandas 
    import itertools
    import pandas as pd
    from tqdm import tqdm
    from pyhere import here
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        GWSDatasetFromPandas,
        XLNetConfig,
        XLNetForTokenClassification,
        device,
        here,
        itertools,
        np,
        pd,
        torch,
        tqdm,
        warnings,
    )


@app.cell
def _(itertools):
    stop_codons = ['TAA', 'TAG', 'TGA']
    id_to_codon = {idx: ''.join(el) for (idx, el) in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id = {v: k for (k, v) in id_to_codon.items()}
    codonid_list = []
    for _i in range(64):
        codon = id_to_codon[_i]
        if codon not in stop_codons:
            codonid_list.append(_i)
    print('Number of codons:', len(codonid_list))
    return codon, codon_to_id, codonid_list, id_to_codon, stop_codons


@app.cell
def _(XLNetConfig, XLNetForTokenClassification, device, here, torch):
    # model parameters
    annot_thresh = 0.3
    longZerosThresh_val = 20
    percNansThresh_val = 0.05
    d_model_val = 512
    n_layers_val = 6
    n_heads_val = 4
    dropout_val = 0.1
    lr_val = 1e-4
    batch_size_val = 2
    loss_fun_name = '4L' # 5L
 
    model_loc = here('checkpoints', 'XLNet-PLabelDH_S2')

    condition_dict_values = {64: 'CTRL', 65: 'ILE', 66: 'LEU', 67: 'LEU_ILE', 68: 'LEU_ILE_VAL', 69: 'VAL'}
    condition_dict = {v: k for k, v in condition_dict_values.items()}

    class XLNetDH(XLNetForTokenClassification):
        def __init__(self, config):
            super().__init__(config)
            self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
    model = XLNetDH(config)

    # load model from the saved model
    model = model.from_pretrained(model_loc + "/best_model")
    model.to(device)
    # set model to evaluation mode
    model.eval()
    return (
        XLNetDH,
        annot_thresh,
        batch_size_val,
        condition_dict,
        condition_dict_values,
        config,
        d_model_val,
        dropout_val,
        longZerosThresh_val,
        loss_fun_name,
        lr_val,
        model,
        model_loc,
        n_heads_val,
        n_layers_val,
        percNansThresh_val,
    )


@app.cell
def _(GWSDatasetFromPandas, here, pd):
    # convert pandas dataframes into torch datasets
    train_path = here('data', 'orig', 'train.csv')
    val_path = here('data', 'orig', 'val.csv')
    test_path = here('data', 'orig', 'test.csv')

    train_dataset = pd.read_csv(train_path)
    val_dataset = pd.read_csv(val_path)
    test_dataset = pd.read_csv(test_path)

    # merge the datasets
    merged_dataset = pd.concat([train_dataset, val_dataset, test_dataset], ignore_index=True)

    # create the datasets
    merged_dataset = GWSDatasetFromPandas(merged_dataset)

    print("samples in merged dataset: ", len(merged_dataset))
    return (
        merged_dataset,
        test_dataset,
        test_path,
        train_dataset,
        train_path,
        val_dataset,
        val_path,
    )


@app.cell
def _():
    # Cell tags: parameters
    num_windows = 1000
    model_bs = 256
    window_size = 21
    return model_bs, num_windows, window_size


@app.cell
def _(condition_dict_values, merged_dataset, np, tqdm, window_size):
    a_pos = int(window_size / 2)
    conditions_windows = {'CTRL': [], 'ILE': [], 'LEU': [], 'LEU_ILE': [], 'LEU_ILE_VAL': [], 'VAL': []}
    conditions_windows_fin = {'CTRL': [], 'ILE': [], 'LEU': [], 'LEU_ILE': [], 'LEU_ILE_VAL': [], 'VAL': []}
    for _i in tqdm(range(len(merged_dataset))):
        sample_condition = merged_dataset[_i][0][0].item()
        g = merged_dataset[_i][3]
        t = merged_dataset[_i][4]
        y_true_sample = merged_dataset[_i][1].numpy()
        x_input_sample = merged_dataset[_i][0].numpy()
        if len(y_true_sample) > 500:
            continue
        non_peak = np.nanmean(y_true_sample) - np.nanstd(y_true_sample)
        for j in range(window_size, len(y_true_sample) - window_size):
            sample_window = x_input_sample[j:j + window_size]
            if len(sample_window) == window_size:
                if sample_window[a_pos] < non_peak:
                    conditions_windows[condition_dict_values[sample_condition]].append((g, t, x_input_sample, j))
    return (
        a_pos,
        conditions_windows,
        conditions_windows_fin,
        g,
        j,
        non_peak,
        sample_condition,
        sample_window,
        t,
        x_input_sample,
        y_true_sample,
    )


@app.cell
def _(conditions_windows, conditions_windows_fin, np, num_windows):
    for _condition in conditions_windows:
        np.random.shuffle(conditions_windows[_condition])
        conditions_windows_fin[_condition] = conditions_windows[_condition][:num_windows]
    return


@app.cell
def _(
    a_pos,
    codonid_list,
    condition_dict,
    device,
    itertools,
    model,
    model_bs,
    np,
    torch,
    window_size,
):
    def AsiteDensity(windows, condition, start):
        condition_val = condition_dict[_condition]
        windows = np.insert(windows, 0, condition_val, axis=1)
        windows = torch.tensor(windows).to(device)
        with torch.no_grad():
            for _i in range(0, windows.shape[0], model_bs):
                pred = model(windows[_i:_i + model_bs])
                if _i == 0:
                    pred_out = pred['logits'][:, 1:, :]
                else:
                    pred_out = torch.cat((pred_out, pred['logits'][:, 1:, :]), 0)
        ctrl = torch.relu(pred_out[:, :, 0])
        dd = pred_out[:, :, 1]
        dd_out = dd[:, start + a_pos]
        ctrl_out = ctrl[:, start + a_pos]
        if condition_val == 64:
            return ctrl_out
        else:
            return dd_out

    def getTopXMutants(full_inp, start, condition, X, mutant_pos1=-1, c_pos1=None, mutant_pos2=-1, c_pos2=None):
        window_density = {}
        if mutant_pos1 == -1 and mutant_pos2 == -1:
            inputs_all = []
            substs_all = []
            for k in range(window_size):
                for c in codonid_list:
                    input_copy = full_inp.copy()
                    input_copy[start + k] = c
                    inputs_all.append(input_copy)
                    substs_all.append((start + k, c))
            inputs_all = np.array(inputs_all)
            preds = AsiteDensity(inputs_all, _condition, start)
            for l in range(len(substs_all)):
                window_density[substs_all[l][0], substs_all[l][1]] = preds[l].item()
        elif mutant_pos1 != -1 and mutant_pos2 == -1:
            inputs_all = []
            substs_all = []
            for k in range(window_size):
                if k + start == mutant_pos1:
                    continue
                for c in codonid_list:
                    input_copy = full_inp.copy()
                    input_copy[start + k] = c
                    input_copy[mutant_pos1] = c_pos1
                    inputs_all.append(input_copy)
                    substs_all.append((start + k, c))
            inputs_all = np.array(inputs_all)
            preds = AsiteDensity(inputs_all, _condition, start)
            for l in range(len(substs_all)):
                window_density[mutant_pos1, c_pos1, substs_all[l][0], substs_all[l][1]] = preds[l].item()
        elif mutant_pos1 != -1 and mutant_pos2 != -1:
            inputs_all = []
            substs_all = []
            for k in range(window_size):
                if k + start == mutant_pos1 or k + start == mutant_pos2:
                    continue
                for c in codonid_list:
                    input_copy = full_inp.copy()
                    input_copy[start + k] = c
                    input_copy[mutant_pos1] = c_pos1
                    input_copy[mutant_pos2] = c_pos2
                    inputs_all.append(input_copy)
                    substs_all.append((start + k, c))
            inputs_all = np.array(inputs_all)
            preds = AsiteDensity(inputs_all, _condition, start)
            for l in range(len(substs_all)):
                window_density[mutant_pos2, c_pos2, mutant_pos1, c_pos1, substs_all[l][0], substs_all[l][1]] = preds[l].item()
        window_density = dict(sorted(window_density.items(), key=lambda item: item[1], reverse=True))
        window_density = dict(itertools.islice(window_density.items(), X))
        return window_density
    return AsiteDensity, getTopXMutants


@app.cell
def _(AsiteDensity, conditions_windows_fin, getTopXMutants, tqdm, window_size):
    mutations_everything = {}
    num_mutants = 5
    for _condition in conditions_windows_fin:
        for sample in tqdm(conditions_windows_fin[_condition]):
            sample_mutations = {}
            window = sample[2][sample[3]:sample[3] + window_size]
            original_density = AsiteDensity(sample[2], _condition, sample[3]).item()
            mutants_one = getTopXMutants(sample[2], sample[3], _condition, num_mutants)
            for x in mutants_one:
                sample_mutations[x] = mutants_one[x]
            for mutant in mutants_one:
                mutant_pos1 = mutant[0]
                c_pos1 = mutant[1]
                mutants_two = getTopXMutants(sample[2], sample[3], _condition, num_mutants, mutant_pos1, c_pos1)
                for x in mutants_two:
                    sample_mutations[x] = mutants_two[x]
                for mutant2 in mutants_two:
                    mutant_pos2 = mutant2[2]
                    c_pos2 = mutant2[3]
                    mutants_three = getTopXMutants(sample[2], sample[3], _condition, num_mutants, mutant_pos1, c_pos1, mutant_pos2, c_pos2)
                    for x in mutants_three:
                        sample_mutations[x] = mutants_three[x]
            mutations_everything[sample[0], sample[1], sample[3], str(window), _condition, original_density] = sample_mutations
    return (
        c_pos1,
        c_pos2,
        mutant,
        mutant2,
        mutant_pos1,
        mutant_pos2,
        mutants_one,
        mutants_three,
        mutants_two,
        mutations_everything,
        num_mutants,
        original_density,
        sample,
        sample_mutations,
        window,
        x,
    )


@app.cell
def _(here, mutations_everything, np, num_windows, window_size):
    # save the mutations to a file
    out_motifs_path = here('data', 'motifs', 'motifs_' + str(window_size) + '_' + str(num_windows) + '.npz')
    np.savez(out_motifs_path, mutations_everything=mutations_everything)
    return (out_motifs_path,)


if __name__ == "__main__":
    app.run()
