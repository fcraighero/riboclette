import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    import numpy as np
    import pandas as pd
    import torch
    from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
    from transformers import XLNetConfig, XLNetForTokenClassification
    from xlnet_plabel_utils import GWSDatasetFromPandas  # custom dataset and trainer, CorrCoef, collate_fn, compute_metrics, compute_metrics_saved  # custom dataset and trainer
    import pytorch_lightning as pl
    from tqdm import tqdm
    import os
    from pyhere import here
    # suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        GWSDatasetFromPandas,
        LayerGradientXActivation,
        LayerIntegratedGradients,
        XLNetConfig,
        XLNetForTokenClassification,
        device,
        here,
        np,
        os,
        pd,
        pl,
        torch,
        tqdm,
        warnings,
    )


@app.cell
def _():
    # Cell tags: parameters
    gbs = 4
    return (gbs,)


@app.cell
def _(LayerIntegratedGradients, device, gbs, np, torch, tqdm):
    class model_CTRL(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
    
        def forward(self, x, index_val):
            # input dict
            out_batch = {}

            out_batch["input_ids"] = x.unsqueeze(0)
            for k, v in out_batch.items():
                out_batch[k] = v.to(device)

            out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)
            out_batch["input_ids"] = out_batch["input_ids"].squeeze(0)
            pred = self.model(out_batch["input_ids"])

            # get dim 0
            pred_fin = torch.relu(pred["logits"][:, :, 0])

            # set output to be values in each examples at index_val
            out_tensor = torch.zeros(len(index_val))
            for el, val in enumerate(index_val):
                out_tensor[el] = pred_fin[el][val]

            return out_tensor
    
    class model_DD(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
    
        def forward(self, x, index_val):
            # input dict
            out_batch = {}

            out_batch["input_ids"] = x.unsqueeze(0)
            for k, v in out_batch.items():
                out_batch[k] = v.to(device)

            out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)
            out_batch["input_ids"] = out_batch["input_ids"].squeeze(0)
            pred = self.model(out_batch["input_ids"])

            # get dim 1
            pred_fin = pred["logits"][:, :, 1]

            # set output to be values in each examples at index_val
            out_tensor = torch.zeros(len(index_val))
            for el, val in enumerate(index_val):
                out_tensor[el] = pred_fin[el][val]

            return out_tensor

    def lig_output(model, x, y, mode='ctrl'):
        if mode == 'ctrl':
            model_fin = model_CTRL(model)
        elif mode == 'dd':
            model_fin = model_DD(model)
        
        lig = LayerIntegratedGradients(model_fin, model_fin.model.transformer.word_embedding)

        # set torch graph to allow unused tensors
        with torch.autograd.set_detect_anomaly(True):    
            # get all indices
            len_sample = len(x)
            attributions_sample = np.zeros((len_sample, len_sample))

            for j in tqdm(range(0, len_sample, gbs)):
                index_val = list(range(j, min(j+gbs, len_sample)))

                index_val = torch.tensor(index_val).to(device)

                out_batch = {}

                out_batch["input_ids"] = x
            
                out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

                baseline_inp = torch.ones(out_batch["input_ids"].shape) * 70 # 70 is the padding token
                baseline_inp = baseline_inp.to(device).to(torch.int32)

                # repeat the input and baseline tensors
                out_batch["input_ids"] = out_batch["input_ids"].repeat(len(index_val), 1)
                baseline_inp = baseline_inp.repeat(len(index_val), 1)

                attributions = lig.attribute((out_batch["input_ids"]), baselines=baseline_inp, 
                                                        method = 'gausslegendre', return_convergence_delta = False, additional_forward_args=index_val, n_steps=10, internal_batch_size=gbs)

                attributions = torch.permute(attributions, (1, 0, 2))
                attributions = torch.sum(attributions, dim=2)

                # norm the attributions per example
                for ex in range(attributions.shape[0]):
                    attributions[ex] = attributions[ex] / torch.norm(attributions[ex])
                attributions = attributions.detach().cpu().numpy()
                attributions_sample[j:j+len(index_val)] = attributions
        
            attributions_sample = np.array(attributions_sample)

            # remove first column which is padding token
            attributions_sample = attributions_sample[1:, 1:]

            # flatten the attributions
            attributions_sample = attributions_sample.flatten()

        return attributions_sample
    return lig_output, model_CTRL, model_DD


@app.cell
def _(here, pl):
    # reproducibility
    seed_val = 2
    pl.seed_everything(seed_val) 

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

    # model name and output folder path
    model_loc = here('checkpoints', 'XLNet-PLabelDH_S2')

    condition_dict_values = {64: 'CTRL', 65: 'ILE', 66: 'LEU', 67: 'LEU_ILE', 68: 'LEU_ILE_VAL', 69: 'VAL'}
    return (
        annot_thresh,
        batch_size_val,
        condition_dict_values,
        d_model_val,
        dropout_val,
        longZerosThresh_val,
        loss_fun_name,
        lr_val,
        model_loc,
        n_heads_val,
        n_layers_val,
        percNansThresh_val,
        seed_val,
    )


@app.cell
def _(
    GWSDatasetFromPandas,
    XLNetConfig,
    XLNetForTokenClassification,
    d_model_val,
    dropout_val,
    here,
    n_heads_val,
    n_layers_val,
    os,
    pd,
    torch,
):
    class XLNetDH(XLNetForTokenClassification):
        def __init__(self, config):
            super().__init__(config)
            self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
    model = XLNetDH(config)

    # set the path to generate attributions
    test_path = here('data', 'orig', 'test.csv')
    test_dataset = pd.read_csv(test_path)
    num_samples = len(test_dataset)

    output_folder = 'attr/'

    # # check files in the output folder
    files = os.listdir(output_folder)

    # convert pandas dataframes into torch datasets
    test_dataset = GWSDatasetFromPandas(test_dataset)
    print("samples in test dataset: ", len(test_dataset))
    return (
        XLNetDH,
        config,
        files,
        model,
        num_samples,
        output_folder,
        test_dataset,
        test_path,
    )


@app.cell
def _(device, model, model_loc):
    model_1 = model.from_pretrained(model_loc + '/best_model')
    model_1.to(device)
    model_1.eval()
    return (model_1,)


@app.cell
def _(
    condition_dict_values,
    lig_output,
    model_1,
    np,
    output_folder,
    test_dataset,
    torch,
    tqdm,
):
    with torch.autograd.set_detect_anomaly(True):
        for (i, (x_input, y_true_full, y_true_ctrl, gene, transcript)) in tqdm(enumerate(test_dataset)):
            x = torch.tensor(x_input)
            y = torch.tensor(y_true_full)
            condition_token = condition_dict_values[int(x[0].item())]
            lig_sample_ctrl = lig_output(model_1, x, y, mode='ctrl')
            lig_sample_dd = lig_output(model_1, x, y, mode='dd')
            x_input_dev = torch.unsqueeze(x_input, 0).to('cuda')
            y_pred_full = model_1(x_input_dev).logits[0]
            y_pred_ctrl = torch.relu(y_pred_full[1:, 0]).cpu().detach().numpy()
            y_pred_depr_diff = y_pred_full[1:, 1].cpu().detach().numpy()
            y_pred_full_sample = y_pred_ctrl + y_pred_depr_diff
            y_true_dd_sample = y_true_full - y_true_ctrl
            out_dict = {'x_input': x_input, 'y_true_full': y_true_full, 'y_pred_full': y_pred_full_sample, 'y_true_ctrl': y_true_ctrl, 'gene': gene, 'transcript': transcript, 'lig_ctrl': lig_sample_ctrl, 'lig_dd': lig_sample_dd, 'y_pred_ctrl': y_pred_ctrl, 'y_pred_depr_diff': y_pred_depr_diff, 'y_true_dd': y_true_dd_sample, 'condition': condition_token}
            np.savez_compressed(output_folder + 'sample_' + str(i) + '.npz', out_dict)
    return (
        condition_token,
        gene,
        i,
        lig_sample_ctrl,
        lig_sample_dd,
        out_dict,
        transcript,
        x,
        x_input,
        x_input_dev,
        y,
        y_pred_ctrl,
        y_pred_depr_diff,
        y_pred_full,
        y_pred_full_sample,
        y_true_ctrl,
        y_true_dd_sample,
        y_true_full,
    )


if __name__ == "__main__":
    app.run()
