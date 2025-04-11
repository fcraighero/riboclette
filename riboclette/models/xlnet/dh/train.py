import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    import torch
    from transformers import XLNetConfig, XLNetForTokenClassification, TrainingArguments, EarlyStoppingCallback
    from ipynb.fs.full.utils import RegressionTrainerFour, RiboDatasetGWS, GWSDatasetFromPandas, collate_fn, compute_metrics  # custom dataset and trainer
    import pytorch_lightning as pl
    import argparse
    return (
        EarlyStoppingCallback,
        GWSDatasetFromPandas,
        RegressionTrainerFour,
        RiboDatasetGWS,
        TrainingArguments,
        XLNetConfig,
        XLNetForTokenClassification,
        argparse,
        collate_fn,
        compute_metrics,
        pl,
        torch,
    )


@app.cell
def _(argparse):
    # Cell tags: parameters
    annot_thresh = 0.3
    longZerosThresh_val = 20
    percNansThresh_val = 0.05
    tot_epochs = 100
    d_model_val = 512
    n_layers_val = 3
    n_heads_val = 4
    dropout_val = 0.1
    lr_val = 1e-4
    batch_size_val = 1
    loss_fun_name = '4L'

    parser = argparse.ArgumentParser(description='Train LSTM model on GWS data')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')

    parsed_args = parser.parse_args()

    seed_val = parsed_args.seed
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
        parsed_args,
        parser,
        percNansThresh_val,
        seed_val,
        tot_epochs,
    )


@app.cell
def _(
    batch_size_val,
    d_model_val,
    dropout_val,
    loss_fun_name,
    lr_val,
    n_heads_val,
    n_layers_val,
    pl,
    seed_val,
):
    # reproducibility
    pl.seed_everything(seed_val)

    # model name and output folder path
    model_name = 'XLNet-DH ' + '[NL: ' + str(n_layers_val) + ', NH: ' + str(n_heads_val) + ', D: ' + str(d_model_val) + ', LR: ' + str(lr_val) + ', BS: ' + str(batch_size_val) + ', LF: ' + loss_fun_name + ', Dr: ' + str(dropout_val) + ', S: ' + str(seed_val) + ']'
    output_loc = "saved_models/" + model_name
    return model_name, output_loc


@app.cell
def _(GWSDatasetFromPandas, RiboDatasetGWS):
    # load dataset
    train_dataset, val_dataset, test_dataset = RiboDatasetGWS()

    # convert pandas dataframes into torch datasets
    train_dataset = GWSDatasetFromPandas(train_dataset)
    val_dataset = GWSDatasetFromPandas(val_dataset)
    test_dataset = GWSDatasetFromPandas(test_dataset)

    print("samples in train dataset: ", len(train_dataset))
    print("samples in val dataset: ", len(val_dataset))
    print("samples in test dataset: ", len(test_dataset))
    return test_dataset, train_dataset, val_dataset


@app.cell
def _(
    XLNetConfig,
    XLNetForTokenClassification,
    d_model_val,
    dropout_val,
    n_heads_val,
    n_layers_val,
    torch,
):
    # load xlnet to train from scratch
    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 6 conds, 64 codons, 1 for padding
    model = XLNetForTokenClassification(config)

    # modify the output layer
    model.classifier = torch.nn.Linear(d_model_val, 2, bias=True)
    return config, model


@app.cell
def _(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", pytorch_total_params)
    return (pytorch_total_params,)


@app.cell
def _(
    EarlyStoppingCallback,
    RegressionTrainerFour,
    TrainingArguments,
    batch_size_val,
    collate_fn,
    compute_metrics,
    lr_val,
    model,
    output_loc,
    tot_epochs,
    train_dataset,
    val_dataset,
):
    # xlnet training arguments
    training_args = TrainingArguments(
        output_dir = output_loc,
        learning_rate = lr_val,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = batch_size_val, # training batch size = per_device_train_batch_size * gradient_accumulation_steps
        per_device_eval_batch_size = 1,
        eval_accumulation_steps = 4, 
        num_train_epochs = tot_epochs,
        weight_decay = 0.01,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        push_to_hub = False,
        dataloader_pin_memory = True,
        save_total_limit = 5,
        dataloader_num_workers = 4,
        include_inputs_for_metrics = True
    )

    # initialize trainer
    trainer = RegressionTrainerFour(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )
    return trainer, training_args


@app.cell
def _(output_loc, trainer):
    # train model
    trainer.train()

    # save best model
    trainer.save_model(output_loc + "/best_model")
    return


@app.cell
def _(test_dataset, trainer):
    # evaluate model
    res = trainer.evaluate(eval_dataset=test_dataset)

    print(res)
    return (res,)


if __name__ == "__main__":
    app.run()
