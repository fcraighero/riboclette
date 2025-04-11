import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # libraries
    from utils import trainLSTM, RiboDatasetGWSDepr, GWSDatasetFromPandas # custom dataset and trainer
    from pytorch_lightning.loggers import WandbLogger
    import pytorch_lightning as pl
    import argparse
    return (
        GWSDatasetFromPandas,
        RiboDatasetGWSDepr,
        WandbLogger,
        argparse,
        pl,
        trainLSTM,
    )


@app.cell
def _(argparse):
    # Cell tags: parameters
    # argparse
    parser = argparse.ArgumentParser(description='Train LSTM model on GWS data')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')

    parsed_args = parser.parse_args()

    seed_val = parsed_args.seed

    tot_epochs = 100
    batch_size = 1
    dropout_val = 0.0
    annot_thresh = 0.3
    longZerosThresh_val = 20
    percNansThresh_val = 0.05
    lr = 1e-4
    num_layers = 4
    num_nodes = 64
    return (
        annot_thresh,
        batch_size,
        dropout_val,
        longZerosThresh_val,
        lr,
        num_layers,
        num_nodes,
        parsed_args,
        parser,
        percNansThresh_val,
        seed_val,
        tot_epochs,
    )


@app.cell
def _(pl, seed_val):
    print("Setting seed to {}".format(seed_val))
    pl.seed_everything(seed_val)
    return


@app.cell
def _(
    batch_size,
    dropout_val,
    lr,
    num_layers,
    num_nodes,
    seed_val,
    tot_epochs,
):
    model_name = 'LSTM DH: ' + '[NL: ' + str(num_layers) + ', NN: ' + str(num_nodes) + ']' + ' [BS: ' + str(batch_size) + ', D: ' + str(dropout_val) + ' E: ' + str(tot_epochs) + ' LR: ' + str(lr) + '] ' + 'Seed: ' + str(seed_val)

    # model parameters
    save_loc = 'saved_models/' + model_name
    return model_name, save_loc


@app.cell
def _(GWSDatasetFromPandas, RiboDatasetGWSDepr):
    # load datasets train and test
    # GWS dataset
    train_dataset, val_dataset, test_dataset = RiboDatasetGWSDepr()

    # convert to torch dataset
    train_dataset = GWSDatasetFromPandas(train_dataset)
    val_dataset = GWSDatasetFromPandas(val_dataset)
    test_dataset = GWSDatasetFromPandas(test_dataset)

    print("samples in train dataset: ", len(train_dataset))
    print("samples in val dataset: ", len(val_dataset))
    print("samples in test dataset: ", len(test_dataset))
    return test_dataset, train_dataset, val_dataset


@app.cell
def _(
    batch_size,
    dropout_val,
    lr,
    num_layers,
    num_nodes,
    save_loc,
    test_dataset,
    tot_epochs,
    trainLSTM,
    train_dataset,
    val_dataset,
):
    # train model
    model, result = trainLSTM(tot_epochs, batch_size, lr, save_loc, train_dataset, test_dataset, val_dataset, dropout_val, num_layers, num_nodes)
    return model, result


@app.cell
def _(result):
    print(result)
    return


if __name__ == "__main__":
    app.run()
