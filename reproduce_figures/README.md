# Figure Reproducibility

## 1. Envirorment installation
The figures were generated with `python==3.12.4`. To install the envirorment, run:

```
pip install -r reproduce_figures/requirements.txt
```

## 2. Download Data
Download the data by running the script `utils/download_data.py` as described in the main `README` (Download Data and Checkpoints). The data should be located in `data` (default behaviour).

```
python utils/download_data.py --setting figures
```

## 3. Reproduce the figures

Scripts to generate the files into `/path/to/figures`.
```python
python reproduce_figures/panel_scatter.py -output_dirpath=/path/to/figures

python reproduce_figures/performance_figures.py -output_dirpath=/path/to/figures

# attributions_figures.py is the only one that requires additional files
# such files are generated in /path/to/plotting/data by make_data_for_attributions_figures.py
# note that /path/to/plotting/data needs to be passed also to attributions_figures.py
python reproduce_figures/make_data_for_attributions_figures.py -output_dirpath=/path/to/plotting/data
python reproduce_figures/attributions_figures.py -output_dirpath=/path/to/figures -plotting_dirpath=/path/to/plotting/data

python reproduce_figures/motifs_figures.py -output_dirpath=/path/to/figures

python reproduce_figures/selected_genes_figures.py -output_dirpath=/path/to/figures

python reproduce_figures/data_stats_figures.py -output_dirpath=/path/to/figures
```

## 4. Run scripts as notebooks (Optional)

Thanks to `marimo`, all the scripts can be executed as marimo notebooks with:

```
marimo edit reproduce_figures/script_name.py
```

or converted to Jupyter notebooks with:

```
marimo export ipynb reproduce_figures/script_name.py -o notebook_name.ipynb
```
