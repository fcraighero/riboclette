[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17100113.svg)](https://doi.org/10.5281/zenodo.17100113)


# 🧬🧑🏾‍💻 Riboclette: Conditional Deep Learning Model Reveals Translation Elongation Determinants during Amino Acid Deprivation

Welcome to **Riboclette**, a transformer-based deep learning model for predicting ribosome densities under various nutrient-deprivation conditions. Follow this tutorial to get started! 🚀

---

## Pip Package

Riboclette can be easily installed as a package using which you can make predictions on new gene sequences, and obtain model derived attributions to understand the predictions! 

🧀 **Package Documentation**: [Riboclette on PyPI](https://pypi.org/project/riboclette/)

```bash
pip install riboclette
```

---

## Web Server 🌐🧬

We provide a web-based server where you can explore codon-level attributions for different genes in the dataset. This server allows you to visualize and analyze the model's predictions and interpretability results interactively.

🔗 **Server Link**: [Ribotly](https://lts2.epfl.ch/ribotly/)

On the server, you can:
- Select genes of interest from the dataset.
- View codon-level attributions for each gene.
- Analyze how nutrient-deprivation conditions affect ribosome densities at a single codon resolution.

---

## Code Tutorial 📖✨

### 1️⃣ Build the Conda Environment 🐍🌳

Run the following commands to build and activate the conda environment:

```bash
conda env create --name envname
conda activate envname
pip install -r requirements.txt
```

### 2️⃣ Download Data and Checkpoints 📂🔗

Download the processed data and the pre-trained model checkpoints from the following link:

[Download Data and Checkpoints](https://doi.org/10.5281/zenodo.17100113)

Inside `utils` we provided a script to download all the data, saved by default in the working directory.

```bash
python utils/download_data.py --setting all
```

By changing `--setting` to `figures` or `model` one can download the data required only to reproduce the figures or test Riboclette.

Following the default setting:
- Data will be in the `data/` folder. 📁
- Checkpoints will be in the `checkpoints/` folder. ✅

Otherwise, one can change the download location with `--data_dir`, but to run the scripts the files should be organized with the structure defined above.

---

### 3️⃣ Prepare the Dataset 🐁📊

To run the data pre-processing pipeline, run the following command:

```bash
cd /riboclette/preprocessing
python processing.py
```

---

### 4️⃣ Train the Riboclette Model 🧠💻

Train the Riboclette model using the following command:

```bash
cd /riboclette/models/xlnet/dh
python train.py
```

---

### 5️⃣ Perform Pseudolabeling ➕

#### Train 5 Seed Models 🌱🌱🌱🌱🌱

To perform pseudolabeling, first train 5 seed models of Riboclette:

```bash
cd /riboclette/models/xlnet/dh
python train.py --seed {1, 2, 3, 4, 42}
```

#### Generate the Pseudolabeling Dataset 🧬📋

Once all seed models are trained, generate the pseudolabeling dataset:

```bash
cd /riboclette/preprocessing
python plabeling.py
```

#### Train Pseudolabeling-Based Models 🧠🔄

Train pseudolabeling-based model using the following command:

```bash
cd /riboclette/models/xlnet/plabel
python train.py 
```

---

### 6️⃣ Generate Interpretability Results 🔍🧬

Generate codon-level interpretations for all sequences for the testing set:

```bash
cd /riboclette/models/xlnet/plabel
python LIGInterpret.py
```

Generate motifs derived from random windows chosen from the full dataset:

```bash
cd /riboclette/models/xlnet/plabel
python beamSearch.py
```

---

### 7️⃣ Downstream Analysis and Figure Recreation 📈🖼️

Recreate the figures from the Riboclette paper using the downstream analysis scripts provided in the repository. These scripts allow you to analyze the model outputs and generate the figures mentioned in the paper.

#### Steps for Downstream Analysis:

1. Navigate to the downstream analysis folder:
   ```bash
   cd /riboclette/downstream_analysis
   ```

2. Run the analysis notebooks to generate the respective figures:
   ```bash
   python figure{2,3,4,5}.py
   ```

3. The generated figures will be saved in the `riboclette/data/results/figures/` folder. 🖼️

---

🎉 **You're all set!** Follow these steps to fully utilize Riboclette for ribosome density prediction, interpretability, and downstream analysis. 🚀
