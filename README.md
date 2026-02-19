# MDML_PeptPerm
**MDML_PeptPerm** is a tool for predicting the membrane permeability coefficient of cyclic peptides by combining descriptors obtained from molecular dynamics simulations with 2D descriptors.

This repository provides the implementation used in the associated publication.



```text
MDML_PeptPerm/
├── README.md
├── requirements.txt
├── environment.yml
├── environment_cuda.yml
├── main.ipynb                # Model training and cross-validation
├── ex_test.ipynb             # External test evaluation
├── data/                     # Training and external datasets
├── predicted/                # Model prediction outputs
├── weight/                   # Saved model weights
└── utils/                    # Utility scripts and helper functions
```

---

## Installation
### 1. Clone the repository
    ```sh
    git clone https://github.com/akiyamalab/MDML_PeptPerm.git
    cd MDML_PeptPerm
    ```

### 2. Create and activate a conda/mamba environment (recommended)

#### Option A — Using environment.yml

```bash
mamba env create -f environment.yml
mamba activate MDMLpept
```

#### Option B — Manual installation

```bash
mamba create -n MDMLpept python=3.9.6 -y
mamba activate MDMLpept
mamba install -c conda-forge $(cat requirements.txt) -y
```

---


## Usage

After activating the environment:

```bash
jupyter notebook
```

- Run `main.ipynb` for model training and cross-validation.
- Run `ex_test.ipynb` for external evaluation.

---

## Citation

If you use this code in your research, please cite:

```
Author(s). Title. Journal. Year. DOI
```

---

## License

See the `LICENSE` file for details.




