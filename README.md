# Uncertainty in Deep Learning for EEG under Dataset Shifts

This repo contains code for Article III of my PhD, focusing on uncertainty in deep learning models for EEG. We test different ensemble models under different conditions, out-of-distribution shift

## Overview

The reliable application of AI in clinical settings like neurophysiology is hindered by concerns about model trustworthiness, especially when models encounter data that differs from their training distribution. This project addresses this challenge by exploring how various ensemble strategies and Bayesian approaches quantify predictive uncertainty in an EEG-based cognitive status classification task.

The core contributions of this work are:
- Implementation of multiple ensemble methods methods, including Deep Ensembles, Snapshot Ensembles, and Monte Carlo Dropout (MCD).
- A framework for systematically evaluating model robustness through a series of ten controlled, synthetic dataset shifts.
- A rigorous comparison of these methods on both in-distribution and out-of-distribution (OOD) public datasets.

## Data

This study utilizes several publicly available EEG datasets:

-   **Primary (In-Distribution) Dataset:** The [CAUEEG dataset](https://github.com/ipis-mjkim/caueeg-dataset) was used for training and in-distribution evaluation. You can find information on how to request access at the provided link.
-   **Out-of-Distribution (OOD) Datasets:**
    1.  The Miltiadous (Greek EEG) dataset.
    2.  The MPI-Leipzig Mind-Brain-Body (MPI) dataset.
    3.  The TDBRAIN dataset.

Please refer to the original article for detailed descriptions and specific preprocessing steps for each dataset.
    
## Publish status

Currently only available as a pre-print:

@article {Tveter2025.07.09.663220,
	author = {Tveter, Mats and Tveitstoel, Thomas and Hatlestad-Hall, Christoffer and Hammer, Hugo  L. and Haraldsen, Ira R. J. Hebold},
	title = {Uncertainty in Deep Learning for EEG under Dataset Shifts},
	elocation-id = {2025.07.09.663220},
	year = {2025},
	doi = {10.1101/2025.07.09.663220},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/07/11/2025.07.09.663220},
	eprint = {https://www.biorxiv.org/content/early/2025/07/11/2025.07.09.663220.full.pdf},
	journal = {bioRxiv}
}


![Tests](https://github.com/matstveter/dl_uncertainty/actions/workflows/tests.yml/badge.svg)\
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)\
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
