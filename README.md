# HybridFragmentTokenization

GitHub repository for a submitted paper "ADMET Prediction from a Fragment Spectrum". Reference to be provided after it's published.

## Installation

- download finetuned weights and logs from https://brocku-my.sharepoint.com/:f:/g/personal/na16dg_brocku_ca/Ei6Dk849_FZInrAEXLiYxDUByV0QbG9eeVG0g1CYihDcNQ?e=RxYh4P
- place folders inside HybridFragmentTokenization/
- extract allmolgen_pretrain_data_100maxlen_FIXEDCOLS.tar.xz for pre-training data

Then, to setup conda environment:

- conda env create -f environment.yml

OR

- conda create -n MTL-BERT numpy==1.21.5 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 rdkit -c rdkit -c pytorch -c nvidia
- pip install tqdm scikit-learn

## Inference

Finetuned weights were provided in the installation steps. To perform inference, view the example provided at the base of inference.py

