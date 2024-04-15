# HybridFragmentTokenization

GitHub repository for a submitted paper "ADMET Prediction from a Fragment Spectrum"

Code and data will be uploaded soon, with instructions on how to operate.

## Installation

extract allmolgen_pretrain_data_100maxlen_FIXEDCOLS.tar.xz for pre-training data

Then, to setup conda environment:

Use environment.yml

OR

conda create -n MTL-BERT numpy==1.21.5 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 rdkit -c rdkit -c pytorch -c nvidia
pip install tqdm scikit-learn