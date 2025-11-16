# LFE-SCCM

This repository provides the core model architecture for LFE-SCCM, a learning-based screen content image compression method.

## Key Points
- Only the model structure (`LFE_SCCM`) is provided. No pre-trained weights (.pth files) or internal datasets are included due to commercial restrictions.
- To reproduce the results in the paper, integrate `LFE_SCCM` into the [CompressAI](https://github.com/InterDigitalInc/CompressAI) framework and train on your own dataset.
- The original paper used a subset of the LSCD dataset for training and an internal SCCM dataset for evaluation.

## Usage
1. Place `lfe_sccm.py` into the CompressAI models folder.
2. Use CompressAI training scripts to train on a suitable dataset.
3. Perform evaluation using CompressAI evaluation tools.
