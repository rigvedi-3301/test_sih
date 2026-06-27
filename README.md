# Transformer based Firewall Pipeline

Protoype of a URL threat classifier securing Rank 6 in SIH PS 25172 2025. Uses a one-class anomaly detection approach, trained only on benign URLs, so malicious ones stand out as high reconstruction error at inference.

## How it works

Two BERT-based encoders (CySecBERT + ELECTRA) produce a fused 1536-dim embedding per URL. An autoencoder learns to reconstruct benign patterns only. At inference, URLs with MSE above a dynamic threshold are flagged malicious.

## Files

- `train.py` - trains the fusion encoder + autoencoder on benign-only data, logs metrics to W&B
- `inference.py` - loads saved weights and classifies a list of URLs
- `data_prep.py` - merges and deduplicates 4 source CSVs into `dataset_1.csv`
- `dataset_1.csv` - combined dataset with `url` and `result` columns (0 = benign, 1 = malicious)

## Usage

```bash
pip install torch transformers wandb pandas numpy
python train.py      # train model
python inference.py  # run inference on sample URLs
```

Edit `test_urls` in `inference.py` to test your own URLs. Model weights and tokenizers are saved to `cysec_electra_oneclass_model_v4/` after training.

## Notes

- Requires a GPU for training
- Training uses up to 300k benign samples; validation includes malicious URLs to track separation
- This is an experimental/test setup - not production ready
