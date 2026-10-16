```bash
# Train split (seq tars only)
python -m prep_data.grab --input_dir datasets/grab --output_dir datasets/grab/data --mode train

# Test split (seq tars only)
python -m prep_data.grab --input_dir datasets/grab --output_dir datasets/grab/data --mode test
```

Output is written to `<output_dir>/<mode>_seq_tars/` (e.g. `datasets/grab/data/train_seq_tars/`).
