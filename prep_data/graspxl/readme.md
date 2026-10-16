```bash
python -m prep_data.graspxl --input_dir datasets/graspxl/data --output_dir datasets/graspxl/data/full_seq_tars --group_seq
```

Use `--group_seq` to store each sequence as one sample; omit it for per-frame
samples. The unused `--split` option has been removed; select the input directory
directly.
