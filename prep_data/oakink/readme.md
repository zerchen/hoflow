```bash
python -m prep_data.oakink --input_dir datasets/oakink/data --output_dir datasets/oakink/data/seq_tars
```

Writes metadata for motion files and mesh/action pairs. The unused `--split`
and `--group_seq` options have been removed; select the input directory directly.
