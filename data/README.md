# Data convention

This clean repository does not include real experiment CSV files. Code keeps
the same loader contract so local training and evaluation can use an external
dataset mounted or copied into the expected layout.

## Expected External Layout

```text
data/
  dataset/
    X1/
      status_data_*.csv
    X2/
    ...
    X27/
    unclassified/
  raw/
  split_manifest.csv
```

`data/dataset/` is the default training and evaluation root. Each grouped
directory contains GBK or UTF-8 status CSV files named `status_data_*.csv`.
`data/raw/` is for original collection folders and is not required by the clean
tests.

`split_manifest.csv` is optional. When present, it maps cleaned dataset files
back to source rounds so loaders can assign train, validation, and test splits.
Without a manifest, tests and examples can pass an explicit temporary data
configuration.

## Required Raw Fields

The shared CSV reader reads by column position because source headers have used
multiple encodings. HanWAM expects these normalized columns after loading:

```text
ts, T_out, T_out_coil, T_out_discharge, freq, eev, fan_out,
I_comp, T_in, T_in_coil, fan_in, RH_in, T_set, mode,
energy_cum, freq_in_tgt
```

Do not commit real CSVs or derived manifests into this repository. Runtime tests
generate temporary mock CSVs when schema coverage is needed.
