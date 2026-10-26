# Experiment: CUDA-Accelerated Healthcare Blockchain

This repository accompanies an experimental study of a proof-of-work

## What the experiment measures

`benchmark_repro.py` runs these CPU-side tests:

## Requirements

Use Python 3.12 or newer. Install the packages used by the experiment,
plotting:

```powershell
python -m pip install numpy matplotlib cryptography
```

`cryptography` is optional for running the benchmark, but without it AES-GCM


## Reproduce the experiment

From the project root, run the standard protocol:

```powershell
python benchmark_repro.py
```

To include the 1,200 MB AES payload:

```powershell
python benchmark_repro.py --full
```


```powershell
python benchmark_repro.py --reported
```

The benchmark writes files under `results/`:

- `measured_local.json` — local AES and SHA measurements and reported values.
- `environment.json` — machine and software fingerprint.
- `benchmark_results.csv` — flat measured and reported values for plotting or
  re-analysis.



```

## Project files

- `benchmark_repro.py` — benchmark protocol and result writer.
- `results/` — benchmark and reference artifacts.

