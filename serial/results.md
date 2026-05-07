# Part 1: Serial Cholesky Factorization — Results

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_serial_<jobid>.out`.

## Results (`cholesky_serial_108259.out`)

| N    | Best time (s) | Avg time (s) | GFLOP/s | Residual  |
|------|--------------|-------------|---------|-----------|
| 256  | 0.003062     | 0.003079    | 1.827   | 5.10e-12  |
| 512  | 0.058234     | 0.058305    | 0.768   | 2.65e-11  |
| 1024 | 0.776923     | 0.781225    | 0.461   | 1.48e-10  |
| 2048 | 12.502958    | 12.802376   | 0.229   | 8.40e-10  |
