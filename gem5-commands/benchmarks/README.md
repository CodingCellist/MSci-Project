# Assumed files layout
- a directory containing the gem5 simulator
- relative to the gem5 directory:
  - `./linux-systems/custom/{binaries,disks}`
  - `../gem5-bootscripts/benchmarks/<all_benchmarks>/<all_setups>.rcS`
  - `../gem5-commands/benchmarks/<all_benchmarks>.sh` (the directory with this
    README)


# Usage
## ff-scripts
Fast-forwards (using the `dvfs-atomic` CPUs) the given benchmark.
```bash
$ ff-<script.sh> GEM5_ROOT N_BIG N_LITTLE N_THREADS GEM5_OUTDIR [KERN_GOV]
```
- `GEM5_ROOT` is the _absolute_ path to the gem5 directory
- `N_BIG` and `N_LITTLE` is the number of big and LITTLE CPUs respectively
- `N_THREADS` is the number of threads to use in the benchmark: 1, 2, 4, 8, or 16
- `GEM5_OUTDIR` is the directory in which _all_ benchmarks' output lives
- `KERN_GOV` is optional. If specified, it is one of "ondemand" or "schedutil".
  If not specified, the default is "ondemand"

## benchmark scripts
Runs the given benchmark (using the `dvfs-timing` CPUs), resuming from the given
benchmark.
```bash
$ <script.sh> GEM5_ROOT N_BIG N_LITTLE N_THREADS GEM5_OUTDIR CPT [KERN_GOV]
```
- `GEM5_ROOT` is the _absolute_ path to the gem5 directory
- `N_BIG` and `N_LITTLE` is the number of big and LITTLE CPUs respectively
- `N_THREADS` is the number of threads to use in the benchmark: 1, 2, 4, 8, or 16
- `GEM5_OUTDIR` is the directory in which _all_ benchmarks' output lives
- `CPT` is the checkpoint (directory) to restore from
- `KERN_GOV` is optional. If specified, it is one of "ondemand" or "schedutil".
  If not specified, the default is "ondemand"

