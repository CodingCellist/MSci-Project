# CS5199 - Individual Masters Project
**Student:** Thomas E. Hansen (teh6, 150015673)

**Supervisor:** Dr. John Thomson


# Tools Used
- [gem5](http://www.gem5.org/)
- [ARMv8 cross compiler](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads)
- [gemstone-applypower](https://github.com/mattw200/gemstone-applypower)


# gem5 details
## Patches
- [power: Fix regStats for PowerModel and PowerModelState](https://gem5-review.googlesource.com/c/public/gem5/+/26643)
- [sim-power: Fix power model to work with stat groups](https://gem5-review.googlesource.com/c/public/gem5/+/26785)

Applied using `git-cherry-pick`

## Linux images used
- [linux-aarch64-20170616](http://dist.gem5.org/dist/current/arm/aarch-system-20170616.tar.xz)
- [linux-aarch64-201901106](http://dist.gem5.org/dist/current/arm/aarch-system-201901106.tar.bz2)

A working index can be found on the old
[m5sim](http://m5sim.org/dist/current/arm/) page. These files should then be
retrieved from
[dist.gem5.org/dist/current/arm/](http://dist.gem5.org/dist/current/arm/)

## Running
### Setup
```bash
$ cd gem5
$ export M5_PATH=path/to/linux/files
```
### Commands
Both the commands below can further be customised by the flags:
- `--big-cpus N`
- `--little-cpus N`
- `--big-cpu-clock HZ`
- `--little-cpu-clock HZ`

Full system simulation without power:
```bash
$ ./build/ARM/gem5.opt configs/example/arm/fs_bigLITTLE.py \
    --dtb=$M5_PATH/binaries/<dtb-name>.dtb \
    --kernel=$M5_PATH/binaries/<kernel-name> \
    --machine-type=VExpress_GEM5_V1 \
    --disk=$M5_PATH/disks/<disk-image-name>.img \
    --caches \
    --bootscript=path/to/bootscript.rcS
```
Full system simulation with power:
```bash
$ ./build/ARM/gem5.opt configs/example/arm/fs_bigLITTLE.py \
    --dtb=$M5_PATH/binaries/<dtb-name>.dtb \
    --kernel=$M5_PATH/binaries/<kernel-name> \
    --machine-type=VExpress_GEM5_V1 \
    --disk=$M5_PATH/disks/<disk-image-name>.img \
    --caches \
    --bootscript=path/to/bootscript.rcS
```


# gemstone details
## Tutorials
- [gemstone-applypower](http://gemstone.ecs.soton.ac.uk/gemstone-website/gemstone/tutorial-gemstone-apply-power.html)

## Commands used
(Assumes having `cd`-ed into `gemstone-applypower` and activated the venv)

For simulating Cortex A15
```
$ ./gemstone_create_equation.py -p models/gs-A15.params -m maps/gem5-A15.map -o gem5-A15
```

For simulating Cortex A7
```
$ ./gemstone_create_equation.py -p models/gs-A7.params -m maps/gem5-A7.map -o gem5-A7
```


# Misc.
- [source for getty used in gem5](https://git.busybox.net/busybox/tree/loginutils/getty.c?h=1_21_stable&id=41f7885f06612edcd525782f7ce3e75bd9a0d787)

