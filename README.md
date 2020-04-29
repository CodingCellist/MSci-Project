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
### Creating a disk image
1. Create a new file of (in this case, 1024B*1024 = 1GiB) zeros using
   ```bash
   $ dd if=/dev/zero of=path/to/file.img bs=1024 count=1024
   ```
   (you may need to be root or use `sudo` for the next couple of steps)
2. Find the next available loopback device
   ```bash
   $ losetup -f
   ```
3. Set up the device returned (e.g. `/dev/loop0`) with the image file at offset
   32256 (63 * 512 bytes; something to do with tracks, see
   [this](https://www.gem5.org/documentation/general_docs/fullsystem/disks))
   ```bash
   $ losetup -o 32256 /dev/loop0 path/to/file.img
   ```
4. Format the device
   ```bash
   $ mke2fs /dev/loop0
   ```
5. Detach the loopback device
   ```bash
   $ losetup -d /dev/loop0
   ```

Done. The image can now be mounted and manipulated using
```bash
$ mount -o loop,offset=32256 path/to/file.img path/to/mountpoint
```

**IMPORTANT: remember to copy the GNU/*NIX binaries necessary for the system
you'll be emulating to their appropriate locations on the new disk**

Some details about what to do next can be found here:
- [gem5-specific files](https://www.gem5.org/documentation/general_docs/fullsystem/disks#setting-up-gem5-specific-files)
- [m5 utility](https://www.gem5.org/documentation/general_docs/m5ops/)

## DVFS details
The gem5 devs/website openly admits that the DVFS documentation
[is outdated](https://www.gem5.org/about/#power-and-energy-modeling), leading
the user to having to manually read their way through source code and example
config scripts to try to figure out how to construct the relevant components.
This my attempt at documenting and understanding how it works.
### Voltage Domains
Voltage Domains dictate the voltage values the system can use. It seems gem5
always simulates voltage in FS mode, but simply sets it to 1.0V if the user does
not care about voltage simulation (see `src/sim/VoltageDomain.py`)

To create a voltage domain, either a voltage value or a list of voltage values
must be given. But not just to the `VoltageDomain` constructor, no that would
be too simple, but instead as a keyword-argument (kwarg), i.e. `voltage`. To my
knowledge, this is not documented anywhere, nor is it easily discoverable from
the `src/sim/{VoltageDomain.py, voltage_domain.hh, voltage_domain.cc}` files.

The example voltage domains I've used are (note that the values have to be 
specified in descending order):

For the big cluster:
```python
odroid_n2_voltages = [  '0.981000V'
                      , '0.891000V'
                      , '0.861000V'
                      , '0.821000V'
                      , '0.791000V'
                      , '0.771000V'
                      , '0.771000V'
                      , '0.751000V'
                     ]
odroid_n2_voltage_domain = VoltageDomain(voltage=odroid_n2_voltages)
```

For the LITTLE cluster:
```python
odroid_n2_voltages = [  '0.981000V'
                      , '0.861000V'
                      , '0.831000V'
                      , '0.791000V'
                      , '0.761000V'
                      , '0.731000V'
                      , '0.731000V'
                      , '0.731000V'
                     ]
odroid_n2_voltage_domain = VoltageDomain(voltage=odroid_n2_voltages)
```

These numbers were obtained by examining the changes in the sysfs files
`/sys/class/regulator/regulator.{1,2}/microvolts` when using the `userspace`
frequency governor and varying the frequency of the big and LITTLE clusters
(respectively) using the `cpupower` command-line tool.

**NOTE:** In gem5 (and, as far as I know, on real hardware) voltage domains
apply to CPU sockets. So make sure that the big and LITTLE clusters in the
simulator are on different sockets if they need to have different voltage
domains (you can inspect the socket through the `socket_id` value associated
with the clusters)
### Clock Domains
Clock domains dictate what frequencies the CPU(s) can be clocked at (what steps
are available for the DVFS handler) and are associated with a Voltage Domain. I
am uncertain as to what precisely the requirements are for the relationship
between these two, especially as the constructor does not seem to complain if
there is a different number of values in the available clocks and voltages.

I obtained the following clock values from the Odroid N2 board using the
`cpupower` command-line tool:

For the big cluster:
```python
odroid_n2_clocks = [  '1800MHz'
                    , '1700MHz'
                    , '1610MHz'
                    , '1510MHz'
                    , '1400MHz'
                    , '1200MHz'
                    , '1000MHz'
                    ,  '667MHz'
                   ]
odroid_n2_clk_domain = SrcClockDomain(clock=odroid_n2_clocks,
                                      voltage_domain=odroid_n2_voltage_domain
                                      )
```

For the LITTLE cluster:
```python
odroid_n2_clocks = [  '1900MHz'
                    , '1700MHz'
                    , '1610MHz'
                    , '1510MHz'
                    , '1400MHz'
                    , '1200MHz'
                    , '1000MHz'
                    ,  '667MHz'
                   ]
odroid_n2_clk_domain = SrcClockDomain(clock=odroid_n2_clocks,
                                      voltage_domain=odroid_n2_voltage_domain
                                      )
```
### Adding DVFS to an existing CPU
The statements below, whilst possibly correct, seem to go against the way things
are done in the example scripts. As such, here is a "better" way of doing
things: It turns out that the `--big-cpu-clock` value(s), when passed on to a
`CpuCluster` sub-class, creates a new `SrcClockDomain` according to that value.
Therefore, there are 2 solutions (of which I have only tested the first):
1. Create sub-classes of the `CpuCluster`. Similar to the existing
   `BigCluster` and `LittleCluster` sub-classes, these will extend `CpuCluster`.
   However, in addition to the config that these classes specify in their body,
   also define the two lists of values for the voltage and clock domains
   respectively. Then, simply pass these lists as the appropriate arguments to
   the `super` call at the end of the sub-class's `__init__` declaration (3rd
   and 4th argument at the time of writing, but double-check with your
   `<gem5-root>/configs/example/arm/devices.py` file). If you want to add DVFS
   to the `AtomicCluster` as well, simply extend this class in a similar manner.
   FINALLY, make sure to add an entry to the `cpu_types` dictionary near the end
   of the file. The entry should have a name for the `--cpu-type` flag to refer
   to your classes by, and a 2-tuple (a pair) of clusters for it to instantiate
   (i.e. put your new DVFS-capable classes here). Your specified DVFS values
   will now be run when using those clusters.

2. As mentioned previously, the value(s) passed to the `--big-cpu-clock` flag is
   used to create a new `SrcClockDomain` internally. Hence, another (possibly
   more flexible) solution is to add a `--big-cpu-voltage` flag, wire up its
   values in the configuration script (e.g.
   `<gem5-root>/configs/example/arm/fs_bigLITTLE.py`), and pass a list of values
   for each of the four flags (both voltage and clock for both big and LITTLE
   cpus).

~~AHAHAHAHAHAHAHAHAHAHAHAHAHAHA! Nothing is simple intuitive in this thing,~~
~~is it??   \*Hrm, hrm\* Anyway...~~

~~As can be seen in the example `configs/example/arm/fs_bigLITTLE.py` script,~~
~~the CPUs are located in a seemingly magic `ObjectList.cpu_list`. This list~~
~~can be printed to stdout by calling `ObjectList.cpu_list.print()` anywhere~~
~~in the scripts using it. Unfortunately, the attributes of the SimObjects~~
~~cannot be modified once instantiated, so in order to add custom DVFS to any~~
~~existing CPU, we have to extend it. The example FS script imports some CPU~~
~~things from `common.cores.arm`, so you'd be forgiven for thinking you can~~
~~extend a CPU by importing the relevant file and extending the CPU class~~
~~found in it.~~

~~Well yes, but actually no.~~

~~It turns out the magical `ObjectList` (whose source code can be found in~~
~~`configs/common/ObjectList.py`) can only see CPUs if they're in the~~
~~`configs/common/cores/arm` directory. So, to extend an existing CPU, create~~
~~a new file in that directory, e.g.~~
~~`configs/common/cores/arm/O3_ARM_v7a_3_DVFS.py`. If you're good/want to put~~
~~the source code up anywhere ever, you should include the disclaimer and~~
~~copyright notice from the orginal file that you're extending. In your new~~
~~file, do `from __futures__ import absolute_import` and~~
~~`from m5.objects import *` (don't know if that one is actually required, but~~
~~better safe than sorry). Then, import the file you want to extend~~
~~`from common.cores.arm import <filename>` (yes, you have to include the~~
~~`common.cores.arm` even though you're in the directory, otherwise it can't~~
~~find the file). Now go ahead and create your python sub-class like you~~
~~normally would.~~

~~**Note:** If you want to use custom variable names, the variables cannot be~~
~~defined in the class. You'll need to just define them in the script and then~~
~~use them in your class by assigning them to the "appropriate" class~~
~~attribute names.~~

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

