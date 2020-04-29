#! /usr/bin/env bash

GEM5_ROOT=$1
N_BIG=$2
N_LITTLE=$3
N_THREADS=$4
GEM5_OUTDIR=$5
KERN_GOV=$6

if [[ -z ${KERN_GOV} ]]
then
    KERN_GOV=ondemand
fi

export M5_PATH=${GEM5_ROOT}/linux-systems/custom
cd ${GEM5_ROOT}
./build/ARM/gem5.opt \
    --outdir=${GEM5_OUTDIR}/barnes/${N_BIG}b${N_LITTLE}L/m5out-p${N_THREADS}-${KERN_GOV} \
    --dot-dvfs-config=dvfs-config.dot \
    --redirect-stdout \
    --redirect-stderr \
    --stats-file=/dev/null \
    configs/example/arm/fs_bL_extended.py \
    --caches \
    --pmus \
    --big-cpus=${N_BIG} \
    --little-cpus=${N_LITTLE} \
    --disk=${M5_PATH}/disks/aarch-system-2017-lin-min-aarch64-Splash3.img \
    --kernel=${M5_PATH}/binaries/vmlinux-5.4.24-armv8-${KERN_GOV} \
    --bootloader=${M5_PATH}/binaries/boot.arm64 \
    --bootscript=../gem5-bootscripts/benchmarks/barnes/n16384_p${N_THREADS}.rcS \
    --cpu-type=dvfs-atomic

