#! /usr/bin/env bash
for P in 1 2 4 8 16
do
    echo "###############"
    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/4b2L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 4 2 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/1b1L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 1 1 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/2b2L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 2 2 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/3b3L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 3 3 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/4b4L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 4 4 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/2b4L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 2 4 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/1b3L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 1 3 $P roi-out ${CPT} &

    CPT=$(find $1/fast-forward/ocean/contiguous_partitions/3b1L/m5out-p$P-ondemand -name cpt.* | tail -n 1)
    ../benchmarks/ocean-contiguous.sh $1 3 1 $P roi-out ${CPT} &

    wait
done

