#! /usr/bin/env bash
for P in 1 2 4 8 16
do
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 0 1 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 0 2 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 0 3 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 0 4 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 1 0 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 2 0 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 3 0 $P fast-forward &
    timeout 2h ../benchmarks/ff-ocean-contiguous.sh $1 4 0 $P fast-forward
done

