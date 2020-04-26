#! /usr/bin/env bash
for P in 1 2 4 8 16
do
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 4 2 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 1 1 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 2 2 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 3 3 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 4 4 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 2 4 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 1 3 $P fast-forward &
    timeout 2h ../benchmarks/ff-non-ocean-contiguous.sh $1 3 1 $P fast-forward
done

