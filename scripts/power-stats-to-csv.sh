#! /usr/bin/env sh

echo 'cluster,cpu,power_type,power' > $2
grep -i cpus[0123].power_model.[dynamic:static] $1 | sed -e 's/\#.*//g;s/[ \t]\+$//g;s/system\.//g;s/power_model\.//g;s/cpus//g;s/ \+/,/g;s/\./,/g;' | sed -r 's/,([^,]*)$/.\1/' | sed '/[a-z]\+\.[0-9]\+$/ { s/\./,/g }' > $2

