#!/bin/bash

rm -rfv ./SUITE/CONSTANT/NORMAL/*.csv
rm -rfv ./SUITE/GLOBAL/NORMAL/*.csv
rm -rfv ./SUITE/CONSTANT/WARP_SIZE/*.csv
rm -rfv ./SUITE/GLOBAL/WARP_SIZE/*.csv

for file in ./SUITE/CONSTANT/NORMAL/*.txt
do
	echo "Parsing file $file..."
	python3 parse_output.py $file 5 10 50 90 95
done
for file in ./SUITE/GLOBAL/NORMAL/*.txt
do
	echo "Parsing file $file..."
	python3 parse_output.py $file 5 10 50 90 95
done

for file in ./SUITE/CONSTANT/WARP_SIZE/*.txt
do
	echo "Parsing file $file..."
	python3 parse_warp_output.py $file 5 10 50 90 95
done
for file in ./SUITE/GLOBAL/WARP_SIZE/*.txt
do
	echo "Parsing file $file..."
	python3 parse_warp_output.py $file 5 10 50 90 95
done