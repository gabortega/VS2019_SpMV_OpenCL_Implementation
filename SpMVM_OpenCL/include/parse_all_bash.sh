#!/bin/bash

rm -rfv ./SUITE/CONSTANT/*.csv
rm -rfv ./SUITE/GLOBAL/*.csv
rm -rfv ./SUITE/SEQ/*.csv
rm -rfv ./OCCUPANCY/*.csv

for file in ./SUITE/CONSTANT/*.txt
do
	echo "Parsing file $file..."
	python3 parse_output.py $file 5 10 50 90 95
done
for file in ./SUITE/GLOBAL/*.txt
do
	echo "Parsing file $file..."
	python3 parse_output.py $file 5 10 50 90 95
done
for file in ./SUITE/SEQ/*.txt
do
    echo "Parsing file $file..."
    python3 parse_output.py $file 5 10 50 90 95
done

for file in ./OCCUPANCY/*.txt
do
    echo "Parsing file $file..."
    python3 parse_occupancy_output.py $file 5 10 50 90 95
done