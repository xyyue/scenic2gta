#!/bin/bash

for value in $(seq 1 1 $2) 
do
    echo $value
    python.exe get_one_img.py -i $value -sc $1
    sleep 5
done

