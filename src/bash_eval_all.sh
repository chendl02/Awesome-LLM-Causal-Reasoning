#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate mutual_infomation

for start in {0..9}
do
    python eval_all.py  --start $start --end $((start+1))&
done