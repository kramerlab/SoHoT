#!/bin/bash


for DATA_SET in 'sleep' 'ann_thyroid' 'churn' 'nursery' 'twonorm' 'optdigits' 'texture' 'satimage' 'AGR_a' 'AGR_g' 'RBF_f' 'RBF_m' 'SEA50' 'SEA_5E5' 'HYP_f' 'HYP_m' 'epsilon' 'poker' 'covtype' 'kdd99'
do
	for SEED in 0 1 2 3 4
	do
		python -u main.py -d $DATASET -s $SEED
		echo "Finished on $DATASET with seed $SEED"
	done
done