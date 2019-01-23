#!/bin/bash

gpu=0
exp=trained_model
eval_set=test
data_dir=data
snapshot=best
n_cls=4
declare -a categs=("chair" "car" "aero")
for cat in "${categs[@]}"; do
	python metrics.py \
		--data_dir ${data_dir} \
		--exp $exp \
		--gpu $gpu \
		--category $cat \
		--eval_set ${eval_set} \
		--snapshot ${snapshot} \
		--n_cls ${n_cls} \
		--tqdm
done

clear
printf "==== ${eval_set} metrics ===\n\n"
declare -a categs=("chair" "car" "aero")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}_${snapshot}_avg.csv
	echo
done