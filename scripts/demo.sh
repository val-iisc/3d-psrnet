#!/bin/bash

gpu=0
exp=trained_model
data_dir=data
eval_set=test
cat=chair
snapshot=best
n_cls=4

python metrics.py \
	--data_dir ${data_dir} \
	--exp $exp \
	--gpu $gpu \
	--category $cat \
	--eval_set ${eval_set} \
	--snapshot ${snapshot} \
	--n_cls ${n_cls} \
	--visualize

