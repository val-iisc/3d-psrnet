python train.py \
	--data_dir data/ \
	--exp expts/1_multicat \
	--gpu 0 \
	--category all \
	--loss_recon chamfer \
	--loss_seg chamfer