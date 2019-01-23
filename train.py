'''
Code for training the 3D-PSRNet model.

Usage:
	python train.py --data_dir data/ --exp expts/1_multicat --gpu 0 --category all --loss_recon chamfer --loss_seg chamfer
'''

import os
import sys
import json
import argparse
import cv2
import numpy as np
import random
import re
import scipy
import tensorflow as tf
import tflearn
import time
from itertools import product
from scipy import misc
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
from itertools import product

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('utils')

from utils.tf_ops.cd import tf_nndistance
from utils.tf_ops.emd.tf_auctionmatch import auction_match
from utils.tf_ops.emd_approx.tf_approxmatch import approx_match, match_cost
from utils.net import joint_seg_net 
from utils.train_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, required=True, 
		help='Path to dataset')
parser.add_argument('--exp', type=str, required=True, 
		help='Name of Experiment Prefixed with index')
parser.add_argument('--gpu', type=str, required=True, 
		help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
		help='Category to train on : ["all", "airplane", "car", "chair"]')
parser.add_argument('--batch_size', type=int, default=32, 
		help='Batch Size during training')
parser.add_argument('--loss_recon', type=str, required=True, 
		help='Loss to optimize on. Choose from [chamfer/emd/both]')
parser.add_argument('--loss_seg', type=str, required=True, 
		help='Loss to optimize on. Choose from [chamfer/emd]')
parser.add_argument('--n_cls', type=int, default=4, 
		help='number of parts or segmentation classes')
parser.add_argument('--LAMBDA_SEG', type=float, default=1., 
		help='weight for segmentation loss')
parser.add_argument('--LAMBDA_RECON', type=float, default=1., 
		help='weight for reconstruction loss')
parser.add_argument('--schedule_recon', action='store_true', 
		help='follow scheduling for reconstruction loss')
parser.add_argument('--schedule_seg', action='store_true', 
		help='follow scheduling for segmentation loss')
parser.add_argument('--lr', type=float, default=0.0005, 
		help='Learning Rate') ###
parser.add_argument('--max_epoch', type=int, default=500, 
		help='Maximum number of epochs to train for') ###
parser.add_argument('--print_n', type=int, default=50, 
		help='print output to terminal every n iterations')
parser.add_argument('--save_n', type=int, default=50, 
		help='save model parameters every n iterations')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

random.seed(1024)
tf.set_random_seed(1024)
np.random.seed(1024)

data_dir = FLAGS.data_dir
VAL_BATCH_SIZE = 32
BATCH_SIZE = FLAGS.batch_size
HEIGHT = 64
WIDTH = 64
NUM_VIEWS = 10
NUM_POINTS = 1024
NUM_CLASSES = FLAGS.n_cls
EXP_DIR = os.path.abspath(FLAGS.exp)


def get_epoch_loss(models, indices, val_batch_size, _lambda_recon, _lambda_seg):

	batches = len(indices)/val_batch_size
	val_chamfer = 0.
	val_forward = 0.
	val_backward = 0.
	val_emd = 0.
	val_seg = 0.
	val_loss = 0.

	for b in xrange(batches):
		batch_ip, batch_gt, batch_lbl, batch_lbl_wts = fetch_batch_joint(models, indices, b, val_batch_size)
		feed_dict = {img_inp:batch_ip, 
					pcl_gt:batch_gt, 
					labels_gt: batch_lbl, 
					lambda_recon: _lambda_recon,
					lambda_seg: _lambda_seg, 
					labels_gt_wts: batch_lbl_wts}
		L,S,C,F,B,E, _summ = sess.run([loss, seg_loss, chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled, emd_scaled, summ], feed_dict=feed_dict)
		val_chamfer += C/batches
		val_forward += F/batches
		val_backward += B/batches
		val_emd += E/batches
		val_seg += S/batches
		val_loss += L/batches

	return val_loss, val_seg, val_chamfer, val_forward, val_backward, val_emd, _summ


if __name__=='__main__':

	print 'start'
	# Create a folder for experiment and copy the training file
	create_folder(join(BASE_DIR, EXP_DIR))
	fname = basename(__file__)
	os.system('cp %s %s'%(fname, join(BASE_DIR, EXP_DIR)))
	with open(join(BASE_DIR, EXP_DIR, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	train_models, train_pair_indices = get_drc_models(data_dir, FLAGS.category, NUM_VIEWS, 'train')
	val_models, val_pair_indices = get_drc_models(data_dir, FLAGS.category, NUM_VIEWS, 'val')
		
	batches = len(train_pair_indices) / BATCH_SIZE

	### Create placeholders
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3), name='pcl_gt')
	labels_gt = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINTS), name='labels_gt')
	labels_gt_wts = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS), name='label_wts')
	lambda_recon = tf.placeholder(tf.float32, shape=(), name='lambda_recon')
	lambda_seg = tf.placeholder(tf.float32, shape=(), name='lambda_seg')

	# Build graph
	with tf.variable_scope('joint_seg_net'):
		out_pcl, out_pcl_seg = joint_seg_net(img_inp, NUM_POINTS, NUM_CLASSES)

	# Scale output and gt for val losses
	out_scaled, pcl_gt_scaled = scale(pcl_gt, out_pcl)

	# Calculate Chamfer Metrics
	dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pcl_gt, out_pcl)
	dists_forward =tf.reduce_mean(dists_forward)
	dists_backward =tf.reduce_mean(dists_backward)
	chamfer_distance = dists_backward + dists_forward
	loss_nodecay=(dists_forward+dists_backward/2.0)*10000

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled,_, dists_backward_scaled,_ = tf_nndistance.nn_distance(pcl_gt_scaled, out_scaled)
	dists_forward_scaled=tf.reduce_mean(dists_forward_scaled)
	dists_backward_scaled=tf.reduce_mean(dists_backward_scaled)
	chamfer_distance_scaled = dists_backward_scaled + dists_forward_scaled

	# Calculate EMD
	match = approx_match(out_pcl, pcl_gt)
	emd = tf.reduce_mean(match_cost(out_pcl, pcl_gt, match))

	# Calculate EMD scaled
	match_scaled = approx_match(out_scaled, pcl_gt_scaled)
	emd_scaled = tf.reduce_mean(match_cost(out_scaled, pcl_gt_scaled, match_scaled))

	if FLAGS.loss_seg == 'chamfer':
		pts_match_fwd, pts_match_bwd = get_labels_seg(pcl_gt, out_pcl, FLAGS.loss_seg)
		idx, _ = tf.meshgrid(range(BATCH_SIZE), range(NUM_POINTS), indexing='ij')
		labels_pred_match = tf.gather_nd(out_pcl_seg, tf.stack([idx, pts_match_fwd], -1))
		labels_gt_match = tf.gather_nd(labels_gt, tf.stack([idx, pts_match_bwd], -1))
		seg_loss_fwd, _, _, _, _ = get_seg_losses(out_pcl_seg, labels_gt_match, labels_gt_wts) 
		seg_loss_bwd, _, _, _, _ = get_seg_losses(labels_pred_match, labels_gt, labels_gt_wts)
		seg_loss = seg_loss_fwd + seg_loss_bwd

	elif FLAGS.loss_seg == 'emd':
		pts_match = get_labels_seg(out_pcl, pcl_gt, FLAGS.loss_seg)
		idx, _ = tf.meshgrid(range(BATCH_SIZE), range(NUM_POINTS), indexing='ij')
		labels_gt_match = tf.gather_nd(labels_gt, tf.stack([idx, pts_match], -1))
		_, seg_loss, _, _, _ = get_seg_losses(out_pcl_seg, labels_gt_match, labels_gt_wts)

	# Loss
	if FLAGS.loss_recon == 'chamfer':
		loss_recon = chamfer_distance
	elif FLAGS.loss_recon == 'emd':
		loss_recon = emd
	elif FLAGS.loss_recon == 'both':
		loss_recon = chamfer_distance + scale_emd(emd, NUM_POINTS)
	else:
		sys.exit('Loss should be chamfer or emd or both')

	loss = lambda_seg * seg_loss
	loss += (lambda_recon * loss_recon)

	# Training and Val data
	print '_'*30, ' DONE  loading models ', '_'*30
	batches = len(train_pair_indices) / BATCH_SIZE

	train_vars = [var for var in tf.global_variables() if 'joint_seg_net' in var.name]

	# Optimizer
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss, var_list=train_vars)

	# Training params
	start_epoch = 1
	max_epoch = FLAGS.max_epoch
	N_ITERS = batches * max_epoch

	# Define Logs Directories
	snapshot_folder = join(EXP_DIR, 'snapshots')
	best_folder = join(EXP_DIR, 'best')
	logs_folder = join(EXP_DIR, 'logs')
	log_file = join(EXP_DIR, 'logs.txt')

	# Define savers to load and store models
	saver = tf.train.Saver(max_to_keep=100)

	# Define summary variables
	summary_loss = tf.summary.scalar('loss', loss)
	summary_seg_loss = tf.summary.scalar('seg_loss', seg_loss)
	summary_forward_scaled = tf.summary.scalar('dists_forward_scaled', dists_forward_scaled)
	summary_backward_scaled = tf.summary.scalar('dists_backward_scaled', dists_backward_scaled)
	summary_chamfer_scaled = tf.summary.scalar('chamfer_distance_scaled', chamfer_distance_scaled)
	summary_emd_scaled = tf.summary.scalar('emd_scaled', emd_scaled)
	summ = tf.summary.merge_all()

	# Create log directories
	create_folder(snapshot_folder)
	create_folder(logs_folder)
	create_folder(best_folder)

	# GPU configurations
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(config=config) as sess:

		print 'Session started'
		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		print 'running initializer'
		sess.run(tf.global_variables_initializer())
		print 'done'

		# Load previous checkpoint
		ckpt = tf.train.get_checkpoint_state(snapshot_folder)
		if ckpt is not None:
			print ('loading '+ckpt.model_checkpoint_path + '  ....')
			saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1)) + 1

		ind = 0
		best_val = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		if start_epoch == 1:
			with open(log_file, 'w') as f:
				f.write(' '.join(['Epoch','Train_loss','Train_seg','Train_chamf','Train_fwd','Train_bkwd','Val_loss','Val_seg','Val_emd','Val_chamf','Val_fwd','Val_bkwd','Minutes','Seconds','\n']))

		for i in xrange(start_epoch, max_epoch+1):

			random.shuffle(train_pair_indices)
			train_epoch_loss = 0.
			train_epoch_seg = 0.
			train_epoch_chamfer = 0.
			train_epoch_forward = 0.
			train_epoch_backward = 0.

			train_loss_PRINT_N = 0.
			train_seg_PRINT_N = 0.
			train_chamfer_PRINT_N = 0.
			train_fwd_PRINT_N = 0.
			train_bkwd_PRINT_N = 0.

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt, batch_lbl, batch_lbl_wts = fetch_batch_joint(train_models, train_pair_indices, b, BATCH_SIZE)
				if FLAGS.schedule_recon:
					_lambda_recon = FLAGS.LAMBDA_RECON * (np.exp(-global_step/N_ITERS))
				else:
					_lambda_recon = FLAGS.LAMBDA_RECON

				if FLAGS.schedule_seg:
					_lambda_seg = FLAGS.LAMBDA_SEG * (1 -
							np.exp(-global_step/N_ITERS))
				else:
					_lambda_seg = FLAGS.LAMBDA_SEG

				feed_dict = {img_inp:batch_ip,
							pcl_gt:batch_gt, 
							labels_gt: batch_lbl,
							labels_gt_wts: batch_lbl_wts,
							lambda_recon: _lambda_recon, 
							lambda_seg: _lambda_seg}

				L, C, F, B, S, _ = sess.run([loss, loss_recon, dists_forward, dists_backward, seg_loss, optim], feed_dict=feed_dict)

				train_epoch_loss += L/batches
				train_epoch_seg += S/batches
				train_epoch_chamfer += C/batches
				train_epoch_forward += F/batches
				train_epoch_backward += B/batches

				train_loss_PRINT_N += L/PRINT_N
				train_seg_PRINT_N += S/PRINT_N
				train_chamfer_PRINT_N += C/PRINT_N
				train_fwd_PRINT_N += F/PRINT_N
				train_bkwd_PRINT_N += B/PRINT_N

				if global_step % PRINT_N == 0:
					C, E = sess.run([chamfer_distance, emd], feed_dict)
					_summ = sess.run(summ, feed_dict)
					train_writer.add_summary(_summ, global_step)
					time_elapsed = time.time() - since
					print '{} batches: Loss = {}  Seg = {:.4f}  Fwd = {:.6f}  Bkwd = {:.6f};  1 batch: Chamfer = {:.6f}  Emd = {:.6f} Iter = {} Minibatch = {}  Time = {:.0f}m {:.0f}s'.format(PRINT_N, train_loss_PRINT_N, train_seg_PRINT_N, train_fwd_PRINT_N, train_bkwd_PRINT_N, train_chamfer_PRINT_N, scale_emd(E, NUM_POINTS), global_step, b, time_elapsed//60, time_elapsed%60)

					train_loss_PRINT_N = 0.
					train_seg_PRINT_N = 0.
					train_chamfer_PRINT_N = 0.
					train_fwd_PRINT_N = 0.
					train_bkwd_PRINT_N = 0.

			if i % FLAGS.save_n == 0:
				print 'Saving Model at epoch %d....................'%i
				saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
				print '..................... Model Saved'

			# Val metrics
			val_epoch_loss, val_epoch_seg, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_epoch_emd, _summ = get_epoch_loss(val_models, val_pair_indices, VAL_BATCH_SIZE, _lambda_recon, _lambda_seg)
			val_writer.add_summary(_summ, global_step)

			time_elapsed = time.time() - since

			with open(log_file, 'a') as f:
				epoch_str = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.0f} {:.0f}'.format(i, train_epoch_loss, train_epoch_seg, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, val_epoch_seg, val_epoch_emd, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60)
				f.write(epoch_str+'\n')

			# Update best model if necessary
			best_val = update_best(sess, saver, val_epoch_loss, best_val, snapshot_folder, best_folder)

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'TRAIN Loss: {:.4f}  Seg: {:.4f}  Chamfer: {:.4f} Forward: {:.6f}  Backward: {:.6f} | Val Seg: {:.4f}  emd: {:.6f} Chamfer: {:.6f}  Forward: {:.6f}  Backward: {:.6f} Time:{:.0f}m {:.0f}s'.format(train_epoch_loss, train_epoch_seg, train_epoch_chamfer, train_epoch_forward, train_epoch_backward, val_epoch_seg, val_epoch_emd, val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60)
			print 'Best val loss so far: {:.6f}'.format(best_val)
			print '-'*140
			print
