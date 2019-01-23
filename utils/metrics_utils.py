from __future__ import division
import os
import sys
import numpy as np
import tensorflow as tf
import time
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import csv
import ipdb

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('utils')

from utils.tf_ops.cd import tf_nndistance
from utils.tf_ops.emd.tf_auctionmatch import auction_match
from utils.dataloader import *
from utils.train_utils import get_labels_seg, get_seg_loss


# Returns image_ids
def fetch_batch_paths(models, indices, batch_num, batch_size):

	paths = []
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		try:
			l = model_path.strip().split('/')
			category_id = l[-2]
			model_id = l[-1]
			fid = '_'.join([l[-2], l[-1], str(ind[1])])	
			paths.append(fid)
		except:
			print fid
			pass
	return paths


def get_rec_metrics(gt_pcl, pred_pcl, batch_size=10, num_points=1024):
	
	dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
	dists_forward = tf.reduce_mean(tf.sqrt(dists_forward), axis=1) # (B, )
	dists_backward = tf.reduce_mean(tf.sqrt(dists_backward), axis=1) # (B, )
	chamfer_distance = dists_backward + dists_forward

	X,_ = tf.meshgrid(range(batch_size), range(num_points), indexing='ij')
	ind, _ = auction_match(pred_pcl, gt_pcl) # Ind corresponds to points in pcl_gt
	ind = tf.stack((X, ind), -1)
	emd = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.gather_nd(gt_pcl, ind) - pred_pcl)**2, axis=-1)), axis=1) # (BATCH_SIZE,NUM_POINTS,3) --> (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)

	return dists_forward, dists_backward, chamfer_distance, emd


def get_seg_metrics(gt_pcl, gt_labels, pred_pcl, pred_labels, batch_size=10, num_classes=4, freq_wt=False):

	# cross entropy
	pts_match_fwd, pts_match_bwd = get_labels_seg(gt_pcl, pred_pcl, 'chamfer')
	pred_labels_match = tf.stack([tf.gather(pred_labels[k], pts_match_fwd[k], axis=0) for k in range(batch_size)], axis=0)
	gt_labels_match = tf.stack([tf.gather(gt_labels[k], pts_match_bwd[k], axis=0) for k in range(batch_size)], axis=0)
	seg_loss_fwd, per_inst_fwd, per_inst_fwd_lbl = get_seg_loss(pred_labels_match, gt_labels) 
	seg_loss_bwd, per_inst_bwd, per_inst_bwd_lbl = get_seg_loss(pred_labels, gt_labels_match)
	per_instance_seg_loss = per_inst_fwd + per_inst_bwd
	seg_loss = (seg_loss_fwd + seg_loss_bwd)/2.

	# IoU
	labels_pred_idx = tf.argmax(pred_labels, axis=2)
	per_part_iou_fwd, per_part_iou_bwd = get_iou(gt_pcl, pred_pcl,
				gt_labels, labels_pred_idx, num_classes,'chamfer', freq_wt) # dict
	per_instance_iou_fwd = get_mIoU(per_part_iou_fwd, freq_wt)
	per_instance_iou_bwd = get_mIoU(per_part_iou_bwd, freq_wt)
	per_instance_iou = (per_instance_iou_fwd + per_instance_iou_bwd)/2.
	
	return per_instance_seg_loss, per_inst_fwd, per_inst_bwd, per_instance_iou, per_instance_iou_fwd, per_instance_iou_bwd


def get_mIoU(iou_dict, freq_wt):
	'''
	Args:
		iou_dict: keys are [0...n_cls]; values are of dim (n_cls, bs)
	Returns:
		mIoU: (bs,)
	'''

	mIoU = []
	for key in sorted(iou_dict.keys()):
		mIoU.append(iou_dict[key])
	mIoU = tf.stack(mIoU)
	if not freq_wt:
		mIoU = tf.reduce_mean(mIoU, axis=0)

	return mIoU


def get_iou(pcl_gt, pcl_pred, labels_gt, labels_pred, n_cls, metric, freq_wt=False):
	'''
	'''
	batch_size, num_points = labels_gt.get_shape()
	if metric == 'chamfer':
		pts_match_fwd, pts_match_bwd = get_labels_seg(pcl_gt, pcl_pred, 'chamfer')
	elif metric == 'emd':
		pts_match = get_labels_seg(pcl_gt, pcl_pred, 'emd')

	idx, _ = tf.meshgrid(range(batch_size), range(num_points), indexing='ij')
	labels_pred_match = tf.gather_nd(labels_pred, tf.stack([idx, pts_match_fwd], -1))
	labels_gt_match = tf.gather_nd(labels_gt, tf.stack([idx, pts_match_bwd], -1))
	
	iou_fwd, _dict = calc_iou(labels_gt, labels_pred_match, n_cls, freq_wt)
	iou_bwd, _ = calc_iou(labels_gt_match, labels_pred, n_cls, freq_wt)

	_dict['lbprm'] = labels_pred_match
	_dict['lbgtm'] = labels_gt_match

	return iou_fwd, iou_bwd


def calc_iou(labels_gt, labels_pred, n_cls, freq_wt=False):
	'''
	'''
	_dict = {};
	gt_onehot = {}; pred_onehot = {}; tp = {}; union = {}; iou = {}; 
	cls_cnt = [];
	for cls in range(n_cls):
		gt_onehot[cls] = tf.to_float(tf.equal(labels_gt, cls))
		cls_cnt.append(tf.reduce_sum(gt_onehot[cls], axis=1))
		pred_onehot[cls] = tf.to_float(tf.equal(labels_pred, cls))
		tp[cls] = tf.reduce_sum(tf.to_float(tf.equal(gt_onehot[cls],
				pred_onehot[cls]))*pred_onehot[cls], axis=1)
		union[cls] = tf.reduce_sum(gt_onehot[cls], axis=1) +\
							tf.reduce_sum(pred_onehot[cls], axis=1) - tp[cls]
		if freq_wt:
			iou[cls] = cls_cnt[cls]*(tp[cls] / (1.*union[cls] + 1e-8))
		else:
			iou[cls] = tf.where(union[cls]>0,
								tp[cls] / (1.*union[cls]),
								tf.ones_like(tp[cls]/(1.*union[cls]))
								)
	tot_cnt = tf.reduce_sum(tf.stack(cls_cnt, axis=1), axis=1)
	if freq_wt:
		for cls in range(n_cls):
			iou[cls] = iou[cls] / tot_cnt

	_dict['gt'] = gt_onehot
	_dict['pred'] = pred_onehot
	_dict['tp'] = tp
	_dict['un'] = union
	_dict['iou'] = iou
	_dict['cnt'] = cls_cnt
	return iou, _dict


def get_averages(csv_path):
	column_sums = None
	with open(csv_path) as f:
		lines = f.readlines()[1:]
		rows_of_numbers = [map(float, line.split(';')[1:]) for line in lines]
		sums = map(sum, zip(*rows_of_numbers))
		averages = [sum_item / len(lines) for sum_item in sums]
		return averages


def load_previous_checkpoint(snapshot_folder, saver, sess, exp, snapshot):

	if snapshot == 'best': # only seg training
		ckpt_path = join(exp, 'best', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)

	elif snapshot == 'best_emd':
		ckpt_path = join(exp, 'best_emd', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)

	elif snapshot == 'best_chamfer':
		ckpt_path = join(exp, 'best_chamfer', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)

	elif snapshot == 'best_seg':
		ckpt_path = join(exp, 'best_seg', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)

	elif snapshot == 'best_joint':
		ckpt_path = join(exp, 'best_joint', 'best')
		print ('loading ' + ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)
	else:
		try:
			epoch_num = int(snapshot)
		except:
			print 'Check the snapshot entered'
			sys.exit(1)
		ckpt = tf.train.get_checkpoint_state(snapshot_folder)
		if ckpt is not None:
			ckpt_path = os.path.abspath(ckpt.model_checkpoint_path)
			ckpt_path = join(snapshot_folder, 'model-%d'%epoch_num)
			print ('loading '+ckpt_path + '  ....')
			saver.restore(sess, ckpt_path)
		else:
			print 'Failed to load checkpoint'
			sys.exit(1)
	return
	

