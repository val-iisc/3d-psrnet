import os
import sys
import json
import argparse
import cv2
import numpy as np
import random
import tensorflow as tf
import time
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import ipdb

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('utils')

from utils.tf_ops.cd import tf_nndistance
from utils.tf_ops.emd.tf_auctionmatch import auction_match
from utils.dataloader import *


shapenet_category_to_id = {'chair':'03001627', 'aero':'02691156', 'car':'02958343'}
shapenet_id_to_category = {'03001627':'chair', '02691156':'aero', '02958343':'car'}

PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png', 'render_8.png', 'render_9.png']


def update_best(sess, saver, current_loss, best_loss, snapshot_folder, best_folder):
	if (current_loss < best_loss):
			print 'Best model found. Saving in %s...'%best_folder
			saver.save(sess, join(snapshot_folder, 'best', 'best'))
			os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
			return current_loss
	return best_loss


def scale_emd(emd, NUM_POINTS):
	return 0.01*emd/float(NUM_POINTS)


def get_seg_loss(seg_pred, seg_gt):
	# seg_pred --> (BATCH_SIZE x NUM_POINTS x NUM_CLASSES)
	# seg_gt --> (BATCH_SIZE x NUM_POINTS)
	per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg_gt), axis=1)
	seg_loss = tf.reduce_mean(per_instance_seg_loss)
	per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

	return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


def get_seg_losses(seg_pred, seg_gt, seg_wts):
	# seg_pred --> (BATCH_SIZE x NUM_POINTS x NUM_CLASSES)
	# seg_gt --> (BATCH_SIZE x NUM_POINTS)
	# seg_wts --> (BATCH_SIZE x NUM_POINTS)

	# seg_loss
	per_point_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg_gt) # (bs x num_points)
	per_instance_seg_loss = tf.reduce_mean(per_point_seg_loss, axis=1) # (bs,)
	seg_loss = tf.reduce_mean(per_instance_seg_loss) # (1,)

	# weighted_seg_loss
	per_point_seg_loss_weighted = per_point_seg_loss * seg_wts # (bs x num_points)
	per_instance_seg_loss_weighted = tf.reduce_mean(per_point_seg_loss_weighted, axis=1) # (bs,)
	weighted_seg_loss = tf.reduce_mean(per_instance_seg_loss_weighted) # (1,) 

	# predicted get_labels_seg
	per_instance_seg_pred_res = tf.argmax(seg_pred, 2) # (bs,)

	return seg_loss, weighted_seg_loss, per_instance_seg_loss, per_instance_seg_loss_weighted, per_instance_seg_pred_res


def get_labels_seg(pcl_gt, pcl_pred, metric):
	'''
        Point wise correspondences between two point sets
        args:
            pcl_gt: (batch_size, n_pts, 3), gt pcl
            pcl_pred: (batch_size, n_pts, 3), predicted pcl
            metric: str, 'chamfer' or 'emd'
                    metric to be considered for returning corresponding points
        returns:
            pts_match_fwd: gt to pred point-wise correspondence
                           each point in gt is mapped to nearest point in pred
            pts_match_bwd: pred to gt point-wise correspondence
                           each point in pred is mapped to nearest point in gt
            pts_match: one-to-one mapping between pred and gt, acc. to emd
	'''
	if metric == 'chamfer':
		_, pts_match_fwd, _, pts_match_bwd = tf_nndistance.nn_distance(pcl_gt, pcl_pred) 
		return pts_match_fwd, pts_match_bwd
	elif metric == 'emd':
		pts_match, _ = auction_match(pcl_gt, pcl_pred)
		return pts_match
	else:
		print 'Undefined metric'
		return None 

