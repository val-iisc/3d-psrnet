from __future__ import division
import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
from itertools import product
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import ipdb

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('utils')


shapenet_category_to_id = {'chair':'03001627', 'aero':'02691156', 'car':'02958343'}
shapenet_id_to_category = {'03001627':'chair', '02691156':'aero', '02958343':'car'}

PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png', 'render_8.png', 'render_9.png']


# xyz is a batch
def tf_rotate(xyz, xangle=0, yangle=0, BATCH_SIZE=10):
	xangle = np.pi*xangle/180
	yangle = np.pi*yangle/180
	rotmat = np.eye(3)

	rotmat=rotmat.dot(np.array([
		[1.0,0.0,0.0],
		[0.0,np.cos(xangle),-np.sin(xangle)],
		[0.0,np.sin(xangle),np.cos(xangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(yangle),0.0,-np.sin(yangle)],
		[0.0,1.0,0.0],
		[np.sin(yangle),0.0,np.cos(yangle)],
		]))

	_rotmat = tf.constant(rotmat, dtype=tf.float32)
	_rotmat = tf.reshape(tf.tile(_rotmat,(BATCH_SIZE,1)), shape=(BATCH_SIZE,3,3))

	return tf.matmul(xyz,_rotmat)


# xyz is a single pcl
def np_rotate(xyz, xangle=0, yangle=0, inverse=False):
	rotmat = np.eye(3)

	rotmat=rotmat.dot(np.array([
		[1.0,0.0,0.0],
		[0.0,np.cos(xangle),-np.sin(xangle)],
		[0.0,np.sin(xangle),np.cos(xangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(yangle),0.0,-np.sin(yangle)],
		[0.0,1.0,0.0],
		[np.sin(yangle),0.0,np.cos(yangle)],
		]))

	if inverse:
		rotmat = np.linalg.inv(rotmat)

	return xyz.dot(rotmat)


def scale(gt_pc, pr_pc): #pr->[-0.5,0.5], gt->[-0.5,0.5]

	pred = tf.cast(pr_pc, dtype=tf.float32)
	gt   = tf.cast(gt_pc, dtype=tf.float32)

	min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)])
	max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)])
	min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)])
	max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)])

	length_gt = tf.abs(max_gt - min_gt)
	length_pr = tf.abs(max_pr - min_pr)

	diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt
	diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr

	new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)])
	new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)])
	new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)])
	new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)])

	size_pr = tf.reduce_max(length_pr, axis=0)
	size_gt = tf.reduce_max(length_gt, axis=0)

	scaling_factor_gt = 1. / size_gt # 1. is the length of the [-0.5,0.5] cube
	scaling_factor_pr = 1. / size_pr

	box_min = tf.ones_like(new_min_gt) * -0.5

	adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
	adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr

	pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
	gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

	return gt_scaled, pred_scaled


def create_folder(folder):
	if not exists(folder):
		makedirs(folder)


def fetch_pcl(model_path):
	pcl_filename = 'pointcloud_1024.npy'
	pcl_path = join(model_path, pcl_filename)
	pcl_gt = np.load(pcl_path)
	return pcl_gt


def fetch_image(model_path, index):
	img_path = join(model_path, PNG_FILES[index])
	ip_image = cv2.imread(img_path)
	ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
	return ip_image


def fetch_labels(model_path):
	label_path = join(model_path, 'pointcloud_labels.npy')
	labels_gt = np.load(label_path)
	labels_gt -= 1
	return labels_gt


def get_label_wts(label):
	'''
	Computes weight for every point based on class count of that point
	Args:
		label: class labels for each point in a pcl --> (NUM_POINTS)
	Returns:
		wts: class weights for each point in a pcl --> (NUM_POINTS)
	'''
	cnt = np.bincount(label)
	tot_cnt = np.sum(cnt)
	classes = np.nonzero(cnt)[0]
	wts = []
	cnt_dict = dict(zip(classes,cnt[classes]))
	for lbl in label:
		wts.append(tot_cnt/cnt_dict[lbl])
	wts = np.asarray(wts, dtype=np.float32)
	return wts


def fetch_batch_joint(models, indices, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	batch_lbl = []
	batch_label_wts = []
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		pcl_gt = fetch_pcl(model_path)
		labels_gt = fetch_labels(model_path)
		label_wts = get_label_wts(labels_gt)
		ip_image = fetch_image(model_path, ind[1])
		batch_ip.append(ip_image)
		batch_gt.append(pcl_gt)
		batch_lbl.append(labels_gt)
		batch_label_wts.append(label_wts)
	batch_ip = np.array(batch_ip)
	batch_gt = np.array(batch_gt)
	batch_lbl = np.array(batch_lbl)
	batch_label_wts = np.array(batch_label_wts)
	return batch_ip, batch_gt, batch_lbl, batch_label_wts


def get_drc_models_util(data_dir, category, eval_set):

	models = []

	if category == 'all':
		cats = ['chair','car','aero']
	else:
		cats = [category]

	for cat in cats:
		category_id = shapenet_category_to_id[cat]
		splits_file_path = join(data_dir, 'splits', category_id+'_%s_list.txt'%eval_set)
		
		with open(splits_file_path, 'r') as f:
			for model in f.readlines():
				models.append(join(data_dir, category_id,model.strip()))

	return models


def get_drc_models(data_dir, category, NUM_VIEWS, eval_set):
	models = get_drc_models_util(data_dir, category, eval_set)
	pair_indices = list(product(xrange(len(models)), xrange(NUM_VIEWS)))
	print '{}: models={}  samples={}'.format(eval_set, len(models),len(models)*NUM_VIEWS)
	return models, pair_indices

