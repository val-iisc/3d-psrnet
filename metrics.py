'''
Code for computing reconstruction (chamfer, emd) and segmentation (mIoU) metrics.

Usage:
	visualize:
	python metrics.py --data_dir data/ --exp ./trained_models --gpu 0 --category chair --eval_set test --snapshot best --n_cls 4 --visualize
	
	metrics:
	python metrics.py --data_dir data/ --exp ./trained_models --gpu 0 --category chair --eval_set test --snapshot best --n_cls 4 --tqdm


'''

import tensorflow as tf
import tflearn
from tqdm import tqdm

from utils.metrics_utils import *
from utils.train_utils import *
from utils.net import joint_seg_net

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, required=True, 
	help='Path to dataset')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment Prefixed with index')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
	help='Category to visualize from : ["airplane", "car", "chair"]')
parser.add_argument('--eval_set', type=str, required=True, 
	help='set to compute metrics on : ["train", "test"]')
parser.add_argument('--snapshot', type=str, required=True, 
	help='Load snapshot : ["<epoch>" ,"best_emd", "best_chamfer"]')
parser.add_argument('--batch_size', type=int, default=10, 
	help='Batch Size during evaluation. Make sure to set a value that perfectly divides the total number of samples.')
parser.add_argument('--n_cls', type=int, required=True, 
	help='Number of classes for category')
parser.add_argument('--debug', action='store_true', 
	help='debug mode. only run the first 10 samples')
parser.add_argument('--freq_wt', action='store_true', 
	help='Calculate frequency weighted IOU')
parser.add_argument('--visualize', action='store_true', 
	help='visualize generated point clouds')
parser.add_argument('--tqdm', action='store_true', 
	help='view progress bar')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

random.seed(1024)

data_dir = FLAGS.data_dir
BATCH_SIZE = FLAGS.batch_size
HEIGHT = 64
WIDTH = 64
NUM_VIEWS = 10
NUM_POINTS = 1024
NUM_CLASSES = FLAGS.n_cls
ballradius = 3


if __name__=='__main__':

	exp_dir = os.path.abspath(FLAGS.exp)
	# Snapshot Folder Location
	snapshot_folder = join(BASE_DIR, exp_dir, 'snapshots')

	# use case
	if FLAGS.visualize:
		from utils.show_3d import show3d_balls
	else:
		if exp_dir == '':
			print 'exp name is empty! Check code.'
		metrics_dir = join(exp_dir, 'metrics', FLAGS.eval_set)
		if not exists(metrics_dir):
			os.makedirs(metrics_dir)
		csv_path = join(metrics_dir, '%s_%s.csv'%(FLAGS.category, FLAGS.snapshot))
		with open(csv_path, 'w') as f:
			f.write('Id;Chamfer;Emd;mIoU\n')
		avg_csv_path = join(metrics_dir, '%s_%s_avg.csv'%(FLAGS.category, FLAGS.snapshot))

	# placeholders
	img_inp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt = tf.placeholder(tf.float32, shape=(None, NUM_POINTS, 3), name='pcl_gt')
	pred_pcl = tf.placeholder(tf.float32, shape=(None, NUM_POINTS, 3), name='pcl_pred')
	labels_gt = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINTS), name='labels_gt')

	# Build graph
	with tf.variable_scope('joint_seg_net'):
		pred_pcl, labels_pred = joint_seg_net(img_inp, NUM_POINTS, NUM_CLASSES)

	# metrics - reconstruction  (chamfer, emd)
	gt_pcl_scaled, pred_pcl_scaled = scale(pcl_gt, pred_pcl)
	dists_forward, dists_backward, chamfer_distance, emd = get_rec_metrics(gt_pcl_scaled, pred_pcl_scaled)
	
	# metrics - segmentation (iou)
	pts_match_fwd, pts_match_bwd = get_labels_seg(pred_pcl_scaled, gt_pcl_scaled, 'chamfer')
	labels_gt_match = tf.stack([tf.gather(labels_gt[k], pts_match_fwd[k], axis=0) for k in range(BATCH_SIZE)], axis=0)
	labels_pred_match = tf.stack([tf.gather(labels_pred[k], pts_match_bwd[k], axis=0) for k in range(BATCH_SIZE)], axis=0)

	seg_loss_fwd, per_inst_fwd, per_inst_fwd_lbl = get_seg_loss(labels_pred, labels_gt_match) 
	seg_loss_bwd, per_inst_bwd, per_inst_bwd_lbl = get_seg_loss(labels_pred_match, labels_gt)
	per_instance_seg_loss = per_inst_fwd + per_inst_bwd
	seg_loss = seg_loss_fwd + seg_loss_bwd

	labels_pred_idx = tf.argmax(labels_pred, axis=2)
	iou_fwd, iou_bwd = get_iou(gt_pcl_scaled, pred_pcl_scaled, labels_gt, labels_pred_idx, NUM_CLASSES,'chamfer', FLAGS.freq_wt)

	# GPU configurations
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())

		# saver to load previous checkpoint
		saver = tf.train.Saver()
		load_previous_checkpoint(snapshot_folder, saver, sess, exp_dir, FLAGS.snapshot)
		
		tflearn.is_training(False, session=sess)

		# data
		if FLAGS.eval_set == 'train':
			models,indices = get_drc_models(data_dir, FLAGS.category, NUM_VIEWS, 'train')
		elif FLAGS.eval_set == 'val':
			models,indices = get_drc_models(data_dir, FLAGS.category, NUM_VIEWS, 'val')
		elif FLAGS.eval_set == 'test':
			models,indices = get_drc_models(data_dir, FLAGS.category, NUM_VIEWS, 'test')
		else:
			print 'Enter train/val/test for eval_set'
			sys.exit(1)

		if FLAGS.visualize:
			random.shuffle(indices)
		batches = len(indices) // BATCH_SIZE

		print('computing metrics for %d samples...'%len(indices))
		if FLAGS.debug:
			iters = 10
		else:
			iters = range(batches)
		if FLAGS.tqdm:
			iters = tqdm(iters)

		iou_parts_all = []
		for cnt in iters:

			# load batch
			batch_ip, batch_gt, batch_lbl, batch_lbl_wts = fetch_batch_joint(models, indices, cnt, BATCH_SIZE)
			fids = fetch_batch_paths(models, indices, cnt, BATCH_SIZE)
			feed_dict={img_inp:batch_ip, pcl_gt:batch_gt, labels_gt: batch_lbl}

			_gt_scaled, _pred_scaled, _labels_pred = sess.run([gt_pcl_scaled, pred_pcl_scaled, labels_pred], feed_dict)

			# metrics
			# C,F,B,E are all arrays of dimension (BATCH_SIZE,)
			feed_dict={gt_pcl_scaled:_gt_scaled, pred_pcl_scaled:_pred_scaled, labels_gt:batch_lbl, labels_pred:_labels_pred}
			C,F,B,E = sess.run([chamfer_distance, dists_forward, dists_backward, emd], feed_dict)
			_pi_seg_loss, _pi_seg_pred = sess.run([per_instance_seg_loss, per_inst_fwd_lbl], feed_dict)
			_iou_fwd, _iou_bwd = sess.run([iou_fwd, iou_bwd], feed_dict)
			S = _pi_seg_loss

			IOU_FWD = []; IOU_BWD = [];
			for key in sorted(_iou_fwd.keys()):
				IOU_FWD.append(_iou_fwd[key])
				IOU_BWD.append(_iou_bwd[key])

			IOU_FWD = np.asarray(IOU_FWD)
			IOU_BWD = np.asarray(IOU_BWD)
			iou_parts = (IOU_FWD+IOU_BWD)/2.
			iou_parts_all.append(iou_parts)
			IOU_FWD = np.mean(IOU_FWD, axis=0)
			IOU_BWD = np.mean(IOU_BWD, axis=0)

			# visualize
			if FLAGS.visualize:
				pr = tf_rotate(_pred_scaled,-90,-90).eval()
				gt = tf_rotate(_gt_scaled,-90,-90).eval()
				for b in xrange(BATCH_SIZE):
					print 'Model:{} C:{:.6f} F:{:.6f} B:{:.6f} E:{:.6f} IOU_FWD:{:.4f} IOU_BWD:{:.4f}'.format(fids[b],C[b],F[b],B[b],E[b],IOU_FWD[b],IOU_BWD[b])
					cv2.imshow('', batch_ip[b])
					show3d_balls.showpoints_partseg(gt[b], batch_lbl[b], ballradius=3)
					show3d_balls.showpoints_partseg(pr[b], _pi_seg_pred[b], ballradius=3)

			# save metrics to csv
			else:
				if np.isnan(C).any() or np.isnan(E).any():
					print fids
					print C
					print E
				else:
					with open(csv_path, 'a') as f:
						for b in xrange(BATCH_SIZE):
							f.write('{};{:.6f};{:.6f};{:.6f}\n'.format(fids[b],C[b],E[b],(IOU_FWD[b]+IOU_BWD[b])/2.))

		# get avg metrics
		C_avg,E_avg,iou_avg = get_averages(csv_path)
		print 'Final Metrics:  Chamfer  EMD  IOU'.format(C_avg,E_avg,iou_avg)
		print '{:.6f};{:.6f};{:.6f}'.format(C_avg,E_avg,iou_avg)

		with open(avg_csv_path, 'w') as f:
			f.write('Chamfer; Emd; mIoU\n')
			f.write('{:.6f};{:.6f};{:.6f}'.format(C_avg,E_avg,iou_avg))
