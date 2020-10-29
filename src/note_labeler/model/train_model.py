import os, sys
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from note_labeler.model.utils import load_embedding_vectors_word2vec_gensim as load_word2vec_matias
from math import floor
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
import gc
import pickle
from note_labeler.model.model import *
from note_labeler.model.utils import *
from note_labeler.model.AL_utils import *
import ast

def get_entropy(p):
	#p=np.array(prob)
	entropy=(-p * np.log2(p)-(1-p)*np.log2(1-p))
	entropy=np.nan_to_num(entropy)
	return entropy

class Options(object):
	def __init__(self):
		self.GPUID = 0
		self.dataset = None
		self.fix_emb = True
		self.restore = False
		self.W_emb = None
		self.W_class_emb = None
		self.maxlen = 538
		self.n_words = None
		self.embed_size = 300
		self.lr = 1e-3
		self.batch_size = 50
		self.max_epochs = 20
		self.dropout = 0.2
		self.part_data = False
		self.portion = 1.0
		self.save_path = "./model/kidney/model/"
		self.log_path = "./model/log/"
		self.print_freq = 100
		self.valid_freq = 10
		self.optimizer = 'Adam'
		self.clip_grad = None
		self.class_penalty = 1.0
		self.ngram = 60
		self.H_dis = 64


	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value


def get_model(train, train_lab,val, val_lab, opt, max_auc=None, det_data=None):
	"""
	Retrained the model and saved it in the original folder.
	inputs:
		train: X training; train_lab: y train; val: X validation; test: X test; test_lab: y test; opt: class containing the hyper-parameters
	output:
		test auc, validation auc, list of determinants for the unlabeled set
	"""
	updated=False
	tf.reset_default_graph()
	pred_batch_size = 80
	with tf.device('/gpu:0'):
		x_ = tf.placeholder(tf.int32, shape=[None, opt.maxlen])
		x_mask_ = tf.placeholder(tf.float32, shape=[None, opt.maxlen])
		keep_prob = tf.placeholder(tf.float32)
		y_ = tf.placeholder(tf.float32, shape=[None, opt.num_class])
		class_penalty_ = tf.placeholder(tf.float32, shape=())
		prob_, logits_, loss_, train_op, W_norm_, global_step, H_enc_, last_layer_ = emb_classifier(
			x_, x_mask_, y_, keep_prob, opt, class_penalty_)
	config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
	config.gpu_options.allow_growth = True
	np.set_printoptions(precision=3)
	np.set_printoptions(threshold=np.inf)
	saver = tf.train.Saver()
	covariance_input=None
	validation_list=[]
	uidx = 0
	init_train=True
	max_val_auc_mean=0
	if max_auc is not None and det_data is not None:
		init_train=False
	if not init_train:
		max_val_auc_mean = max_auc
	val_auc_mean_list = []
	val_auc_lists = []
	with tf.Session(config=config) as sess:
			train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
			test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
			saver = tf.train.Saver()

			if opt.restore:
				try:
					t_vars = tf.trainable_variables()
					save_keys = tensors_key_in_file(opt.save_path)
					ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
					cc = {var.name: var for var in t_vars}
					# only restore variables with correct shape
					ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

					loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
					loader.restore(sess, opt.save_path)

					print("Loading variables from '%s'." % opt.save_path)
					print("Loaded variables:" + str(ss))

				except:
					print("No saving session, using random initialization")
					sess.run(tf.global_variables_initializer())
					sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

			try:
				#opt.max_epochs = 320
				epoch_num_iter = 5000
				opt.valid_freq=1
				for epoch in range(opt.max_epochs):
					print("Starting epoch %d" % epoch)
					train_loss_list = val_loss_list = []
					#kf = get_balanced_batch_idx(len(train), train_lab, opt.batch_size, opt.num_class, epoch_num_iter)
					kf=get_minibatches_idx(len(train), len(train), shuffle=False)
					#kf=get_balanced_batch(train_lab, len(train_lab))
					for _, train_index in kf:
						uidx += 1
						sents = [train[t] for t in train_index]
						x_labels = [train_lab[t] for t in train_index]
						x_labels = np.array(x_labels)
						x_labels = x_labels.reshape((len(x_labels), opt.num_class))
						x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
						_, train_loss, step = sess.run([train_op, loss_, global_step],
													   feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels,
																  keep_prob: opt.dropout, class_penalty_:opt.class_penalty})
						train_loss_list.append(train_loss)

						if uidx % opt.valid_freq == 0:
							train_logits_list = []
							train_prob_list = []
							train_true_list = []

							kf_train = get_minibatches_idx(len(train), len(train), shuffle=False)
							for _, train_index in kf_train:
								train_sents = [train[t] for t in train_index]
								train_labels = [train_lab[t] for t in train_index]
								train_labels = np.array(train_labels)
								train_labels = train_labels.reshape((len(train_labels), opt.num_class))
								x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)
								train_prob, train_logits , train_last_layer= sess.run([prob_, logits_,last_layer_],
																	feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask,
																			   y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})
								#last_layer needs to be appended if not full batches
								#print(type(train_last_layer))
								#train_last_layer=train_last_layer.numpy()
								train_logits_list += train_logits.tolist()
								train_prob_list += train_prob.tolist()
								train_true_list += train_labels.tolist()

							train_logits_array = np.asarray(train_logits_list)
							train_prob_array = np.asarray(train_prob_list)
							train_true_array = np.asarray(train_true_list)
							train_auc_list = []
							for i in range(opt.num_class):
								train_auc = roc_auc_score(y_true = train_true_array[:,i], y_score = train_logits_array[:,i],)
								train_auc_list.append(train_auc)

							train_auc_mean = np.mean(train_auc_list)

							val_loss = 0.
							val_logits_list = []
							val_prob_list = []
							val_true_list = []

							kf_val = get_minibatches_idx(len(val), len(val), shuffle=False)
							for _, val_index in kf_val:
								val_sents = [val[t] for t in val_index]
								val_labels = [val_lab[t] for t in val_index]
								val_labels = np.array(val_labels)
								val_labels = val_labels.reshape((len(val_labels), opt.num_class))
								x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
								val_prob, val_logits, val_loss_ = sess.run([prob_, logits_, loss_],
																		   feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
																					  y_: val_labels, keep_prob: 1.0, class_penalty_:0.0})
								val_loss += val_loss_ * len(val_index)

								val_logits_list += val_logits.tolist()
								val_prob_list += val_prob.tolist()
								val_true_list += val_labels.tolist()


							val_loss_list.append(val_loss/len(val))

							val_logits_array = np.asarray(val_logits_list)
							val_prob_array = np.asarray(val_prob_list)
							val_true_array = np.asarray(val_true_list)
							val_auc_list = []

							for i in range(opt.num_class):
								val_auc = roc_auc_score(y_true = val_true_array[:,i], y_score = val_logits_array[:,i],)
								val_auc_list.append(val_auc)

							val_auc_mean = np.mean(val_auc_list)
							validation_list.append(val_auc_mean)
							print("--  Validation AUC Mean %f " % val_auc_mean)

							if val_auc_mean > max_val_auc_mean:
								max_val_auc_mean = val_auc_mean
								val_auc_mean_list.append(val_auc_mean)
								val_auc_lists.append(val_auc_list)
								save_path = saver.save(sess, opt.save_path)
								print('Max Validation AUC: '+str(max_val_auc_mean))
								print ('Model saved...')
								updated=True
								#last layer
								covariance_phi=train_last_layer
								covariance_sigma=train_prob_array
							else:
								pass
			except KeyboardInterrupt:
				print('Training interupted')
				print("Max VAL AUC Mean %f " % max_val_auc_mean)
	if init_train:
		return max_val_auc_mean, [validation_list], covariance_matrix(covariance_sigma, covariance_phi)
	else:
		if not updated:
			return max_val_auc_mean, None
		else:
			cov_matrix=covariance_matrix(covariance_sigma, covariance_phi)
			sigma, phi=input_determinants(det_data, opt)
			determinants=get_determinants(cov_matrix, phi, sigma)
			print ('determinants gotten.....')
			return max_val_auc_mean, determinants
