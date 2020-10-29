import os, sys
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from note_labeler.model.utils import load_embedding_vectors_word2vec_gensim as load_word2vec_matias
from math import floor
from sklearn.metrics import roc_auc_score
import random
#import matplotlib.pyplot as plt
import pandas as pd
import gc
import pickle
from note_labeler.model.model import *
from note_labeler.model.utils import *

#binary cross-entropy
def get_entropy(p):
	#p=np.array(prob)
	entropy=(-p * np.log2(p)-(1-p)*np.log2(1-p))
	entropy=np.nan_to_num(entropy)
	return entropy

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
# comment notation
	# b: batch size, s: sequence length, e: embedding dim, c : num of class
	x_emb, W_norm = embedding(x, opt)  #  b * s * e
	y_pos = tf.argmax(y, -1)
	y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
	W_class_tran = tf.transpose(W_class, [1,0]) # e * c
	x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
	H_enc = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
	#H_enc = tf.squeeze(H_enc)
	logits, last_layer1 = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)	# b * c
	logits_class, last_layer2 = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
	# prob = tf.nn.softmax(logits)
	prob = tf.nn.sigmoid(logits)
	class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class],
						  dtype=tf.float32, value=np.identity(opt.num_class),)
	# correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)) + class_penalty * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=class_y, logits=logits_class))

	global_step = tf.Variable(0, trainable=False)
	train_op = layers.optimize_loss(
		loss,
		global_step=global_step,
		optimizer=opt.optimizer,
		learning_rate=opt.lr)
	return prob, logits, loss, train_op, W_norm, global_step, H_enc, last_layer1
#input: reports
#output: sigma and phi
def input_determinants(data, opt):
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
	prob_list=[]
	last_layer=[]
	with tf.Session(config=config) as sess:
			train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
			test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
			saver = tf.train.Saver()

			if True:
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
			ke=entropy_batch(data, 300)
			for batch in ke:
				x_batch, x_batch_mask = prepare_data_for_emb(batch, opt)
				cov_prob, cov_last_layer= sess.run([prob_, last_layer_],feed_dict={x_: x_batch, x_mask_: x_batch_mask, keep_prob: 1.0,class_penalty_:0.0})
				prob_list+=cov_prob.tolist()
				last_layer.append(cov_last_layer)
			prob=np.array(prob_list)
			last_layer=np.concatenate(last_layer)
	return prob, last_layer

#input: sigma and phi
#output: covariance matrix
def covariance_matrix(sigma, phi):
	t=np.reshape(sigma, (sigma.shape[0],))
	c=t*(1-t)
	B=np.diag(c)
	#print(phi.shape)
	covar=np.matmul(np.matmul(np.transpose(phi),B), phi)+np.identity(phi.shape[1])
	return covar

#get determinants
def get_determinants(matrix, data_h, prob_data):
	prob_=prob_data.reshape((prob_data.shape[0],))
	sig=prob_*(1-prob_)
	det_covariance=np.linalg.det(matrix)
	def new_det(matrix, det_covar, phi, beta):
		inv_covar=np.linalg.solve(matrix,phi)
		new_determinant=det_covar*(1+beta*np.matmul(np.transpose(phi), inv_covar))
		return 1/new_determinant
	result=[]
	for phi, beta in zip(data_h, sig):
		det_cov=new_det(matrix,det_covariance, phi, beta)
		result.append(det_cov)
	return result
