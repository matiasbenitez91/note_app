# coding: utf-8

# #### CREATED BY MATIAS BENITEZ 10/02/20018
# #### CODE THAT TRAINS THE LEAM MODEL AND CREATES INPUTS FOR T-SNE.

# ### loading packages

# In[1]:


import os, sys, cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from SWEM import load_embedding_vectors_word2vec_gensim as load_word2vec_matias
from math import floor
from sklearn.metrics import roc_auc_score
import random
import matplotlib.pyplot as plt
import pandas as pd
import gc
import pickle
from model import *
from utils import *


# ### defining object

# In[2]:


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
        self.max_epochs = 300
        self.dropout = 0.2
        self.part_data = False
        self.portion = 1.0 
        self.save_path = "./save/"
        self.log_path = "./log/"
        self.print_freq = 100
        self.valid_freq = 10

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 30
        self.H_dis = 64


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


# ### define `emb_classifier()`

# In[3]:


def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # comment notation
    # b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  #  b * s * e
    #print (x_emb.shape, W_norm.shape)
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    print (x_emb.shape, x_mask.shape, W_class.shape)
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    #H_enc = tf.squeeze(H_enc)
    print(H_enc.shape)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # b * c
    logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    # prob = tf.nn.softmax(logits)
    prob = tf.nn.sigmoid(logits)
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class],
                          dtype=tf.float32, value=np.identity(opt.num_class),)
    # correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)) +            class_penalty * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=class_y, logits=logits_class))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return prob, logits, loss, train_op, W_norm, global_step, H_enc

def get_balance_val(y, length):
    prop=y.sum()/(float(len(y)))
    pos=np.reshape(np.argwhere(y==1), (-1,))
    neg=np.reshape(np.argwhere(y==0), (-1,))
    pos=np.random.choice(pos, size=int(prop*length), replace=False)
    neg=np.random.choice(neg, size=length-len(pos), replace=False)
    result=np.append(pos, neg, axis=0)
    np.random.shuffle(result)
    return list(result)


def get_balanced_batch(labels, minibatch_size):
    def get_balance(y, length, prop):
        pos=np.reshape(np.argwhere(y==1), (-1,))
        neg=np.reshape(np.argwhere(y==0), (-1,))
        pos=np.random.choice(pos, size=int(prop*length), replace=False)
        neg=np.random.choice(neg, size=length-len(pos), replace=False)
        result=np.append(pos, neg, axis=0)
        np.random.shuffle(result)
        return list(result)
    result=[]
    n=len(labels)
    iters=int(n/minibatch_size)
    y=np.array(labels)
    rest=[]
    all_=range(n)
    prop=labels.sum()/(float(len(labels)))
    #print prop
    for i in range(iters):
        idx=get_balance(y, minibatch_size, prop)
        rest+=[all_[t] for t in idx]
        all_=[k for k in all_ if k not in set(rest)]
        y=np.array(labels)[all_]
        result.append((i, idx))
    result.append((i+1, all_))
    return result


# In[9]:


#RUN THE MODEL
#EVERY TIME THE CODE BELOW IS RUN, RUN THIS CHUNK. IT RESETS THE MODEL AND AVOIDS TO BE FITTING NEW DATA IN AN EXISTING MODEL
#DEFINE THE MODEL
def get_auc_test(train, train_lab,val, val_lab, test, test_lab,opt):
    final_list=[]
    for xxxxx in range(10):
        opt.fix_emb=True
        opt.H_dis=150
        tf.reset_default_graph()
        pred_batch_size = 80
        with tf.device('/gpu:0'):
            x_ = tf.placeholder(tf.int32, shape=[None, opt.maxlen])
            x_mask_ = tf.placeholder(tf.float32, shape=[None, opt.maxlen])
            keep_prob = tf.placeholder(tf.float32)
            y_ = tf.placeholder(tf.float32, shape=[None, opt.num_class])
            class_penalty_ = tf.placeholder(tf.float32, shape=())    
            prob_, logits_, loss_, train_op, W_norm_, global_step, H_enc_ = emb_classifier(
                x_, x_mask_, y_, keep_prob, opt, class_penalty_)


        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
        config.gpu_options.allow_growth = True
        np.set_printoptions(precision=3)
        np.set_printoptions(threshold=np.inf)
        saver = tf.train.Saver()



        uidx = 0
        max_val_auc_mean = 0.
        max_test_auc_mean = 0.
        test_auc_mean_list = []
        val_auc_mean_list = []
        test_auc_lists = []
        val_auc_lists = []

        with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
                test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
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
                    opt.max_epochs = 320
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
                                    train_prob, train_logits = sess.run([prob_, logits_],
                                                                        feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, 
                                                                                   y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

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

                                #print("Iteration %d: Training Loss %f " % (uidx, train_loss))
                                #print("--  Train AUC Mean %f " % train_auc_mean)


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
                                print("--  Validation AUC Mean %f " % val_auc_mean)

                                if val_auc_mean > max_val_auc_mean:
                                    max_val_auc_mean = val_auc_mean
                                    val_auc_mean_list.append(val_auc_mean)
                                    val_auc_lists.append(val_auc_list)

                                    test_logits_list = []
                                    test_prob_list = []
                                    test_true_list = []

                                    kf_test = get_minibatches_idx(len(test), len(test), shuffle=False)
                                    for _, test_index in kf_test:
                                        test_sents = [test[t] for t in test_index]
                                        test_labels = [test_lab[t] for t in test_index]
                                        test_labels = np.array(test_labels)
                                        test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                                        x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)
                                        test_prob, test_logits = sess.run([prob_, logits_],
                                                                          feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask, 
                                                                                     y_: test_labels, keep_prob: 1.0,class_penalty_:0.0})

                                        test_logits_list += test_logits.tolist()
                                        test_prob_list += test_prob.tolist()
                                        test_true_list += test_labels.tolist()

                                    test_logits_array = np.asarray(test_logits_list)
                                    test_prob_array = np.asarray(test_prob_list)
                                    test_true_array = np.asarray(test_true_list)
                                    test_auc_list = []

                                    for i in range(opt.num_class):
                                        test_auc = roc_auc_score(y_true = test_true_array[:,i], y_score= test_logits_array[:,i],)
                                        test_auc_list.append(test_auc)

                                    test_auc_mean = np.mean(test_auc_list)
                                    test_auc_mean_list.append(test_auc_mean)
                                    test_auc_lists.append(test_auc_list)
                                    print("Test AUC %s" % test_auc_list)
                                    print("--  Test AUC Mean %f " % test_auc_mean)
                                    save_path = saver.save(sess, opt.save_path)
                                    print("Model saved in path: %s" % save_path)
                    print("Max Test AUC Mean %f " % max_test_auc_mean)

                except KeyboardInterrupt:
                    print('Training interupted')
                    print("Max Test AUC Mean %f " % max_test_auc_mean)
                    print ('Last test auc mean %f' % test_auc_mean)

#        final_list.append(test_auc_mean)
#    return final_list
	
	
	
def main():
	# ### loading inputs

	# In[4]:


	labels=pickle.load(open('new_dict_updt.p', 'rb'))


	# In[5]:


	#import files created by Guoyin
	with open('random_sampling/M0.p', 'rb') as f:
		data=pickle.load(f)
	#dictionary token to id
	word2id=data[3]
	id2word=data[4]
	embeddings=load_word2vec_matias(word2id, 'random_sampling/CTword2vec_clean')
	embeddings_50=load_word2vec_matias(word2id, 'random_sampling/CTword2vec_clean_50')
	x_full=data[0]
	#x=np.array(x)
	#print x.shape
	#labels for lung, liver and kidney
	y=data[2]
	class_weight={0:(1.0/np.sum(y[:,0])), 1:(1.0/np.sum(y[:,1])), 2:(1.0/np.sum(y[:,2]))}
	#import files created by 'create_datasets.ipynb'
	with open('random_sampling/M1.p', 'rb') as f:
		review1=pickle.load(f)
	with open('random_sampling/M2.p', 'rb') as f:
		review2=pickle.load(f)
	with open('random_sampling/M3.p', 'rb') as f:
		review3=pickle.load(f)
	#add data from the first round of reviews
	y1=np.append(y,review1[2], axis=0)
	#add data from the first round of reviews
	y2=np.append(y1,review2[2], axis=0)
	y3=np.append(y2,review3[2], axis=0)
	with open('random_sampling/test.p', 'rb') as f:
		test=pickle.load(f)
		


	# In[24]:


	len(review1[0][0])


	# In[6]:


	#emb=gensim.models.KeyedVectors.load_word2vec_format('batch_by_batch/CTword2vec_clean')
	noteix_0=x_full
	noteix_1=noteix_0+review1[0]
	noteix_2=noteix_1+review2[0]
	noteix_3=noteix_2+review3[0]
	noteix_test=test[0]
	y_test=test[2]
	wordtoix=word2id
	ixtoword=id2word


	max_test=test[-1]
	max_len_0=max(data[-1], max_test)
	max_len_1=max(max(max_len_0, review1[-1]), max_test)
	max_len_2=max(max(max_len_1, review2[-1]), max_test)
	max_len_3=max(max(max_len_2, review3[-1]), max_test)



	# ### `main()`

	# In[12]:


	#DEFINE THE NUMBER OF ORGANS THE CLASSIFIER WILL HAVE IN THE opt.num_class variable.
	#LABELS[3] CONTAINS GENERAL TERMS
	#LABELS 0, 1, 2 CONTAIN SPECIFIC ORGAN TERMS. FOR EXAMPLE labels[0][0] contains general lung terms and labels[0][1] specific organ positive terms
	#In order to change the model specification change opt.num_class and the organs you include in opt.class_name. NUmber of organs include in this list should be consistent with the number in opt.num+class
	from utils import *

	lung=labels[0][0]+labels[0][1]#+labels[3][0]
	liver=labels[1][0]+labels[1][1]#+labels[3][0]
	kidney=labels[2][0]+labels[2][1]#+labels[3][0]

	opt = Options()

	opt.num_class = 1
	opt.class_name = [liver]

	opt.n_words = len(ixtoword)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)
	opt.W_emb = np.float32(embeddings)
	opt.W_class_emb = load_embb_new(wordtoix, opt)
	#print(dict(opt))
	print('Total words: %d' % opt.n_words)

	opt.maxlen=max_len_0
	opt.emb_size=300
	#Notelb needs to be specified. 0 lung, 1 liver, 2 kidney
	#notelb=y[:,0]
	valIdx=range(100)#get_balance_val(y1, 30)
	train_lab = y[:,0]
	val_lab = y_test[:100,0]
	test_lab = y_test[100:,0]

	train_lab_= np.array(train_lab, dtype='float32')
	val_lab = np.array(val_lab, dtype='float32')
	test_lab = np.array(test_lab, dtype='float32')  


	test=noteix_test[100:]
	train=noteix_0
	val=noteix_test[:100]
	opt.save_path="saved_models/lungs/init_model.ckpt"
	get_auc_test(train, train_lab,val, val_lab, test, test_lab,opt)

    
    
if __name__=='__main__':
    main()
	# In[7]:
