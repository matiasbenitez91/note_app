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
from note_labeler.model import train_model
from note_labeler.model.utils import *
from note_labeler.model.AL_utils import *
import ast

def iterate_model(artifacts_path, encoding="utf-8"):
	labels=pickle.load(open(os.path.join(artifacts_path,'new_dict_updt.pkl'), 'rb'))
	max_auc=pd.read_csv(os.path.join(artifacts_path,"model_iteration",'max_auc.csv'))
	max_auc_val=max_auc['AUC_val'].iloc[0]
	#import files created by Guoyin
	with open(os.path.join(artifacts_path,"model_iteration",'train.pkl'), 'rb') as f:
		data=pickle.load(f)
	#dictionary token to id
	word2id=data[3]
	id2word=data[4]
	embeddings=load_word2vec_matias(word2id, os.path.join(artifacts_path,'CTword2vec_clean'))
	x_full=data[0]
	y=data[2]
	#class_weight={0:(1.0/np.sum(y[:,0])), 1:(1.0/np.sum(y[:,1])), 2:(1.0/np.sum(y[:,2]))}
	#import files
	with open(os.path.join(artifacts_path,'test.pkl'), 'rb') as f:
		test=pickle.load(f)
	noteix_0=x_full
	noteix_test=test[0]
	y_test=test[2]
	wordtoix=word2id
	ixtoword=id2word
	max_test=test[-1]
	max_len_0=max(data[-1], max_test)
	kidney=labels[0][0]

	opt = train_model.Options()

	opt.num_class = 1
	opt.class_name = [kidney]

	opt.n_words = len(ixtoword)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)
	opt.W_emb = np.float32(embeddings)
	opt.W_class_emb = load_embb_new(wordtoix, opt)
	print('Total words: %d' % opt.n_words)

	opt.maxlen=max_len_0
	opt.emb_size=300
	opt.H_dis=150
	opt.restore=True
	opt.max_epochs=25

	train=noteix_0
	val=noteix_test
	train_lab=y[:]
	val_lab=y_test
	database=pd.read_csv(os.path.join(artifacts_path,'database.csv'), encoding=encoding)
	det_data=database.loc[database['valid'],'tokens_num'].values
	det_data=[ast.literal_eval(x) for x in det_data]
	return train_model.get_model(train, train_lab, val, val_lab, opt, max_auc_val, det_data)
