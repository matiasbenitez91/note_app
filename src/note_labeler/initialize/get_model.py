"""
File to train the model on the initial data and create the initial database.csv

"""
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
from time import gmtime, strftime
import ast
pd.set_option('display.max_colwidth', -1)

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
		self.max_epochs = 30
		self.dropout = 0.2
		self.part_data = False
		self.portion = 1.0
		self.save_path = "./data/model_iteration/"
		self.log_path = "./log/"
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

def init_model(artifacts_path, encoding='utf-8'):
	#import label embeddings
	labels=pickle.load(open(os.path.join(artifacts_path,'new_dict_updt.pkl'), 'rb'))
	#import dataset
	with open(os.path.join(artifacts_path,"model_iteration",'train.pkl'), 'rb') as f:
		data=pickle.load(f)
	#dictionary token to id
	word2id=data[3]
	id2word=data[4]
	embeddings=load_word2vec_matias(word2id, os.path.join(artifacts_path,'CTword2vec_clean'))
	x_full=data[0]
	y=data[2]
	#import files created by 'create_datasets.py'
	with open(os.path.join(artifacts_path,'test.pkl'), 'rb') as f:
		test=pickle.load(f)
	#load inputs for get_model()
	noteix_0=x_full
	noteix_test=test[0]
	y_test=test[2]
	#dictionaries
	wordtoix=word2id
	ixtoword=id2word
	max_test=test[-1]
	max_len_0=max(data[-1], max_test)
	kidney=labels[0][0]
	#set hyper-parameters
	opt = Options()
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
	opt.max_epochs=300#opt.restore=True
	#set inputs
	train=noteix_0
	val=noteix_test
	train_lab=y
	val_lab=y_test
	database=pd.read_csv(os.path.join(artifacts_path,'database.csv'), encoding=encoding)
	det_data_=database.loc[database['valid'],'tokens_num']
	det_data=[[int(s) for s in m.split(",")] for m in det_data_.tolist()]
	max_auc, list_aucs, covariance=train_model.get_model(train, train_lab, val, val_lab, opt)
	#edit existing files
	with open('initial_auc.pkl','wb') as f:
		pickle.dump(list_aucs, f)
	df=pd.DataFrame({'AUC_val':[max_auc]})
	#compute new determinants for AL
	sigma, phi=input_determinants(det_data, opt)
	determinants=get_determinants(covariance, phi, sigma)
	database.loc[:,'determinants']=float('inf')
	database.loc[database['valid'],'determinants']=determinants
	database.to_csv(os.path.join(artifacts_path,'database.csv'), encoding=encoding,index=False)
	df.to_csv(os.path.join(artifacts_path,"model_iteration",'max_auc.csv'), index=False)
	#add performance to the records
	k=pd.DataFrame({'Report#':['initial'],'AUC_val':[max_auc],"initial_time":float('nan'),'final_time':[strftime("%Y-%m-%d %H:%M:%S", gmtime())]})
	#dt=pd.read_csv(os.path.join(artifacts_path,"output.csv"), index=False)
	#df=pd.DataFrame(columns=dt.columns.tolist())
	#df.to_csv(os.path.join(artifacts_path, "output.csv", index=False)
	k.to_csv(os.path.join(artifacts_path,"model_iteration",'auc.csv'), index=False)
