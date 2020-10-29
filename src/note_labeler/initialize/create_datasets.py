import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import pickle
from random import sample
import logging
import gensim
import itertools
from collections import Counter
from note_labeler.initialize.utils import *
pd.set_option('display.max_colwidth', -1)

class RawDataset:
    def __init__(self, data_path, schema, label_encoder, artifacts_path, sample_size, encoding):
        logging.info("reading csv...")
        self.data=pd.read_csv(data_path, encoding=encoding)
        self.schema=schema
        self.label_encoder=label_encoder
        if sample_size is not None:
            sample_size=int(sample_size)
        self.sample_size=sample_size
        self.artifacts_path=artifacts_path
        self.encoding=encoding
        logging.info("preprocessing data...")
        self.preprocess()
        init_train=self.data.loc[~self.data["label"].isnull()].index.tolist()
        self.index_test=sample(init_train, int(len(init_train)/2))
        self.init_train=list(set(init_train).difference(self.index_test))

    def preprocess(self):
        self.data=self.data.rename(self.schema, axis="columns")
        if "label" not in self.data.columns:
            self.data["label"]=np.nan
        set_possible=set(list(self.label_encoder.keys()))
        if self.label_encoder is not None:
            self.data['label']=self.data.label.apply(lambda x: self.label_encoder[x]  if x in set_possible else np.nan)
        if self.sample_size is not None and self.sample_size<len(self.data):
            logging.info("sampling size {}".format(self.sample_size))
            self.data=self.data.sample(self.sample_size, random_state=123)
        self.data.reset_index(inplace=True)

    def gen_artifacts(self):
        logging.info("Cleaning artifacts...")
        #tokenize
        text_clean = [clean_str(l) for l in self.data['review']]
        #create list to construct vocabulary
        word_clean_count = Counter(list(itertools.chain.from_iterable(text_clean)))
            #create dict of unique words
        v_clean = [x for x, y in word_clean_count.items() if y >=5]
        v_clean_all = [x for x, y in word_clean_count.items()]
        logging.info("creating vocab...")
        #create vocabulary and inverse vocabulary. Words with frequency >5
        inv_vocabulary_clean ={}
        inv_vocabulary_clean[0] = 'END'
        inv_vocabulary_clean[1] = 'UNK'
        inv_vocabulary_clean[2] = '<PAD/>'
        vocabulary_clean ={}
        vocabulary_clean['END'] = 0
        vocabulary_clean['UNK'] = 1
        vocabulary_clean['<PAD/>'] = 2
        ix=3
        for v in v_clean_all:
            if v in v_clean:
                vocabulary_clean[v] = ix
                inv_vocabulary_clean[ix] = v
                ix +=1
        #convert list of words to list of id's
        clean_num = convert_word_to_ix_clean(text_clean, vocabulary_clean)
        #apply functions
        train_x_clean_num, train_x_clean, label_L=[clean_num[x] for x in self.init_train], [text_clean[x] for x in self.init_train], [self.data.label.tolist()[x] for x in self.init_train]
        test_x_clean_num, test_x_clean, label_test=[clean_num[x] for x in self.index_test], [text_clean[x] for x in self.index_test], [self.data.label.tolist()[x] for x in self.index_test]
        max_len_x=max([len(x) for x in train_x_clean_num])
        max_len_test=max([len(x) for x in test_x_clean_num])
        max_len_unlab=max([len(x) for x in clean_num])
        #export files
        if not os.path.exists(os.path.join(self.artifacts_path,'model_iteration')):
            os.makedirs(os.path.join(self.artifacts_path,'model_iteration'))
        with open(os.path.join(self.artifacts_path,'M0.pkl'), 'wb') as f:
            pickle.dump([train_x_clean_num, train_x_clean, label_L, vocabulary_clean, inv_vocabulary_clean, max_len_x], f, protocol=2)
        with open(os.path.join(self.artifacts_path,'test.pkl'), 'wb') as f:
            pickle.dump([test_x_clean_num, test_x_clean, label_test, max_len_test], f, protocol=2)
        with open(os.path.join(self.artifacts_path,'all_valid_reports.pkl'), 'wb') as f:
            pickle.dump([clean_num, text_clean,max_len_unlab], f, protocol=2)
        with open(os.path.join(self.artifacts_path,'model_iteration','train.pkl'), 'wb') as f:
            pickle.dump([train_x_clean_num, train_x_clean, label_L, vocabulary_clean, inv_vocabulary_clean, max_len_x], f, protocol=2)
        #word2vec
        model_clean = gensim.models.Word2Vec(text_clean, size=300, window=5, min_count=5, workers=4)
        model_clean.wv.save_word2vec_format(os.path.join(self.artifacts_path,'CTword2vec_clean'),os.path.join(self.artifacts_path,'CTvocab_clean'))

        self.data['additional_note']=float('nan')
        self.data['valid']=[valid(x, self.init_train, self.index_test) for x in list(self.data.index)]
        self.data['group']=[group(x, self.init_train, self.index_test) for x in list(self.data.index)]
        self.data['tokens_num']=[",".join([str(y) for y in x]) for x in clean_num]
        self.data["Report#"]=list(self.data.index)
        self.data.loc[self.data['valid'],'label']=float('nan')

        self.data[["review","label","additional_note","valid","group","tokens_num","Report#"]].to_csv(os.path.join(self.artifacts_path, 'database.csv'), encoding=self.encoding)
        #data.to_csv('model/random_sampling/database.csv', encoding="latin-1")
        pickle.dump([["good"],["bad"]],open(os.path.join(self.artifacts_path,"new_dict_updt.pkl"),"wb"), protocol=2)
