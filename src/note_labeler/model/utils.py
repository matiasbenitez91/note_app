import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import sys
import os
import random
import gensim

def prepare_data_for_emb(seqs_x, opt):
    maxlen = opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token
    return x, x_mask

def restore_from_save(t_vars, sess, opt):
    save_keys = tensors_key_in_file(opt.save_path)
    #print(save_keys.keys())
    ss = set([var.name for var in t_vars])&set([s+":0" for s in save_keys.keys()])
    cc = {var.name:var for var in t_vars}
    ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])  # only restore variables with correct shape

    if opt.reuse_discrimination:
        ss2 = set([var.name[2:] for var in t_vars])&set([s+":0" for s in save_keys.keys()])
        cc2 = {var.name[2:][:-2]:var for var in t_vars if var.name[2:] in ss2 if var.get_shape() == save_keys[var.name[2:][:-2]]}
        for s_iter in ss_right_shape:
            cc2[s_iter[:-2]] = cc[s_iter]

        loader = tf.train.Saver(var_list=cc2)
        loader.restore(sess, opt.save_path)
        print("Loaded variables for discriminator:"+str(cc2.keys()))

    else:
        loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss_right_shape])
        loader.restore(sess, opt.save_path)
        print("Loading variables from '%s'." % opt.save_path)
        print("Loaded variables:"+str(ss_right_shape))

    return loader

def tensors_key_in_file(file_name):
    """Return tensors key in a checkpoint file.
    Args:
    file_name: Name of the checkpoint file.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        return reader.get_variable_to_shape_map()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        return None

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return zip(range(len(minibatches)), minibatches)


def get_balanced_batch_idx(n, labels, minibatch_size, num_class, num_iteration):
    class_idx = []
    for i in range(num_class):
        idx = [ix for ix in range(n) if labels[ix][i] == 1]
        class_idx.append(idx)

    num_draw = np.array([minibatch_size//num_class]*num_class)
    idx_draw_add1 = random.sample(range(num_class), minibatch_size%num_class)
    num_draw[idx_draw_add1] += 1

    draw_idx_iter = []
    for i in range(num_iteration):
        draw_idx = [random.sample(class_idx[i], num_draw[i]) for i in range(num_class)]
        draw_idx = np.array([y for x in draw_idx for y in x], dtype=np.int32)
        draw_idx_iter.append(draw_idx)

    return zip(range(num_iteration), draw_idx_iter)


def load_class_embedding(wordtoidx, opt):
    print("load class embedding")
    name_list = [ k.lower().split(' ') for k in opt.class_name]
    id_list = [ [ wordtoidx[i] for i in l] for l in name_list]
    value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
    value_mean = [ np.mean(l,0)  for l in value_list]
    return np.asarray(value_mean)
def clean_list(lista, wordtoix):
    return [x for x in lista if x in set(wordtoix.keys())]
def load_embb_new(wordtoidx, opt):
    print("load class embedding")
    name_list = [[ k.lower().split(' ') for k in l] for l in opt.class_name]
    id_list = [ [[ wordtoidx[i] for i in clean_list(l,wordtoidx)] for l in k] for k in name_list]
    value_list = [ [ [opt.W_emb[i] for i in l]    for l in k if l!=[]] for k in id_list]
    value_mean = [[ np.mean(l,0)  for l in k] for k in value_list]
    value= [ np.mean(l,0)  for l in value_mean]
    return np.asarray(value)

def entropy_batch(lis, size):
    final=[]
    t=list(range(0, len(lis), size))
    for i in range(len(t)-1):
        final.append(lis[t[i]:t[i+1]])
    final.append(lis[t[-1]:])
    return final

#funtion to create the embedding matrix
#input: word2vec file of embedding, word2id
#output: embedding matrix of the words in word2id, if word not in word2id then random number in range (-0.25,0.25) is assigned
#####################################################################
def load_embedding_vectors_word2vec_gensim(vocabulary, filename):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    vector_size = model.vector_size
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    word2vec_vocab = list(model.vocab.keys())
    count = 0
    mis_count = 0
    for word in vocabulary.keys():
        idx = vocabulary.get(word)
        if word in word2vec_vocab:
            embedding_vectors[idx] = model.wv[word]
            count += 1
        else:
            mis_count += 1
    print("num of vocab in word2vec: {}".format(count))
    print("num of vocab not in word2vec: {}".format(mis_count))
    return embedding_vectors

def inv_operation(A, v):
    return np.linalg.solve(A,v)
