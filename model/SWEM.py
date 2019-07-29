# CREATED 07/15/2018 BY MATIAS BENITEZ
# DESCRIPTION: Class to implement SWEM model and evaluate its performance




#import all the required packages
###########################################
from keras import *
import pandas as pd
import pickle
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import average
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
######################################################


#Class required to take average-pooling (it is used when the model is defined)
########################################################
class MeanPool(Layer):
  def __init__(self, **kwargs):
      self.supports_masking = True
      super(MeanPool, self).__init__(**kwargs)

  def compute_mask(self, input, input_mask=None):
      # do not pass the mask to the next layers
      return None

  def call(self, x, mask=None):
      if mask is not None:
          # mask (batch, time)
          mask = K.cast(mask, K.floatx())
          # mask (batch, x_dim, time)
          mask = K.repeat(mask, x.shape[-1])
          # mask (batch, time, x_dim)
          mask = tf.transpose(mask, [0,2,1])
          x = x * mask
      return K.sum(x, axis=1) / K.sum(mask, axis=1)
  def compute_output_shape(self, input_shape):
      # remove temporal dimension
      return (input_shape[0], input_shape[2])
###################################################

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
######################################################################



##############################################
############  CLASS TO IMPLEMENT SWEM  #######
##############################################
class model(object):
    def __init__(self, word2id, embeddings, input_length, emb_size, output_size, trainable, layers=1, units=200, dropout=0.2):
        self.clf=Sequential()
        self.clf.add(Embedding(len(word2id), emb_size, weights=[embeddings], input_length=input_length, mask_zero=True, trainable=trainable))
        self.clf.add(MeanPool())
        self.clf.add(BatchNormalization())
        for i in range (layers):
            self.clf.add(Dense(units, activation='relu', kernel_initializer='uniform'))
            self.clf.add(Dropout(dropout))
            self.clf.add(BatchNormalization())
        self.clf.add(Dense(output_size, activation='sigmoid', kernel_initializer='uniform'))
        self.clf.compile(optimizer=SGD(), loss='binary_crossentropy')

#Initialize variables of the model
#Required Inputs: 
#word_vectors= matrix created by load_embedding_vectors_word2vec_gensim()
#word2id=Dictionary that maps word to id numbers
#id2word= DIctionary that maps id numbers to words
#input_length= Length of notes
#######################################################
class swem(object):
    def __init__(self, word_vectors, word2id, id2word, input_length, emb_size=300, output_size=3, trainable=True, layers=1, units=200, dropout=0.2, batch_size=50, epochs=200):
        #Defined variables
        self.output_size=output_size
        self.embeddings=word_vectors
        self.word2id=word2id
        self.id2word=id2word
        #self.embeddings[self.word2id["<PAD/>"]]=np.zeros((300,))
        self.input_length=input_length
        self.batch_size=batch_size
        self.epoch=epochs
        self.emb_size=emb_size
        self.trainable=trainable
        self.layers=layers
        self.units=units
        self.dropout=dropout
        #Defined model
        self.model_class= model(word2id, self.embeddings, input_length, emb_size, output_size,self.trainable, self.layers, self.units, dropout=self.dropout)
        self.model=self.model_class.clf
        
####################################################################


#Method to train model
#Inputs:
#x_train, y_train= data, labels
#early_stop: If true, it stops when validation loss goes up
#show_graph: if TRue, then it shows validation and training losses
##################################
    def fit(self, x_train, y_train, batch_size=400, epochs=300, early_stop=False, show_graph=False, class_weight=None):
        early_stop_= EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
        #train in label data
        if early_stop:
             history= self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.1, verbose=0, callbacks=[early_stop_], class_weight=class_weight)
        else:
             history= self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.1, verbose=0)
        #train in label data
        if show_graph:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
##############################################

#Method to predict probabilities
#input: independent variables (reports)
#################################################
    def predict(self, x_test):
        return self.model.predict(x_test)
#################################################


#Method to get the results of the average-pooling layer (Mean_pool)
#Input: dependent variable (reports)
#Output: Vector of dimension = emb_size representing the report
#######################################################
    def get_mean_emb(self, x):
        inp=self.model.input
        functor=K.function([inp], [self.model.layers[1].output])
        test=np.array(x)
        return np.array(functor([test])[0])
##########################################################


#Method to compute 10-folds cross-validation
#output: auc cross-validation for all the outputs and a additional result for the average (it is the last number)
####################################################################
    def cross_validation(self, x, y,  batch_size=400, epochs=300, early_stop=False, class_weight=None, random_state=2018):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score
        AUC=[]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
        matrix=np.zeros(y.shape)
        TEST_INDEX=[]
            
        for train_index, test_index in skf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            self.model_class= model(self.word2id, self.embeddings, self.input_length, self.emb_size, self.output_size,self.trainable, self.layers, self.units)
            self.model=self.model_class.clf
            self.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, early_stop=early_stop, class_weight=class_weight)
            y_pred=self.predict(X_test)
            matrix[test_index]=np.reshape(y_pred, (len(y_pred),))
            auc_=[]
            #for i in range(self.output_size):
            auc_.append(roc_auc_score(Y_test, y_pred))
            AUC.append(auc_)
            TEST_INDEX.append(test_index)
        cross=[]
        for x in range(self.output_size):
            summ=[]
            for i in range(10):
                summ.append(AUC[i][x])
            print (np.mean(summ), np.std(summ))
            cross.append(np.mean(summ))
        print ('Average AUC: '+str(np.mean(cross)))
        return (AUC, matrix, TEST_INDEX)
#############################################################        
        
 #MEthod to plot the average ROC of the cross validation and also compute the auc
#Inputs: independent and dependent variables, and title of the figure
###########################################################
    def cross_roc(self, x, y,title, early_stop=False, random_state=2018):


        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
        mean=[[],[],[]]
        mean_tpr = [0.0, 0.0, 0.0]
        mean_fpr = [np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
        all_tpr = []
        auc_=[[],[],[]]
        for train_index, test_index in skf.split(x, y[:,0]):
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            self.model_class= model(self.word2id, self.embeddings, self.input_length, self.emb_size, self.output_size,self.trainable)
            self.model=self.model_class.clf
            self.fit(X_train, Y_train, early_stop=early_stop)
            y_pred=self.predict(X_test)
            for i in range(3):
                fpr, tpr, thresholds = roc_curve(Y_test[:,i], y_pred[:, i])
                mean_tpr[i] += interp(mean_fpr[i], fpr, tpr)
                mean_tpr[i][0] = 0.0
                roc_auc = roc_auc_score(Y_test[:,i], y_pred[:,i])
                auc_[i].append(roc_auc)

        mean_tpr[0] /= 10
        mean_tpr[0][-1] = 1.0
        mean_auc = auc(mean_fpr[0], mean_tpr[0])
        plt.plot(mean_fpr[0], mean_tpr[0], 'k--',
         label='Mean ROC Lung(AUC = %0.2f)' % np.mean(auc_[0]), lw=2, color='black')

        mean_tpr[1] /= 10
        mean_tpr[1][-1] = 1.0
        mean_auc = auc(mean_fpr[1], mean_tpr[1])
        plt.plot(mean_fpr[1], mean_tpr[1],'k--',
         label='Mean ROC Liver(AUC = %0.2f)' % np.mean(auc_[1]), lw=2, color='blue')

        mean_tpr[2] /= 10
        mean_tpr[2][-1] = 1.0
        mean_auc = auc(mean_fpr[2], mean_tpr[2])
        plt.plot(mean_fpr[2], mean_tpr[2], 'k--',
         label='Mean ROC Kidney(AUC = %0.2f)' % np.mean(auc_[2]), lw=2, color='orange')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
    def wrong_cross_fit(self, x, y,  batch_size=400, epochs=300, early_stop=False, class_weight=None):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score
        self.model_class= model(self.word2id, self.embeddings, self.input_length, self.emb_size, self.output_size,self.trainable, self.layers, self.units)
        self.model=self.model_class.clf
        AUC=[]
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        matrix=np.zeros(y.shape)
        for train_index, test_index in skf.split(x, y[:,0]):
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            self.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, early_stop=early_stop, class_weight=class_weight)
            y_pred=self.predict(X_test)
            matrix[test_index,:]=y_pred
        return matrix
            
            #############################

#plot calibration curves
def calibration_graph(y, probs, title):
    r=calibration_curve(y[:,0], probs[:,0], n_bins=10)
    plt.plot(r[1],r[0], color='r', label='Lung')
    r=calibration_curve(y[:,1], probs[:,1], n_bins=10)
    plt.plot(r[1],r[0], color='b', label='Liver')

    r=calibration_curve(y[:,2], probs[:,2], n_bins=10)
    plt.plot(r[1],r[0], color='g', label='Kidney')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

    
    
#plot probability histogram
             
def probability_dist(probs, bins=20):
    names=['Lung','Liver','Kidney']
    fig = plt.figure(figsize=(7,16))
    for k in range(probs.shape[1]):
        plt.subplot(3,1,k+1)
        plt.hist(probs[:,k], bins=bins)
        plt.title(names[k])
        plt.xlabel('Probability')
        plt.ylabel('Notes')    
    plt.show()

    
    
    
    