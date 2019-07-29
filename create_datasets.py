import pandas as pd
import nltk
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
from random import sample 


def main():
    #import data
    data=pd.read_csv("imdb_master.csv", encoding="latin-1")
    
    #set labels and subset
    data=data.loc[data.label.apply(lambda x: x=="neg" or x=="pos"),:]
    data['label']=data.label.apply(lambda x: 1 if x=="pos" else 0)
    data=data.sample(5000, random_state=123)
    data.reset_index(inplace=True)

    #sample by index
    index_test=sample(range(5000),300)
    index_train=list(set(range(5000)).difference(set(index_test)))
    init_train=sample(index_train, 100)
    
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),\.!?]", " ", string)  
        string = re.sub(r"[^A-Za-z]", " ", string)  # remove numbers
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"e\.g\.,", " ", string) 
        string = re.sub(r"a\.k\.a\.", " ", string) 
        string = re.sub(r"i\.e\.,", " ", string) 
        string = re.sub(r"i\.e\.", " ", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"\'", "", string) 
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string)
        string = re.sub(r"br", "", string)
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " ( ", string) 
        string = re.sub(r"\)", " ) ", string) 
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " . ", string)  
        string = re.sub(r"\s{2,}", " ", string) 
        string = re.sub(r"u\.s\.", " us ", string)
        return string.strip().lower().split()
    #tokenize
    text_clean = [clean_str(l) for l in data['review']]
    #creat dicts
    import itertools
    from collections import Counter
    #create list to construct vocabulary
    word_clean_count = Counter(list(itertools.chain.from_iterable(text_clean)))
        #create dict of unique words
    v_clean = [x for x, y in word_clean_count.items() if y >=5]
    v_clean_all = [x for x, y in word_clean_count.items()]
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
    #function to convert list of words to list of id's
    def convert_word_to_ix_clean(data):
        result = []
        for sent in data:
            temp = []
            for w in sent:
                if w in vocabulary_clean:
                    temp.append(vocabulary_clean.get(w,1))
                else:
                    temp.append(1)
            temp.append(0)
            result.append(temp)
        return result
    #convert list of words to list of id's
    clean_num = convert_word_to_ix_clean(text_clean)
    #apply functions
    train_x_clean_num, train_x_clean, label_L=[clean_num[x] for x in init_train], [text_clean[x] for x in init_train], [data.label.tolist()[x] for x in init_train]
    test_x_clean_num, test_x_clean, label_test=[clean_num[x] for x in index_test], [text_clean[x] for x in index_test], [data.label.tolist()[x] for x in index_test]
    max_len_x=max([len(x) for x in train_x_clean_num])
    max_len_test=max([len(x) for x in test_x_clean_num])
    max_len_unlab=max([len(x) for x in clean_num])
    #export files
    import pickle
    with open('model/random_sampling/M0.p', 'wb') as f:
        pickle.dump([train_x_clean_num, train_x_clean, label_L, vocabulary_clean, inv_vocabulary_clean, max_len_x], f, protocol=2)  
    with open('model/random_sampling/test.p', 'wb') as f:
        pickle.dump([test_x_clean_num, test_x_clean, label_test, max_len_test], f, protocol=2) 
    with open('model/random_sampling/all_valid_reports.p', 'wb') as f:
        pickle.dump([clean_num, text_clean,max_len_unlab], f, protocol=2)
    with open('model/kidney/train.p', 'wb') as f:
        pickle.dump([train_x_clean_num, train_x_clean, label_L, vocabulary_clean, inv_vocabulary_clean, max_len_x], f, protocol=2)   
        
    #word2vec
    import gensim
    model_clean = gensim.models.Word2Vec(text_clean, size=300, window=5, min_count=5, workers=4)
    model_clean.wv.save_word2vec_format('model/random_sampling/CTword2vec_clean','model/random_sampling/CTvocab_clean')
    
    #create dataset
    def valid(report):
        if report not in set(index_test) and report not in set(init_train):
            return True
        else:
            return False
    def group(report):
        if report in set(index_test):
            return 'test'
        elif report in set(init_train):
            return 'control'
    data['note_clinician']=float('NaN')
    data['valid']=[valid(x) for x in list(data.index)]
    data['group']=[group(x) for x in list(data.index)]    
    data.drop(['type', 'index','Unnamed: 0',"file"], axis=1, inplace=True)
    data['tokens_num']=clean_num
    data["Report#"]=list(data.index)
    data.to_csv('database.csv', encoding="latin-1")
    data.to_csv('model/random_sampling/database.csv', encoding="latin-1")
    pickle.dump([["good"],["bad"]],open("model/new_dict_updt.p","wb"), protocol=2)
    
if __name__=="__main__":
    run()