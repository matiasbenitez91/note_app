import re

#create dataset
def valid(report, init_train, index_test):
    if report not in set(index_test) and report not in set(init_train):
        return True
    else:
        return False
def group(report, init_train, index_test):
    if report in set(index_test):
        return 'test'
    elif report in set(init_train):
        return 'control'

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


#function to convert list of words to list of id's
def convert_word_to_ix_clean(data, vocabulary_clean):
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
