#coding=utf-8
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import numpy as np


UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO = '**GO**'

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()

word_freq = defaultdict(int)

#filename = 'dataset/snli_1.0_dev.jsonl'
#filename = 'dataset/fake_3val_dev_equal.txt'
#filename = 'dataset/testdata.txt'

embedding_path = 'glove.6B.100d.txt'

num_classes = 4

max_len1 = 20
max_len2 = 30
#add
def _generate_random_vector(size):
    """
    Generate a random vector from a uniform distribution between
    -0.1 and 0.1.
    """
    return np.random.uniform(-0.1, 0.1, size)
    
def loadGloVe(emb_path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    vocab = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    print('Loading embedding file...')
    with open(emb_path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            vocab.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)

    embeddings = np.array(vectors, dtype=np.float32)
    print('Loaded GloVe!')

    return vocab, embeddings

def load_embeddings(emb_path):
    wordlist, embeddings = loadGloVe(emb_path)
    mapping = zip(wordlist, range(3, len(wordlist) + 3))

    print('creating word dict...')
    # always map OOV words to 0
    wd = defaultdict(int, mapping)
    wd[UNKNOWN] = 0
    wd[PADDING] = 1
    wd[GO] = 2

    vector_size = embeddings.shape[1]
    extra = [_generate_random_vector(vector_size),
             _generate_random_vector(vector_size),
             _generate_random_vector(vector_size)]

    embeddings = np.append(extra, embeddings, 0)

    print('finished!')
    
    return wd, embeddings

#to here


def read_corpus(file_name):
    data_x = []
    data_y = []
    data_z = []
    useful_data = []
    
    with open(file_name, 'rb') as f:
        if file_name.endswith('.tsv') or file_name.endswith('.txt'):    
            for line in f:
                line = line.decode('utf-8').strip()
                present1, present2, prelabel = line.split('\t')
                
                senten1 = present1.lower()
                senten2 = present2.lower()
                
                if prelabel == '---':
                    continue
                #if prelabel == 'Can not judge':
                    #continue
                #if prelabel == '----':
                #   continue
                #if prelabel == 'Unrelated':
                    #continue
                body_sent = []
                sent1 = sent_tokenizer.tokenize(senten1)#articlebody
                for sent in sent1:
                    sents1 = word_tokenizer.tokenize(sent)
                    body_sent.append(sents1)

                sents2 = word_tokenizer.tokenize(senten2)#headline
                                           
                label = str(prelabel)
                data_x.append(body_sent)
                data_z.append(sents2)
                useful_data.append( (body_sent, sents2, label) )

        else:
            for line in f:
                review = json.loads(line)
                sent1 = sent_tokenizer.tokenize(review['sentence1'])
                
                body_sent = []
                for sent in sent1:
                    sents1 = word_tokenizer.tokenize(sent)
                    body_sent.append(sents1)

                sents2 = word_tokenizer.tokenize(review['sentence2'])

                label = str(review['gold_label'])
                
                useful_data.append( (body_sent, sents2, label) )
                
        return useful_data,data_x,data_z #x:body, z:headline


def create_dataset(pairs, word_dict, label_dict=None,
                   max_len1=None, max_len2=None):
    """
    Generate and return a RTEDataset object for storing the data in numpy format.

    :param pairs: list of tokenized tuples (sent1, sent2, label)
    :param word_dict: a dictionary mapping words to indices
    :param label_dict: a dictionary mapping labels to numbers. If None,
        labels are ignored.
    :param max_len1: the maximum length that arrays for sentence 1
        should have (i.e., time steps for an LSTM). If None, it
        is computed from the data.
    :param max_len2: same as max_len1 for sentence 2
    :return: RTEDataset
    """
    sentences1 = []
    sentences2 = []
    tokens1 = [pair[0] for pair in pairs]
    for sents1 in tokens1:
        sent1, sizes1 = _convert_body_to_indices(sents1, word_dict)
        sentences1.append(sent1.tolist())
        sent1 = []

    tokens2 = [pair[1] for pair in pairs]
    for sents2 in tokens2:
        sent2, sizes2 = _convert_headline_to_indices(sents2, word_dict)
        sentences2.append(sent2.tolist())
        sent2 = []

    label = [pair[2] for pair in pairs]
    #print(label)
    lab_data = []
    for lab in label:
        labels = [0] * num_classes
        if lab == 'True':
            labels[0] = 1
        if lab == 'False':
            labels[1] = 1
        if lab == 'Can not judge':
            labels[2] = 1
        if lab == 'Unrelated':
            labels[3] = 1
        #if lab == 'entailment':
            #labels[0] = 1
        #if lab == 'contradiction':
            #labels[1] = 1
        #if lab == 'neutral':
        #    labels[2] = 1
        #if lab == 'unrelated':
        #    labels[3] = 1     
        #else:
        #    continue
        lab_data.append(labels)
    
    #print(lab_data)
    print('data_created!')
    

    return sentences1, sentences2, lab_data
    
def _convert_body_to_indices(sentences, word_dict):
    sizes = np.array([len(sent) for sent in sentences])

    array = np.full((20,30), word_dict[PADDING], dtype=np.int32)

    for i, sent in enumerate(sentences):
        indices = [word_dict[token] for token in sent]
        if i < 20:
            for j in range(0,len(indices)):
                if j < 30:            
                    if indices[j] != None:
                        array[i][j] = indices[j]          
    return array, sizes
    
def _convert_headline_to_indices(sentences, word_dict):
    sizes = np.array([len(sent) for sent in sentences])
    """  
    array = np.full((1,30), word_dict[PADDING], dtype=np.int32)

    for i, sent in enumerate(sentences):
        indices = [word_dict[token] for token in sentences]
        if i < 1:
            for j in range(0,len(indices)):
                if j < 30:            
                    if indices[j] != None:
                        array[i][j] = indices[j]
            
    """
    array = np.full(30, word_dict[PADDING], dtype=np.int32)

    indices = [word_dict[token] for token in sentences]
    for j in range(0,len(indices)):
        if j < 30:            
            if indices[j] != None:
                array[j] = indices[j]     
  

    return array, sizes

    
def convert_labels(pairs, label_map):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :param label_map: dictionary mapping label strings to numbers
    :return: a numpy array
    """
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)
    
def create_label_dict(pairs):
    """
    Return a dictionary mapping the labels found in `pairs` to numbers
    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a dict
    """
    labels = set(pair[2] for pair in pairs)
    mapping = zip(labels, range(len(labels)))
    print('created_label_dict!')
    return dict(mapping)

#filename = 'dataset/testdata.txt'
#word_dict, embed = load_embeddings(embedding_path)
#data_pair = read_corpus(filename)
#label_dict = create_label_dict(data_pair)
#data_x, data_z, data_y = create_dataset(data_pair, word_dict,label_dict)

#print(data_y)
#print(data_z)
#print(vars(obj))
#print(obj.__dict__)
