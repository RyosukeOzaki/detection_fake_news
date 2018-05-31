#coding=utf8
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

from collections import defaultdict
import numpy as np
from scipy.sparse import hstack

config = tf.ConfigProto(allow_soft_placement=True,device_count={'GPU': 1})
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config=config)



#add
UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO = '**GO**'


emb_path = 'glove.6B.100d.txt'

#add
#def loadGloVe(filename):
#    print('Loading embedding file...')
#    vocab = []
#    embd = []
#    file = open(filename,'r')
#    for line in file.readlines():
#        row = line.strip().split(' ')
#        vocab.append(row[0])#word
#        embd.append(row[1:])#vector
#    print('Loaded GloVe!')
#    file.close()
#    return vocab,embd

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


vocab, embd = load_embeddings(emb_path)
embeddings = np.asarray(embd)
vocab_size = len(vocab)
embedding_dim = len(embd[0])

print('vocab size :',vocab_size)
print('embedding dim :',embedding_dim)


#to here


def length(sequences):
#シーケンス中の各要素の長さを返す
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, num_classes, embedding_size=100, hidden_size=50):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

            # x:[batch_size, 文の数, 単語数] 足りない部分は0で補う
            # y:[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
            
            self.input_z = tf.placeholder(tf.int32, [None, None], name='input_z')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            

        word_embedded, input_x = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        doc_vec = self.doc2vec(sent_vec)
        
        head_word_embedded = self.headline_word2vec()
        #headline_vec = self.headline_sent2vec(head_word_embedded)
        headline_article_vec = K.concatenate([head_word_embedded, doc_vec], axis=1)
        #headline_article_vec = tf.concat([head_word_embedded, doc_vec],1)
        news_vec = self.news2vec(headline_article_vec)
        #out = self.classifer(doc_vec)
        out = self.classifer(news_vec)

        self.out = out
        
        self.input_x = input_x
        #self.doc_vec = doc_vec
        #self.news_vec = news_vec
        #self.headline_article_vec = headline_article_vec
        #self.sent_vec=sent_vec


    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            
            #add
            #embedding_mat = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]), trainable=True, name="pre_emb")
            embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
            embedding_init = embedding_mat.assign(embedding_placeholder)
            sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings})
            # to here
            
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            return word_embedded, self.input_x

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            doc_vecs = self.AttentionLayer(doc_encoded, name='sent_attention')
            doc_vec = tf.reshape(doc_vecs,[-1,1,self.hidden_size*2])
            return doc_vec
##add
    def headline_word2vec(self):
        with tf.name_scope("head_embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
            embedding_init = embedding_mat.assign(embedding_placeholder)
            sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings})
            
            head_word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_z)
            return head_word_embedded

#    def headline_sent2vec(self, head_word_embedded):
#        with tf.name_scope("headline_sent2vec"):
#            head_word_embedded = tf.reshape(head_word_embedded, [-1, self.max_sentence_length, self.embedding_size])
#            head_word_encoded = self.BidirectionalGRUEncoder(head_word_embedded, name='head_word_encoder')
#            headline_vec = self.AttentionLayer(head_word_encoded, name='head_word_attention')
#            return headline_vec

          
    def news2vec(self, headline_article_vec):
        with tf.name_scope("headline_article2vec"):
            headline_article_vec = tf.reshape(headline_article_vec, [-1, self.max_sentence_length+1, self.hidden_size*2])
            han3_encoded = self.BidirectionalGRUEncoder(headline_article_vec, name='headline_article_encoder')
            news_vec = self.AttentionLayer(han3_encoded, name='han3_attention')
            return news_vec
##to here
    def classifer(self, news_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=news_vec, num_outputs=self.num_classes, activation_fn=None)
            #out = layers.fully_connected(inputs=news_vec, num_outputs=self.num_classes, activation_fn=tf.nn.sigmoid)
            #out = tf.nn.dropout(outs, keep_prob = self.keep_prob)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        #双方向GRUコーディングレイヤー、文中のすべての単語、または文書内のすべての文章ベクトルは、2×hidden_size出力ベクトルを得るためにエンコードされます。次にAttention層を通過した後、最終的な文章/文書ベクトルを得るためにすべての単語または文章の出力ベクトルに重み付けをする。
        
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            #fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            #outputs的size是[batch_size, max_time, hidden_size*2]
            #outputs = tf.concat((fw_outputs, bw_outputs), 2)
            output = tf.concat((fw_outputs, bw_outputs), 2)
            outputs = tf.nn.dropout(output, keep_prob = self.keep_prob)
            return outputs

    def AttentionLayer(self, inputs, name):
        #inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            #   u_context:文章/文書の異なる単語/文章の重要性を区別する文脈的重要度ベクトル
            #   双方向GRUのため、その長さはhidden_sizeの2倍
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            #使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            #shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output
            
