#coding=utf-8
import tensorflow as tf
import model
import time
import datetime
import os
#from load_data import read_dataset

#add↓
import random
import csv
import argparse
import numpy as np
import IDtxtprepare
#  ↑


embedding_path = 'glove.6B.100d.txt'
num_ = input("num:")
train_filename = 'dataset/train{}.txt'.format(num_)
#train_filename = 'dataset/fake_train.txt'
dev_filename = 'dataset/test{}.txt'.format(num_)

#parser = argparse.ArgumentParser(description=__doc__)
#parser.add_argument('embeddings', help='Text or numpy file with word embeddings')
#parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy embedding file is given)')

# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 400003, "vocabulary size")#46960:yelp academ  2170:ent dev  13761:ent train  
tf.flags.DEFINE_integer("num_classes", 4, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")#default:0.01
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
#tf.flags.DEFINE_float("keep_prob", 0.8, "dropout rate")

FLAGS = tf.flags.FLAGS

word_dict, embed = IDtxtprepare.load_embeddings(embedding_path)

train_data_pair, train_body, train_head = IDtxtprepare.read_corpus(train_filename)
train_x, train_z, train_y = IDtxtprepare.create_dataset(train_data_pair, word_dict)#x:Body z:Headline y:label
print('train data size : ', len(train_y))

dev_data_pair, dev_body, dev_head = IDtxtprepare.read_corpus(dev_filename)
dev_x, dev_z, dev_y = IDtxtprepare.create_dataset(dev_data_pair, word_dict)
print('dev data size : ', len(dev_x))

Head_sent=[]
head_sent=[]
for i in dev_head:
    head_sent=" ".join(i)
    Head_sent.append([head_sent])


Body_sent=[]
body_sent=[]
for i in range(len(dev_body)):
    for j in dev_body[i]:
        body_sent=" ".join(j)
        Body_sent.append([body_sent])


#config = tf.ConfigProto(allow_soft_placement=True,device_count={'GPU': 1})
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session() as sess:
    han = model.HAN(vocab_size=FLAGS.vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
        
    with tf.name_scope('softmax'):
        softmax = tf.nn.softmax(han.out)
    
    output = han.out
    input_x = han.input_x
    #doc_vec = han.doc_vec
    #news_vec = han.news_vec
    #ha_vec = han.headline_article_vec
    #Sent = han.sent_vec
    
       
    timestamp = str(int(time.time()))
    log_dir = os.path.abspath("log{0}".format(num_))
    if not os.path.exists(log_dir):
       os.makedirs(log_dir)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    #optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)
    pred_summary = tf.summary.scalar('predict', predict)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch, z_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.input_z: z_batch,
            han.max_sentence_num: 20,
            han.max_sentence_length: 30,
            han.batch_size: 24,
            han.keep_prob: 0.8
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)
        #time_str = str(int(time.time()))
        dateT = str(datetime.datetime.now())
        if step % 10 == 0:
            print("{}: step {}, loss {:g}, acc {:g}".format(dateT, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)

        return step

    def dev_step(x_batch, y_batch, z_batch, dev_BODY, dev_HEAD, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.input_z: z_batch,
            han.max_sentence_num: 20,
            han.max_sentence_length: 30,
            han.batch_size: 64,
            han.keep_prob: 1.0
        }#default num:30, len:30, batch:64
        step, summaries, cost, accuracy, preds, targets, SM, XXX = sess.run([global_step, dev_summary_op, loss, acc, predict,label, softmax, input_x], feed_dict)
        #time_str = str(int(time.time()))
        dateT = str(datetime.datetime.now())
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(dateT, step, cost, accuracy))

        
        with open('log{0}/predictions.csv'.format(num_), 'w') as predfile, open('log{0}/target.csv'.format(num_), 'w') as labfile, open('log{0}/softmax.csv'.format(num_),'w') as softfile, open('log{0}/body.csv'.format(num_), 'w') as bodyfile, open('log{0}/head.csv'.format(num_), 'w') as headfile:
            writer_pred = csv.writer(predfile,delimiter=str(u' '), lineterminator='\n')
            writer_pred.writerow('0')
            writer_lab = csv.writer(labfile, lineterminator='\n')
            writer_lab.writerow('0')
            writer_soft = csv.writer(softfile,lineterminator='\n')
            writer_soft.writerow('0')
            writer_body = csv.writer(bodyfile,lineterminator='\n')
            writer_body.writerow('0')
            writer_head = csv.writer(headfile,lineterminator='\n')
            writer_head.writerow('0') 
            for pred in preds:
                writer_pred.writerow(str(pred))
            for target in targets:
                writer_lab.writerow(str(target))
            for soft in SM:
                K = str(np.max(soft))
                softfile.write(K+'\n')
            for body in dev_BODY:
                writer_body.writerow(body)
            for head in dev_HEAD:
                writer_head.writerow(head)

        if writer:
            writer.add_summary(summaries, step)
    
    ID = range(len(dev_x))
    X=[]
    Y=[]
    Z=[]
    XX=[]
    ZZ=[]
    index = random.sample(ID,len(dev_x))
    for idx in index:
        X.append(dev_x[idx])#bodyの分散表現
        Y.append(dev_y[idx])
        Z.append(dev_z[idx])#headlineの分散表現
        XX.append(Body_sent[idx])#bodyの生文
        ZZ.append(Head_sent[idx])#headlineの生文
        
    for epoch in range(0,FLAGS.num_epochs):
        print('\n-----------------current epoch %s--------------------' % (epoch + 1))
        #for i in range(0, len(train_x)-1, FLAGS.batch_size):
        for i in range(0, len(train_x)-1, 24):
            #x = train_x[i:i + FLAGS.batch_size]
            #y = train_y[i:i + FLAGS.batch_size]
            #z = train_z[i:i + FLAGS.batch_size]
            x = train_x[i:i+24]
            y = train_y[i:i+24]
            z = train_z[i:i+24]
            #X = dev_x[1:1001]
            #Y = dev_y[1:1001]
            #Z = dev_z[1:1001]
            step = train_step(x, y, z)
            if step % FLAGS.evaluate_every == 0:
                #dev_step(dev_x, dev_y, dev_z, dev_summary_writer)
                dev_step(X, Y, Z, XX ,ZZ, dev_summary_writer)
