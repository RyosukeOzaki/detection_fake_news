#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Example of attention coefficients visualization

Uses saved model, so it should be executed after train.py
"""
from batcher import Batcher
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os
import csv

def model_load(word2vec, dataset, data_ids, parameters):
    model_number = input("model num:")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
         saver = tf.train.import_meta_graph("../runs/attention_lstm/save-{}.meta".format(model_number))
         saver.restore(sess, "../runs/attention_lstm/save-{}".format(model_number))
         graph = tf.get_default_graph()
         headline = graph.get_tensor_by_name("headline:0")
         body = graph.get_tensor_by_name("body:0")
         targets = graph.get_tensor_by_name("targets:0")
         keep_prob = graph.get_tensor_by_name("keep_prob:0")

         batcher = Batcher(word2vec=word2vec)
         test_batches = batcher.batch_generator(dataset=dataset["test"], data_ids=data_ids["test"], num_epochs=1, batch_size=1000, sequence_length=parameters["sequence_length"], return_body=True)
         for test_step, (test_batch,_) in enumerate(test_batches):
             feed_dict = {
                           headline: np.transpose(test_batch["headline"], (1, 0, 2)),
                           body: np.transpose(test_batch["body"], (1, 0, 2)),
                           targets: test_batch["targets"],
                           keep_prob: 1.,
                          }
             accuracy_op = graph.get_tensor_by_name("accuracy/Mean:0")
             predictions_op = graph.get_tensor_by_name("accuracy/ToInt32:0")
             test_accuracy, predictions = sess.run([accuracy_op, predictions_op], feed_dict=feed_dict)
             print"\nTEST | accuracy={0:.2f}%   ".format(100.*test_accuracy)
             print "prediction={0}".format(predictions)
             predictions_name = []
             test_predictions = map(str,predictions)
             for predictions_num in range(len(test_predictions)):
                 #if test_predictions[predictions_num] == "0":
                    #predictions_name.append(["neutral"])
                 #if test_predictions[predictions_num] == "1":
                    #predictions_name.append(["entailment"])
	         #if test_predictions[predictions_num] == "2":
                    #predictions_name.append(["contradiction"])
                 #if test_predictions[predictions_num] == "3":
                    #predictions_name.append(["unrelated"])
                 if test_predictions[predictions_num] == "0":
                    predictions_name.append(["True"])
                 if test_predictions[predictions_num] == "1":
                    predictions_name.append(["False"])
             with open('../runs/attention_lstm/log/predictions.csv', "w") as f_predictions:
                  writer_predictions = csv.writer(f_predictions, lineterminator="\n")
                  writer_predictions.writerow("0")
                  writer_predictions.writerows(predictions_name)
             


             alphas_values = []
             for i in range(parameters["sequence_length"]-1):
                 num = str(3*i+5)
                 alphas_op = graph.get_tensor_by_name("body_1/ExpandDims_{}:0".format(num))
                 alphas_values.append(sess.run(alphas_op, feed_dict=feed_dict))
             for j in range(parameters["sequence_length"]-1):
                 if j==0:
                     attention_matrix = np.array(alphas_values[j])
                 else:
                     attention_matrix = np.append(attention_matrix, np.array(alphas_values[j]),axis = 0)
             attention_matrix = np.matrix(attention_matrix)   
    
             df_headlines = pd.read_csv(os.path.join("../runs/attention_lstm/log/", "headlines.csv"), delimiter=",")
             df_bodies = pd.read_csv(os.path.join("../runs/attention_lstm/log/", "bodies.csv"), delimiter=",")
             df_headlines = df_headlines.dropna(axis=1)
             df_bodies = df_bodies.dropna(axis=1)
             header_headlines = range(len(df_headlines.columns))
             header_headlines = map(str,header_headlines)
	     header_bodies = range(len(df_bodies.columns))
             header_bodies = map(str,header_bodies)
             headlines = df_headlines[header_headlines]
             bodies = df_bodies[header_bodies]
             resize_row_attention_matrix = np.delete(attention_matrix, np.s_[len(df_headlines.columns):parameters["sequence_length"]], 1)
             resize_line_attention_matrix = np.delete(resize_row_attention_matrix, np.s_[len(df_bodies.columns):parameters["sequence_length"]-1], 0)
             fig = plt.figure()
             ax = fig.add_subplot(1,1,1)
             ax.set_aspect('equal')
             plt.imshow(resize_line_attention_matrix, interpolation='nearest', cmap=plt.cm.Greys)
             plt.yticks(np.arange(0,len(bodies.ix[0])), bodies.ix[0])
             plt.xticks(np.arange(0,len(headlines.ix[0])), headlines.ix[0])
             plt.ylabel('articlebody',fontsize=18)
             plt.xlabel('headline',fontsize=18)
             plt.colorbar()
             plt.show()
             break
