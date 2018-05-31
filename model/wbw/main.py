#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from utils import load_data, train
import numpy as np
from model_load import model_load

dirname, filename = os.path.split(os.path.abspath(__file__))
GIT_DIR = "/".join(dirname.split("/")[:-1])

PYTHON_DIR = os.path.join(GIT_DIR, "python")
DATA_DIR = os.path.join(GIT_DIR, "data")
RUNS_DIR = os.path.join(GIT_DIR, "runs")


if __name__ == "__main__":

    # ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="active this flag to train the model")
    parser.add_argument("--model", action="store_true", help="read the model and test input data")
    parser.add_argument("--data_dir", default=os.path.join(DATA_DIR, "dataset"), help="path to the SNLI dataset directory")
    parser.add_argument("--word2vec_path", default=os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin"), help="path to the pretrained Word2Vect .bin file")
    parser.add_argument("--model_name", type=str, default="attention_lstm")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--batch_size_train", type=int, default=24)
    parser.add_argument("--batch_size_test", type=int, default=80)
    parser.add_argument("--gpu", type=str, default="0", help="set gpu to '' to use CPU mode")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=300, help="Word2Vec dimension")
    parser.add_argument("--sequence_length", type=int, default=100, help="final length of each sequence (premise and hypothesis), padded with null-words if needed")
    parser.add_argument("--num_units", type=int, default=100, help="LSTM output dimension (k in the original paper)")
    args = parser.parse_args()

    # PARAMETERS
    parameters = {
                    "runs_dir": RUNS_DIR,
                    "embedding_dim": args.embedding_dim,
                    "num_units": args.num_units,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "keep_prob": args.keep_prob,
                    "model_name": args.model_name,
                    "gpu": args.gpu or None,
                    "batch_size": {"train": args.batch_size_train, "test": args.batch_size_test},
                    "sequence_length": args.sequence_length,
                    "weight_decay": args.weight_decay,
                }

    for key, parameter in parameters.iteritems():
        print "{}: {}".format(key, parameter)

    # MAIN
    
    
    if args.train:
        word2vec, dataset, data_ids = load_data(data_dir=args.data_dir, word2vec_path=args.word2vec_path)
        #class_weights=[1.0/dataset["train"]["targets"].count(0), 1.0/dataset["train"]["targets"].count(1), 1.0/dataset["train"]["targets"].count(2)]
        class_weights=[1.0, 1.0]
        #class_weights=[1.0, 1.0, 1.0, 1.0]
        train(word2vec=word2vec, dataset=dataset, data_ids=data_ids, parameters=parameters, class_weights=class_weights)
    if args.model:
        word2vec, dataset, data_ids = load_data(data_dir=args.data_dir, word2vec_path=args.word2vec_path, test=True)
        model_load(word2vec=word2vec, dataset=dataset, data_ids=data_ids, parameters=parameters)
