#!/usr/bin/env python3

import tensorflow as tf
import os
import sys
import argparse
from model import MeshCNN
from data_prep import sent_embedding, load_data_and_labels


"""
restore most recently trained model, and use it to make an inference.
A user will need to provide a blob of sentences. Given the text, it will 
print out the scores of each document class (diagnosis, test, treatment, 
and others) along with the classified label.
"""
# data loading
sequence_len = 58             # max number of sentences in a document

# model hyperparameters
embedding_dim = 200           # word embedding size (using pre-trained word2vec)
filter_sizes = "3,4,5"        # comma-separated filter sizes
num_filters = 128             # number of filters per filter size
dropout_keep_prob = 0.5       # dropout keep probability
l2_reg_lambda = 0.0           # l2 regularization lambda (optional)

# Training parameters
batch_size = 32               # batch size (default: 64)
num_epochs = 600              # Number of training epochs, 200
evaluate_every = 200          # Evaluate model on dev set after this many steps
checkpoint_every = 1000       # Save model after this many steps (default: 100)
num_checkpoints = 5           # number of checkpoints to store (default: 5)


def app_run(chk_path):
    with tf.Graph().as_default():
        # define a session
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # instantiate MeshCNN model
            cnn = MeshCNN(
                sequence_length=sequence_len,
                num_classes=4,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(','))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda
            )
            cnn.inference()
            # cnn.loss_accuracy()

            # restore session with checkpoint data
            if not os.path.exists(chk_path + '.meta'):
                SystemError("checkpoint path does not exist [{}]".format(chk_path))


            print("=== restoring a model ===")
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, chk_path)
            print("model restored [{}]".format(chk_path))

            # print("=== inference (on test file) ===")
            #
            # x, y = load_data_and_labels("data/pdata_toy.test",
            #                                       sequence_len, embedding_dim)
            #
            #
            # scores, predict, acc = sess.run([cnn.scores, cnn.predictions,
            #                             cnn.accuracy],
            #                            {cnn.input_x: x,
            #                             cnn.input_y: y,
            #                             cnn.dropout_keep_prob: 1.0})
            # print("=== results ===")
            # print("Accuracy: {:g}".format(acc))

            while True:
                print("type senteneces ending with ctrl+d, to quit just ctrl+d: \n")
                line = input()
                # line = sys.stdin.read()
                emb = sent_embedding(line, sequence_len, embedding_dim)


                scores, predict = sess.run([cnn.scores, cnn.predictions],
                                           {cnn.input_x: [emb],
                                            cnn.dropout_keep_prob: 1.0})
                print("=== results ===")
                print(scores, predict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name',
                        help="relative path to checkpoint file,\n " +
                        "ex. ./runs/1490917391/checkpoints/model-9200")
    args = parser.parse_args()

    app_run(args.file_name)

