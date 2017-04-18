#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import sys
import time
import datetime
from model import MeshCNN
import data_prep

# Config
# ==================================================

# data loading
sequence_len = 58             # max number of sentences in a document

# model hyperparameters
embedding_dim = 200           # word embedding size (using pre-trained word2vec)
filter_sizes = "3,4,5"        # comma-separated filter sizes
num_filters = 128             # number of filters per filter size
dropout_keep_prob = 0.5       # dropout keep probability
l2_reg_lambda = 0.0           # l2 regularization lambda (optional)

# Training parameters
batch_size = 100               # batch size (default: 64)
num_epochs = 100              # Number of training epochs, 200
evaluate_every = 100          # Evaluate model on dev set after this many steps
checkpoint_every = 500       # Save model after this many steps (default: 100)
num_checkpoints = 5           # number of checkpoints to store (default: 5)


# Data Preparation
#   if the size of dataset is not small, you may need to implement feeder
#   https://www.tensorflow.org/programmers_guide/reading_data
# ==================================================

print("Loading data" + '='*60)
x, y = data_prep.load_data_and_labels("data/pdata_toy.train",
                                      sequence_len, embedding_dim)
shuffled_indices = np.random.permutation(np.arange(len(y)))
x_test, y_test = data_prep.load_data_and_labels("data/pdata_toy.test",
                                                sequence_len, embedding_dim)
shuffled_indices_test = np.random.permutation(np.arange(len(y_test)))

# # Build Vocabulary
# max_document_length = max([len(x.split(' ')) for x in x_text])
# print("Max_document_length: {}".format(max_document_length))
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))
# print("Vocabulary Size: {:d}".format(len(
# vocab_processor.vocabulary_)))

# Training
# ==================================================
with tf.Graph().as_default():
    # define a session
    session_conf = tf.ConfigProto(  # API http://bit.ly/2nNy325
        allow_soft_placement=False,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    # the actual computation occurs within a session
    with sess.as_default():
        # instantiate MeshCNN model
        cnn = MeshCNN(
            sequence_length=sequence_len,
            num_classes=y.shape[1],
            # vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, filter_sizes.split(','))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda
        )
        cnn.inference()
        cnn.loss_accuracy()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set.
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Action !
        #--- generate batches
        batches = data_prep.batch_iter(shuffled_indices, batch_size,
                                              num_epochs)

        curr_epoch = -1
        for epoch, batch in batches:
            if epoch != curr_epoch:
                print("running epoch #{}".format(epoch))
                curr_epoch = epoch
            train_step(x[batch], y[batch])
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                sample = np.random.choice(shuffled_indices_test,
                                          replace=False, size=50)
                dev_step(x_test[sample], y_test[sample],
                         writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
