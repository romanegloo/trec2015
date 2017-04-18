import tensorflow as tf
import numpy as np


class MeshCNN(object):
    """
    A CNN for document classification with four classes (diagnosis, test, 
    treatment, and other types).
    Uses an embedding layer, followed by a convolutional, max-pooling and 
    softmax layer
    """
    def __init__(self,
                 sequence_length, num_classes, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param sequence_length: fixed length of sentences. Sentences shorter 
                than the length will be padded with zeroes
        :param num_classes: number of classes in the output layer, four in 
                our cases
        :param vocab_size: the size of vocabulary, needed for W_embedding
        :param embedding_size: the dimensionality of word embedding
        :param filter_sizes: the sizes of convolutional filters
        :param num_filters: the depth of colv layer
        :param l2_reg_lambda: default 0, not considered because it has little 
                effect in sentence classification problems
        """
        # may use setattr
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)  # optional
        self.input_x = None
        self.input_y = None
        self.dropout_keep_prob = None
        self.scores = None
        self.predictions = None
        self.loss = None
        self.accuracy = None

    def inference(self):
        # Placeholders for input, output and dropout
        self.input_x = \
            tf.placeholder(tf.float32,
                           [None, self.sequence_length, self.embedding_size],
                           name="input_x")
        self.input_y = \
            tf.placeholder(tf.int32,
                           [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = \
            tf.placeholder(tf.float32, name="dropout_keep_prob")
        """dropout prob. is set to 0.5 for training and 1 for evaluation"""

        l2_loss = self.l2_loss
        embedded_words_expanded = tf.expand_dims(self.input_x, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = \
                    [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]),
                                name="b")
                conv = tf.nn.conv2d(
                    embedded_words_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
                            name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def loss_accuracy(self):
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + \
                        self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

