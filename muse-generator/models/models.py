"""LSTM Cell for music generator in Tensorflow."""

import os
import numpy as np
import tensorflow as tf


def rnn_model(model, dropout=True, num_units=128, num_layers=2, batch_size=64, learning_rate=0.01):
    """Construct RNN LSTM model.

    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param num_units:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    # Default should be LSTM
    if model == 'rnn':
        rnn_layer = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        rnn_layer = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        rnn_layer = tf.contrib.rnn.BasicLSTMCell

    # dropout
    dropout = tf.placeholder(tf.float32)

    cells = []
    for _ in range(num_layers):
        cell = rnn_layer(num_units)  # Or LSTMCell(num_units)
        if dropout:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)


class RNN(object):
    """LSTM model."""

    def __init__(self, model='lstm', ndims=156, num_layers=2, num_units=128, dropout=True):
        """Initialize model.

        Args:

        """
        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Define RNN hyper-parameters
        self._ndims = ndims
        self._model = model
        self._dropout = dropout
        self._num_units = num_units
        self._num_layers = num_layers

        # Define dropout wrapper parameters
        self.input_keep_prob = tf.placeholder(tf.float32, [])
        self.output_keep_prob = tf.placeholder(tf.float32, [])

        # Build computational graph
        blabla = self._sequential(self.x_placeholder)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _sequential(self, x):
        """Build sequential model.

        Args:

        Returns:

        """
        # get lasting time for particular MIDI files.
        time, _ = x.shape

        # Define helper function to declare layer class
        if self._model == 'rnn':
            rnn_layer = tf.contrib.rnn.BasicRNNCell
        elif self._model == 'gru':
            rnn_layer = tf.contrib.rnn.GRUCell
        elif self._model == 'lstm':
            rnn_layer = tf.contrib.rnn.BasicLSTMCell

        cells = []

        # Define MultiRNNCell with Dropout Wrapper
        for _ in range(self._num_layers):
            cell = rnn_layer(self._num_units, state_is_tuple=True)
            if self._dropout:
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Simulate the recurrent network over the time
        output, last_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        output = tf.reshape(output, [-1, self._num_units])
        # output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_units, time)
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

        # weights = tf.Variable(tf.truncated_normal([self._num_units, vocab_size + 1]))
        # bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
        # logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

    def _generator():
        pass

    def _load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        """Initialize weight and bias for dense layer."""
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
