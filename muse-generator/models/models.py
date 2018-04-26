"""LSTM Cell for music generator in Tensorflow."""

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


class Model(object):
    """LSTM model."""

    def __init__(self, ndims=156, num_layers=2, dropout=True):
        """Initialize model."""
        # self.t_layer_sizes = t_layer_sizes
        # self.p_layer_sizes = p_layer_sizes

        # # From our architecture definition, size of the notewise input
        # self.t_input_size = 80

        # # time network maps from notewise input size to various hidden sizes
        # self.time_model = StackedCells(
        #     self.t_input_size, celltype=LSTM, layers=t_layer_sizes)
        # self.time_model.layers.append(PassthroughLayer())

        # # pitch network takes last layer of time model and state of last note, moving upward
        # # and eventually ends with a two-element sigmoid layer
        # p_input_size = t_layer_sizes[-1] + 2
        # self.pitch_model = StackedCells(
        #     p_input_size, celltype=LSTM, layers=p_layer_sizes)
        # self.pitch_model.layers.append(
        #     Layer(p_layer_sizes[-1], 2, activation=T.nnet.sigmoid))

        # self.dropout = dropout

        # self.conservativity = T.fscalar()
        # self.srng = T.shared_randomstreams.RandomStreams(
        #     np.random.randint(0, 1024))

        # self.setup_train()
        # self.setup_predict()
        # self.setup_slow_walk()

        # Define RNN layers hyper-parameters
        self._ndims = ndims
        self._num_layers = num_layers
        self._dropout = dropout

        # Define dropout wrapper parameters
        self.input_keep_prob = tf.placeholder(tf.float32, [])
        self.output_keep_prob = tf.placeholder(tf.float32, [])

    def _sequential(self, num_layers=2, num_units=128, dropout=True):
        """Build sequential model.

        Args:

        Returns:

        """
        # Define helper function to declare layer class
        if model == 'rnn':
            rnn_layer = tf.contrib.rnn.BasicRNNCell
        elif model == 'gru':
            rnn_layer = tf.contrib.rnn.GRUCell
        elif model == 'lstm':
            rnn_layer = tf.contrib.rnn.BasicLSTMCell

        cells = []

        # Define MultiRNNCell with dropout
        for _ in range(num_layers):
            cell = rnn_layer(num_units)  # Or LSTMCell(num_units)
            if dropout:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=1.0 - dropout)
            cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Simulate the recurrent network over the time
        output, last_state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    def _generator():
        pass















    def _load(checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
