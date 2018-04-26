"""LSTM Cell for music generator in Tensorflow."""

import os
import numpy as np
import tensorflow as tf
from utils.midi_to_matrix import noteStateMatrixToMidi


class RNN(object):
    """LSTM model."""

    def __init__(self, model='lstm', ndims=156, num_layers=2, num_units=128, dropout=True):
        """Initialize model.

        Args:

        """
        # Input and targets
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.target_placeholder = tf.placeholder(tf.float32, [None])

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
        logits = self._encoder(self.x_placeholder)

        # Compute loss
        self.cost = self._encoder_loss(logits, y)

        # Add optimizers for appropriate variables
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder).minimize(
            self.cost)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _encoder(self, x):
        """Build sequential encoder model.

        Args:

        Returns:

        """
        batch_size, ndims = x.shape

        # Declare Layer Class
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
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell,
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Simulate the recurrent network over the time
        # outputs is a tensor of shape [batch_size, max_time, cell_state_size]
        # => [batch_size, sequence_length, hidden_dim]
        outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        # outputs = tf.reshape(outputs, [-1, self._num_units])
        outputs = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_units, self._time)
        logits = tf.nn.softmax(tf.matmul(last, weight) + bias)

        return logits

    def _encoder_loss(self, logits, gt):
        """Calculate loss of RNN.

        Args:

        Returns:

        """
        cross_entropy = tf.reduce_sum(
            gt * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        return cross_entropy

    def _decoder():
        """Build sequential decoder model."""
        pass
        outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs,
                                                               initial_state,
                                                               cell,
                                                               loop_function=None,
                                                               scope=None)

    def _load(self, checkpoint_dir):
        """Load model to resume training.

        Not finished!

        Args:


        """
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

def generate_songs(self):
    """Generate new songs from Gibbs Sampling"""
    # Now the model is fully trained, so let's make some music!
    # Run a gibbs chain where the visible nodes are initialized to 0
    sample = gibbs_sample(1).eval(session=self.session, feed_dict={x: np.zeros((50, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Here we reshape the vector to be time x notes, and then save the vector as a midi file
        new_song = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        noteStateMatrixToMidi(new_song, "generated_chord_{}".format(i))
