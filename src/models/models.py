"""LSTM GAN for music generator in Tensorflow."""

import re
import os
import numpy as np
import tensorflow as tf
import midi_to_matrix
from utils.midi_to_matrix import noteStateMatrixToMidi
from tensorflow.nn import conv2d, relu
from tensorflow.contrib import layers


class RNN(object):
    """LSTM model."""

    def __init__(self, model='lstm', ndims=156, nlatent=10,
                 num_layers_g=3, num_units=128, dropout=True,
                 bn=False, num_timesteps=15):
        """Initialize RNN-GAN model.

        Args:
            ndims (int): Number of dimensions in the feature.
            nlatent (int): Number of dimensions in the latent space.
        """
        # Input and targets
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Define RNN hyper-parameters
        self._model = model
        self._nlatent = nlatent
        self._dropout = dropout
        self._num_units = num_units

        # Define input tensor shape
        self._num_timesteps = num_timesteps
        self._ndims = ndims

        # Define number of layers in generator
        self._num_layers_g = num_layers_g

        # Define batch normalization for discriminator or not
        self._bn = bn

        # Define dropout wrapper parameters
        self.input_keep_prob = tf.placeholder(tf.float32, [])
        self.output_keep_prob = tf.placeholder(tf.float32, [])

        # Build computational graph
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='d_optimizer').minimize(self.e_loss, var_list=d_vars)

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='g_optimizer').minimize(self.d_loss, var_list=g_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Build sequential encoder model.

        Args:

        Returns:

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            self._build_convlayers(x, self._bn, self._num_layers_d)

    def _discriminator_loss(self, y_hat, y):
        """Calculate loss of RNN.

        Args:

        Returns:

        """
        # Label smoothing
        # smooth = 0.1

        # create labels for discriminator
        d_labels_real = tf.ones_like(y)
        d_labels_fake = tf.zeros_like(y_hat)

        # compute loss for real/fake songs
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y, labels=d_labels_real)
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_hat, labels=d_labels_fake)

        loss = tf.reduce_mean(d_loss_fake + d_loss_real)
        return loss

    def _generator(self, z, reuse=False):
        """Build sequential decoder model.

        Args:

        Returns:

        """
        with tf.variable_scope("generator", reuse=reuse) as scope:

            batch_size, ndims = z.shape

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
            outputs, _ = tf.nn.dynamic_rnn(cell, z, dtype=tf.float32)

            # outputs = tf.reshape(outputs, [-1, self._num_units])
            outputs = tf.transpose(outputs, [1, 0, 2])
            last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # Softmax layer.
            weight, bias = self._weight_and_bias(
                self._num_units, self._nlatent)
            # logits = tf.nn.softmax(tf.matmul(last, weight) + bias)
            y_hat = tf.matmul(last, weight) + bias

        return y_hat

    def _generator_loss(self, y_hat):
        """Calculate loss of RNN.

        Args:

        Returns:

        """
        # create labels for generator
        labels = tf.ones_like(y_hat)

        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_hat, labels=labels))
        return l

    @staticmethod
    def _lrelu(x, n, leak=0.2):
        return tf.maximum(x, leak * x, name=n)

    def _build_convlayers(self, input_tensor):
        """Build layers for discriminator.

        Args:
            input_tensor:
            bn (boolean): batch normalization or not
            num_layers: number of layers for discriminator

        Return:
            layers
        """
        num_h1 = 392
        num_h2 = 196

        # Reshape song
        x_song = tf.reshape(
            input_tensor, [-1, self._num_timesteps, self._ndims, 1])

        # Conv layer 1
        w1 = tf.get_variable(name="w1",
                             shape=[self._ndims, num_h1],
                             dtype=tf.float32,
                             initializer=layers.xavier_initializer(uniform=False))

        b1 = tf.get_variable(name="b1",
                             shape=[num_h1],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())

        h1 = self._lrelu(conv2d(x_song, w1) + b1)

        # Conv layer 2
        w2 = tf.get_variable(name="w2",
                             shape=[num_h1, num_h2],
                             dtype=tf.float32,
                             initializer=layers.xavier_initializer(uniform=False))

        b2 = tf.get_variable(name="b2",
                             shape=[num_h2],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())

        h2 = self._lrelu(conv2d(h1, w2) + b2)

        # Conv layer 3
        w3 = tf.get_variable(name="w3",
                             shape=[num_h2, 1],
                             dtype=tf.float32,
                             initializer=layers.xavier_initializer(uniform=False))

        b3 = tf.get_variable(name="b3",
                             shape=[1],
                             dtype=tf.float32,
                             initializer=tf.zeros_initializer())

        y = tf.matmul(h2, w3) + b3

        return y

    def _save(self, checkpoint_dir, component='all', global_step=None):

        if component == 'all':
            saver_names = ['midinet', 'G', 'D', 'invG']
        elif component == 'GD':
            saver_names = ['midinet', 'G', 'D']
        elif component == 'invG':
            saver_names = ['midinet', 'invG']

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print('[*] Saving checkpoints...')
        for saver_name, saver in self.saver_dict.iteritems():
            if saver_name in saver_names:
                if not os.path.exists(os.path.join(checkpoint_dir, saver_name)):
                    os.makedirs(os.path.join(checkpoint_dir, saver_name))
                saver.save(self.sess, os.path.join(checkpoint_dir, saver_name, saver_name),
                           global_step=global_step)

    def _load(self, checkpoint_dir):
        """Load model to resume training.

        Not finished!

        Args:


        """
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
        """Generate new songs from Gibbs Sampling.

        Args:

        Returns:

        """
        # Index of the lowest note on the piano roll
        lowest_note = midi_to_matrix.lowerBound

        # Index of the highest note on the piano roll
        highest_note = midi_to_matrix.upperBound

        # Note range
        note_range = highest_note - lowest_note

        # Number of timesteps that we will create at a time
        num_timesteps = 15

        sample = self._gibbs_sample(1).eval(
            session=self.session,
            feed_dict={self.z_placeholder: np.zeros((50, self._nlatent))})

        for i in range(sample.shape[0]):
            if not any(sample[i, :]):
                continue
            # Reshape the vector to be time x notes, then save as a midi file
            new_song = sample[i, :].reshape(num_timesteps, 2 * note_range)
            noteStateMatrixToMidi(new_song, "generated_chord_{}".format(i))

    def _gibbs_sample():
        """Perform Gibbs Sampling.

        Args:

        Returns:

        """
        pass

    @staticmethod
    def _sample(probs):
        """Sample from a vector of probabilities.

        Args:

        Return:


        """
        # Takes in a vector of probabilities, and returns a random vector of 0s
        # and 1s sampled from the input vector
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
