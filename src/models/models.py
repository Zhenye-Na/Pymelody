"""LSTM GAN for music generator in Tensorflow."""

import re
import os
import numpy as np
import tensorflow as tf
from utils.midi_to_matrix import noteStateMatrixToMidi
from tensorflow.contrib import layers


class RNN(object):
    """LSTM model."""

    def __init__(self, model='lstm', ndims=156, nlatent=10,
                 num_layers_g=3, num_units=128, dropout=True,
                 bn=False, num_timesteps=15):
        """Initialize RNN-GAN model.

        Args:
            ndims (int): Number of dimensions in the feature space.
            nlatent (int): Number of dimensions in the latent space.
        """
        # Input songs
        self.x_placeholder = tf.placeholder(
            tf.float32, [None, ndims * num_timesteps])

        # Input sampling
        self.z_placeholder = tf.placeholder(tf.float32, [None, 16, nlatent])

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
        # self.input_keep_prob = tf.placeholder(tf.float32, [])
        # self.output_keep_prob = tf.placeholder(tf.float32, [])
        self.sequence_length_placeholder = tf.placeholder(tf.int32, [None])

        # Build computational graph
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Learning rate
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Add optimizers for appropriate variables
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='d_optimizer').minimize(self.d_loss, var_list=d_vars)

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='g_optimizer').minimize(self.g_loss, var_list=g_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Build sequential encoder model.

        Args:

        Returns:

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            y = self._build_convlayers(x)

        return y

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
        """Build generator.

        Args:

        Returns:

        """
        with tf.variable_scope("generator", reuse=reuse) as scope:

            # Declare Layer Class
            if self._model == 'rnn':
                rnn_layer = tf.contrib.rnn.BasicRNNCell
            elif self._model == 'gru':
                rnn_layer = tf.contrib.rnn.GRUCell
            elif self._model == 'lstm':
                rnn_layer = tf.contrib.rnn.BasicLSTMCell

            cells = []

            # Define MultiRNNCell with Dropout Wrapper
            for _ in range(self._num_layers_g):
                cell = rnn_layer(self._num_units, state_is_tuple=True)
                if self._dropout:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell)
                    # input_keep_prob=self.input_keep_prob,
                    # output_keep_prob=self.output_keep_prob)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            # Simulate the recurrent network over the time
            # outputs is a tensor of shape [batch_size, max_time, cell_state_size]
            # => [batch_size, sequence_length, hidden_dim]
            outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=z,
                                           sequence_length=self.sequence_length_placeholder,
                                           dtype=tf.float32)

            # Swap the axis of 0 and 1
            # outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
            # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # Output layer.
            weight, bias = self._weight_and_bias(
                self._num_units, self._nlatent)
            # logits = tf.nn.softmax(tf.matmul(last, weight) + bias)
            y_hat = tf.matmul(outputs[-1], weight) + bias

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

    def _build_convlayers(self, input_tensor):
        """Build layers for discriminator.

        Args:
            input_tensor:
            bn (boolean): batch normalization or not
            num_layers: number of layers for discriminator

        Return:
            layers
        """
        keep_prob = 0.8

        # Reshape input tensor [15, 156]
        x_song = tf.reshape(
            input_tensor, [16, self._num_timesteps, self._ndims, 1])

        # Conv Layer 1
        w_conv1 = self._weight_variable([5, 5, 1, 3])
        b_conv1 = self._bias_variable([3])
        h_conv1 = self._lrelu(self._conv2d(x_song, w_conv1) + b_conv1)

        # Conv Layer 2
        w_conv2 = self._weight_variable([5, 5, 3, 6])
        b_conv2 = self._bias_variable([6])
        h_conv2 = self._lrelu(self._conv2d(h_conv1, w_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # fc layer 1
        w_fc1 = self._weight_variable([8 * 78 * 16, 1024])
        b_fc1 = self._bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 78 * 16])
        h_fc1 = self._lrelu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # fc layer 2
        w_fc2 = self._weight_variable([1024, 1])
        b_fc2 = self._bias_variable([1])
        y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

        return y

    @staticmethod
    def _lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def _save(self, checkpoint_dir, global_step=None):
        """Save model.

        Args:

        Returns:

        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print('[*] Saving checkpoints...')
        for saver_name, saver in self.saver_dict.iteritems():
            if saver_name in saver_names:
                if not os.path.exists(os.path.join(checkpoint_dir, saver_name)):
                    os.makedirs(os.path.join(checkpoint_dir, saver_name))
                saver.save(self.sess, os.path.join(checkpoint_dir,
                                                   saver_name,
                                                   saver_name),
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
        """Initialize weight and bias for dense layer.

        Args:

        Returns:


        """
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def generate_songs(self, z_sampling):
        """Generate new songs from Gibbs Sampling.

        Args:

        Returns:

        """
        # Index of the lowest note on the piano roll
        lowest_note = 24

        # Index of the highest note on the piano roll
        highest_note = 102

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
