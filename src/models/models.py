"""LSTM GAN for music generator in Tensorflow."""

import re
import os
import numpy as np
import tensorflow as tf
from utils.midi_to_matrix import noteStateMatrixToMidi
from tensorflow.contrib import layers


class RNN_gan(object):
    """LSTM model."""

    def __init__(self, model='lstm', ndims=156, nlatent=156,
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
        self.z_placeholder = tf.placeholder(
            tf.float32, [None, num_timesteps, nlatent])

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
        # self.input_keep_prob_placeholder = tf.placeholder(tf.float32, [])
        # self.output_keep_prob_placeholder = tf.placeholder(tf.float32, [])
        # self.sequence_length_placeholder = tf.placeholder(tf.int32, [None])

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
            else:
                raise Exception("Invalid cell type: {}".format(self._model))

            cells = []

            # Define MultiRNNCell with Dropout Wrapper
            for _ in range(self._num_layers_g):
                cell = rnn_layer(self._num_units, state_is_tuple=True)
                if self._dropout:
                    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                         input_keep_prob=1.0,
                                                         output_keep_prob=0.87)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            # Perform fully dynamic unrolling of inputs
            outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=z,
                                           dtype=tf.float32)

            # Swap the axis of 0 and 1
            # outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
            # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # Output layer.
            weight, bias = self._weight_and_bias(
                self._num_units, self._ndims * self._num_timesteps)
            # logits = tf.nn.softmax(tf.matmul(last, weight) + bias)
            y_hat = tf.nn.sigmoid(tf.matmul(outputs[-1], weight) + bias)

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
        keep_prob = 0.9

        # Reshape input tensor [-1, 15, 156, 1]
        x_song = tf.reshape(
            input_tensor, [-1, self._num_timesteps, self._ndims, 1])

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
        w_fc1 = self._weight_variable([8 * 78 * 6, 1024])
        b_fc1 = self._bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 78 * 6])
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
        sample = self.x_hat.eval(
            session=self.session,
            feed_dict={self.z_placeholder: z_sampling})

        return sample

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


# class RNN_rbm(object):
#     """Implement rnn_rbm in tf."""

#     def __init__(self):
#         """Build the RNN-RBM."""
#         # The range of notes that we can produce
#         self.note_range = 156

#         # The size of each data vector and the size of the RBM visible layer
#         self.n_visible = 156 * 60

#         # The size of the RBM hidden layer
#         self.n_hidden = 50

#         # The size of each RNN hidden layer
#         self.n_hidden_recurrent = 100

#         # The placeholder variable that holds our data
#         self.x = tf.placeholder(tf.float32, [None, self.n_visible])

#         # learning rate
#         self.lr = tf.placeholder(tf.float32)

#         # the batch size
#         self.size_bt = tf.shape(self.x)[0]

#         # Define variables
#         self.W = tf.Variable(
#             tf.zeros([self.n_visible, self.n_hidden]), name="W")
#         self.Wuh = tf.Variable(
#             tf.zeros([self.n_hidden_recurrent, self.n_hidden]), name="Wuh")
#         self.Wuv = tf.Variable(
#             tf.zeros([self.n_hidden_recurrent, self.n_visible]), name="Wuv")
#         self.Wvu = tf.Variable(
#             tf.zeros([self.n_visible, self.n_hidden_recurrent]), name="Wvu")
#         self.Wuu = tf.Variable(
#             tf.zeros([self.n_hidden_recurrent, self.n_hidden_recurrent]), name="Wuu")
#         self.bh = tf.Variable(tf.zeros([1, self.n_hidden]), name="bh")
#         self.bv = tf.Variable(tf.zeros([1, self.n_visible]), name="bv")
#         self.bu = tf.Variable(
#             tf.zeros([1, self.n_hidden_recurrent]), name="bu")
#         self.u0 = tf.Variable(
#             tf.zeros([1, self.n_hidden_recurrent]), name="u0")
#         self.BH_t = tf.Variable(tf.zeros([1, self.n_hidden]), name="BH_t")
#         self.BV_t = tf.Variable(tf.zeros([1, self.n_visible]), name="BV_t")

#     def _rnn_recurrence(self, u_tm1, sl):
#         # Iterate through the data in the batch and generate the values of the RNN hidden nodes
#         sl = tf.reshape(sl, [1, self.n_visible])
#         u_t = (tf.tanh(self.bu + tf.matmul(sl, self.Wvu) +
#                        tf.matmul(u_tm1, self.Wuu)))
#         return u_t

#     def _visible_bias_recurrence(self, bv_t, u_tm1):
#         # Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
#         bv_t = tf.add(self.bv, tf.matmul(u_tm1, self.Wuv))
#         return bv_t

#     def _hidden_bias_recurrence(self, bh_t, u_tm1):
#         """bla."""
#         # Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
#         bh_t = tf.add(self.bh, tf.matmul(u_tm1, self.Wuh))
#         return bh_t

#     def _generate_recurrence(self, count, k, u_tm1, primer, x, music):
#         # This function builds and runs the gibbs steps for each RBM in the chain to generate music
#         # Get the bias vectors from the current state of the RNN
#         bv_t = tf.add(self.bv, tf.matmul(u_tm1, self.Wuv))
#         bh_t = tf.add(self.bh, tf.matmul(u_tm1, self.Wuh))

#         # Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
#         x_out = RBM.gibbs_sample(primer, W, bv_t, bh_t, k=25)

#         # Update the RNN hidden state based on the musical output and current hidden state.
#         u_t = (tf.tanh(self.bu + tf.matmul(x_out, self.Wvu) +
#                        tf.matmul(u_tm1, self.Wuu)))

#         # Add the new output to the musical piece
#         music = tf.concat(0, [music, x_out])
#         return count + 1, k, u_t, x_out, x, music

#     def _generate(self, num, x=x, size_bt=size_bt, u0=u0, n_visible=n_visible, prime_length=100):
#         """
#             This function handles generating music. This function is one of the outputs of the build_rnnrbm function
#             Args:
#                 num (int): The number of timesteps to generate
#                 x (tf.placeholder): The data vector. We can use feed_dict to set this to the music primer. 
#                 size_bt (tf.float32): The batch size
#                 u0 (tf.Variable): The initial state of the RNN
#                 n_visible (int): The size of the data vectors
#                 prime_length (int): The number of timesteps into the primer song that we use befoe beginning to generate music
#             Returns:
#                 The generated music, as a tf.Tensor
#         """
#         Uarr = tf.scan(rnn_recurrence, x, initializer=u0)
#         U = Uarr[np.floor(
#             prime_length / midi_manipulation.num_timesteps), :, :]
#         [_, _, _, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
#                                                         generate_recurrence, [tf.constant(1, tf.int32), tf.constant(num), U,
#                                                                               tf.zeros(
#                                                                                   [1, n_visible], tf.float32), x,
#                                                                               tf.zeros([1, n_visible],  tf.float32)])
#         return music

#     # # Reshape our bias matrices to be the same size as the batch.
#     # tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
#     # tf.assign(BV_t, tf.tile(BV_t, [size_bt, 1]))
#     # # Scan through the rnn and generate the value for each hidden node in the batch
#     # u_t = tf.scan(rnn_recurrence, x, initializer=u0)
#     # # Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
#     # BV_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros(
#     #     [1, n_visible], tf.float32)), [size_bt, n_visible])
#     # BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros(
#     #     [1, n_hidden], tf.float32)), [size_bt, n_hidden])
#     # # Get the free energy cost from each of the RBMs in the batch
#     # cost = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k=15)
#     # return x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, u0
