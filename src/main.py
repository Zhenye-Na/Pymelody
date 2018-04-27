"""High level pipeline."""

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pprint import pprint
from models.models import RNN
from utils.data_tools import get_songs
from utils.midi_to_matrix import noteStateMatrixToMidi


flags = tf.app.flags
FLAGS = flags.FLAGS

# Define file directory
tf.app.flags.DEFINE_string('data_dir', os.path.abspath(
    '../midi'), 'input midi files path.')
tf.app.flags.DEFINE_string('midi_dir', os.path.abspath(
    '../output'), 'output midi files path.')
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.abspath(
    './checkpoint'), 'model save path.')

# Define training hyper-parameters
tf.app.flags.DEFINE_integer('epochs', 5, 'training epochs.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_boolean(
    'resume', 'False', 'whether resume from checkpoint')

# Define model
tf.app.flags.DEFINE_string('model', 'lstm', 'RNN-GAN model class.')

# Define song parameters
tf.app.flags.DEFINE_integer(
    'num_timesteps', 15, 'num_timesteps of each training song.')
tf.app.flags.DEFINE_integer(
    'note_range', 78, 'note_range of each training song.')


FLAGS = tf.app.flags.FLAGS


def train(model, songs, learning_rate=0.0005, batch_size=16, epochs=200):
    """Training process.

    Args:
        model:
        songs (List of np.ndarray):
        learning_rate:
        batch_size:
        epochs:

    Returns:

    The songs are stored in a time x notes format. The size of each
    song is timesteps_in_song x 2*note_range.

    Here we reshape the songs so that each training example is a
    vector with num_timesteps x 2*note_range elements.

    """
    # Iterations for discriminator
    d_iters = 5

    # Iterations for generator
    g_iters = 1

    print('batch size: %d, epoch num: %d, learning rate: %f' %
          (batch_size, epochs, learning_rate))

    # Number of timesteps that we will create at a time
    num_timesteps = 15

    # print("[*] Transform songs...")
    # for song in tqdm(songs):

    #     # Transform to np.ndarray
    #     song = np.array(song)

    #     # Transform song so that timesteps can be divided by num_timesteps and batch_size
    #     song = song[:int(np.floor(
    #         song.shape[0] / (num_timesteps * batch_size)) * (num_timesteps * batch_size))]

    #     # Transform song to matrix representation
    #     song = np.reshape(
    #         song, [song.shape[0] / num_timesteps, song.shape[1] * num_timesteps])

    # seq_length = np.ones(batch_size).astype(np.int32) * 15

    print("[*] Start training...")
    for step in tqdm(range(epochs)):

        print("[*] Transform songs...")
        for song in songs:

            # Transform to np.ndarray
            song = np.array(song)

            # Transform song so that timesteps can be divided by num_timesteps and batch_size
            song = song[:int(np.floor(
                song.shape[0] / (num_timesteps * batch_size)) * (num_timesteps * batch_size))]

            # Transform song to matrix representation
            song = np.reshape(
                song, [song.shape[0] / num_timesteps, song.shape[1] * num_timesteps])

            # Training process
            for i in range(0, len(song), batch_size):

                # generate batch_x
                batch_x = song[i:i + batch_size]

                # Sampling noise from Gibbs Sampling   (Need change here!)
                batch_z = np.random.uniform(
                    0, 1, [batch_size, num_timesteps, model._nlatent])

                for k in tqdm(range(d_iters)):
                    _, d_loss = model.session.run(
                        [model.d_optimizer, model.d_loss],
                        feed_dict={model.x_placeholder: batch_x,
                                   model.z_placeholder: batch_z,
                                   model.learning_rate_placeholder: learning_rate}
                    )

                print('D_loss: {:.4}\n'.format(d_loss))

                for j in tqdm(range(g_iters)):
                    _, g_loss = model.session.run(
                        [model.g_optimizer, model.g_loss],
                        feed_dict={model.z_placeholder: batch_z,
                                   # model.sequence_length_placeholder: seq_length,
                                   model.learning_rate_placeholder: learning_rate}
                    )
                    # model.input_keep_prob: 0.90,
                    # model.output_keep_prob: 0.90,
                print('G_loss: {:.4}\n'.format(g_loss))

        # if step % 100 == 0:
        #     print('Iter: {}'.format(step))
        #     print('D_loss: {:.4}'.format(d_loss))
        #     print('G_loss: {:.4}'.format(g_loss))


def main(_):
    """Main function.

    This script reads data and perform training.
    """
    # Get hyperparameters
    data_dir = FLAGS.data_dir
    midi_dir = FLAGS.midi_dir
    checkpoint_dir = FLAGS.checkpoint_dir
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    resume = FLAGS.resume
    model_type = FLAGS.model
    num_timesteps = FLAGS.num_timesteps
    note_range = FLAGS.note_range

    # pprint(flags.FLAGS.__flags)

    # Get data
    print("[*] Collecting songs...")
    songs = get_songs(data_dir)

    # Build model
    model = RNN()

    # Train model
    train(model, songs, epochs=epochs)

    # Sample noise
    batch_z = np.random.uniform(
        0, 1, [batch_size, num_timesteps, model._nlatent])

    sample = model.generate_songs(batch_z)

    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Reshape the vector to be time x notes, then save as a midi file
        new_song = sample[i, :].reshape(num_timesteps, 2 * note_range)
        noteStateMatrixToMidi(new_song, "generated_chord_{}".format(i))


if __name__ == '__main__':
    tf.app.run()
