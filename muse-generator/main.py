"""High level pipeline."""

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pprint import pprint
from models.models import RNN
from utils.data_tools import get_songs


flags = tf.app.flags
FLAGS = flags.FLAGS

# Define file directory
tf.app.flags.DEFINE_string('data_dir', os.path.abspath(
    './midi'), 'input midi files path.')
tf.app.flags.DEFINE_string('midi_dir', os.path.abspath(
    './output'), 'output midi files path.')
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.abspath(
    './checkpoint'), 'model save path.')

# Define training hyper-parameters
tf.app.flags.DEFINE_integer('epochs', 200, 'training epochs.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_boolean(
    'resume', 'False', 'whether resume from checkpoint')

# Define model
tf.app.flags.DEFINE_string('model', 'lstm', 'model class.')

FLAGS = tf.app.flags.FLAGS


def train(model, songs, learning_rate=0.0005, batch_size=16, epochs=200):
    """Training process.

    Args:

    Returns:

    The songs are stored in a time x notes format. The size of each
    song is timesteps_in_song x 2*note_range.

    Here we reshape the songs so that each training example is a
    vector with num_timesteps x 2*note_range elements.

    """
    # Number of timesteps that we will create at a time
    num_timesteps = 15

    for step in tqdm(range(0, epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(
                np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
            song = np.reshape(
                song, [song.shape[0] / num_timesteps, song.shape[1] * num_timesteps])

            for i in tqdm(range(1, len(song), batch_size)):

                batch_x = song[i:i + batch_size]
                print(batch_x)
                model.session.run(
                    model.update_op_tensor,
                    feed_dict={model.x_placeholder: batch_x,
                               model.learning_rate_placeholder: learning_rate}
                )


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

    pprint(flags.FLAGS.__flags)

    # Get dataset
    # mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    songs = get_songs(data_dir)

    # Build model
    model = RNN()

    # Train model
    train(model, songs)


if __name__ == '__main__':
    tf.app.run()
