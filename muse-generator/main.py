"""High level pipeline."""

import os
import numpy as np
import tensorflow as tf
from pprint import pprint
from models.models import rnn_model

flags = tf.app.flags
FLAGS = flags.FLAGS

# Define file directory
tf.app.flags.DEFINE_string('data_dir', os.path.abspath('./midi'), 'input midi files path.')
tf.app.flags.DEFINE_string('midi_dir', os.path.abspath('./output'), 'output midi files path.')
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.abspath('./checkpoint'), 'model save path.')

# Define training hyper-parameters
tf.app.flags.DEFINE_integer('epochs', 50, 'training epochs.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_boolean('resume', 'False', 'whether resume from checkpoint')

# Define model
tf.app.flags.DEFINE_string('model', 'lstm', 'model class.')

FLAGS = tf.app.flags.FLAGS


def train():
    """Training process."""
    pass


def main(_):
    """Main function."""
    pprint(flags.FLAGS.__flags)
    train()


if __name__ == '__main__':
    tf.app.run()
