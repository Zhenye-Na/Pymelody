"""Manipulate data."""

import os
import glob
import subprocess
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from utils.midi_to_matrix import midiToNoteStateMatrix


def get_songs(path):
    """Get songs from MIDI files.

    Args:

    Returns:
        songs (List of np.ndarray):
    """
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs


def merge_songs():
    """Merge all output songs to a single midi file."""
    try:
        files = glob.glob('generated*.mid*')
    except Exception as e:
        raise e

    songs = np.zeros((0, 156))
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))

            if np.array(song).shape[0] > 10:
                # songs.append(song)
                songs = np.concatenate((songs, song))
        except Exception as e:
            raise e
    print "samlpes merging ..."
    print np.shape(songs)
    noteStateMatrixToMidi(songs, "final")


def generate_batch():
    """Gnerate midi sample batch."""
    pass


def convert_midi2mp3(input_dir, output_dir):
    """Convert all midi files of the given directory to mp3.

    Args:

    Returns:

    """
    assert os.path.exists(input_dir)
    os.makedirs(output_dir)

    print('Converting:')
    i = 0
    for filename in glob.iglob(os.path.join(input_dir, '*.mid')):
        print(filename)
        in_name = filename
        out_name = os.path.join(output_dir, os.path.splitext(
            os.path.basename(filename))[0] + '.mp3')
        # TODO: Redirect stdout to avoid polluting the screen (have cleaner printing)
        command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(
            in_name, out_name)
        subprocess.call(command, shell=True)
        i += 1
    print('Converting finished! {} files converted.'.format(i))


def sample(probs):
    """Sample.

    Args:
        probs
    Returns:
        a random vector of 0s and 1s sampled from the input vector

    """
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


def gibbs_sample(k):
    """Perform Gibbs sampling."""
    # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        # Propagate the visible values to sample the hidden values
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))

        # Propagate the hidden values to sample the visible values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))

        return count + 1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0)
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])

    x_sample = tf.stop_gradient(x_sample)
    return x_sample
