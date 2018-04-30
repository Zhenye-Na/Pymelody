"""Manipulate data."""

import os
import glob
import subprocess
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from utils.midi_to_matrix import midiToNoteStateMatrix, noteStateMatrixToMidi


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
        # files = glob.glob('../../output/generated*.mid*')
        files = glob.glob('{}/*.mid*'.format("../output"))
    except Exception as e:
        raise e

    songs = np.zeros((0, 156))
    print("[*] Songs merging ...")

    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))

            if np.array(song).shape[0] > 10:

                songs = np.concatenate((songs, song))
        except Exception as e:
            raise e

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
