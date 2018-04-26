"""Manipulate data."""

import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils.midi_to_matrix import midiToNoteStateMatrix


def get_songs(path):
    """Get songs from MIDI files.

    Args:

    Returns:
        songs (List):
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

def generate_batch():
    """blablabla."""
    pass




