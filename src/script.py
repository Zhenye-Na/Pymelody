"""Transform midi files to mp3."""


import numpy as np
from utils.midi_to_matrix import midiToNoteStateMatrix
from utils.data_tools import convert_midi2mp3, merge_songs
from utils.data_tools import convert_midi2mp3

convert_midi2mp3('./output/', './mp3/')

"""Test."""


# matrix1 = np.array(midiToNoteStateMatrix("../midi/0000.mid"))
# matrix2 = midiToNoteStateMatrix("./midi/AfterYou.mid")
# matrix3 = midiToNoteStateMatrix("./midi/AliceInWonderland.mid")

# print(matrix1.shape)
# print(matrix1[220:225, :])
# # print(matrix2.shape)
# # print(matrix3.shape)

# (1184, 156)
# (2223, 156)
# convert_midi2mp3("../midi2mp3", "../midi22mp3")
merge_songs()
