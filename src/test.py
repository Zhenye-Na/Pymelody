"""Test."""

import numpy as np
from utils.midi_to_matrix import midiToNoteStateMatrix

matrix1 = np.array(midiToNoteStateMatrix("../midi/0000.mid"))
# matrix2 = midiToNoteStateMatrix("./midi/AfterYou.mid")
# matrix3 = midiToNoteStateMatrix("./midi/AliceInWonderland.mid")

print(matrix1.shape)
print(matrix1[220:225, :])
# print(matrix2.shape)
# print(matrix3.shape)

# (1184, 156)
# (2223, 156)
