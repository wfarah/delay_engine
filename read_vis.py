import numpy as np
import matplotlib.pyplot as plt
import sys

NFILES  = 1
NBLOCKS = 128  # number of data blocks per GUPPI file
NCHANS  = 96*2 # number of channels per block
NTIMES  = 8192 # number of time samples per block
NPOLS   = 2    # number of polarisations per block
NANTS   = 20   # number of antennas per block
NFFT    = 1    # number of FFTs we want to take. 0.5/4 = 125 kHz channel width

vis_name = sys.argv[1]

vis = np.fromfile(vis_name, dtype=np.complex64).\
        reshape((NFILES*NBLOCKS, NANTS, NCHANS*NFFT, NPOLS*NPOLS))

for iant in range(NANTS):
    plt.figure(iant)
    plt.plot(np.angle(vis[:, iant, :, 0].sum(axis=-1)))

plt.show()
