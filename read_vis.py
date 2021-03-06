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

antnames = "1cB,1eB,1gB,1hB,1kB,2aB,2bB,2cB,2eB,2hB,2jB,2kB,2lB,2mB,3cB,3dB,3lB,4jB,5bB,4gB"
antnames = antnames.split(",")

vis_name = sys.argv[1]

#vis = np.fromfile(vis_name, dtype=np.complex64).\
#        reshape((NFILES*NBLOCKS, NANTS, NCHANS*NFFT, NPOLS*NPOLS))
vis = np.fromfile(vis_name, dtype=np.complex64).\
        reshape((-1, NANTS, NCHANS*NFFT, NPOLS*NPOLS))

for iant in range(NANTS):
    plt.figure(2*iant)
    plt.plot(np.angle(vis[:, iant, :, 0].sum(axis=0)), ".")
    plt.title(antnames[iant])
    plt.figure(2*iant+1)
    plt.plot(np.abs(np.fft.ifftshift(np.fft.ifft(vis[:, iant, :, 0].sum(axis=0)))))
    plt.title(antnames[iant])

plt.show()
