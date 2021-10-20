import numpy as np
import sys
from guppi import guppi
# Guppi is a custom library, might need some tweeking too?

# Let's assume the following
# they should be true for our configuration
NBLOCKS = 128  # number of data blocks per GUPPI file
NCHANS  = 96*2 # number of channels per block
NTIMES  = 8192 # number of time samples per block
NPOLS   = 2    # number of polarisations per block
NANTS   = 20   # number of antennas per block
NFFT    = 4    # number of FFTs we want to take. 0.5/4 = 125 kHz channel width


def do_npoint_spec(arr, nfft):
    """
    arr is 1D, nfft is an integer
    """
    if nfft == 1:
        return arr[:, np.newaxis]
    nspecs = arr.size // nfft # assuming nfft divides arr.size!
    arr_fft = np.zeros_like(arr).reshape(nspecs, nfft)

    for ispec in range(nspecs):
        arr_fft[ispec] = np.fft.fft(arr[ispec*nfft:(ispec+1)*nfft])

    return arr_fft



assert NTIMES % NFFT == 0, "NFFT must divide NTIMES"

iref_ant = 5 # this is the reference antenna ID that everything is multiplied against
assert iref_ant < NANTS, "reference antenna must be included in the antennas"

# file name. Note we will have more than 1 file per obs
fname = sys.argv[1]
outname = "out.vis"

# "Load" the guppi file
# This does nothing except opening the file
g = guppi.Guppi(fname)

# These are the "visibilities"
# i.e. the cross-multiplication data products
vis = np.zeros((NBLOCKS, NANTS, NCHANS*NFFT, NPOLS*NPOLS), dtype=np.complex64)



for iblock in range(NBLOCKS):
    hdr, data = g.read_next_block() # read next hdr+data block 
    # data have shape of: (NANTS, NCHANS, NTIMES, NPOLS)
    # TODO: assert data.shape == (NANTS, NCHANS, NTIMES, NPOLS)

    ant_ref = data[iref_ant] # this is the reference antenna    

    for iant in range(NANTS):
        print(iant)
        ant = data[iant] # current antenna we need to correlate with reference
        # ant have shape of (NCHANS, NTIMES, NPOLS)
        # note if iant == iref_ant, we end up with zero phase

        for ichan in range(NCHANS):
            ch_ant    = ant[ichan]
            ch_refant = ant[ichan]

            ch_ant_fftd_x    = do_npoint_spec(ch_ant[:,0], NFFT)
            ch_ant_fftd_y    = do_npoint_spec(ch_ant[:,1], NFFT)
            ch_refant_fftd_x = do_npoint_spec(ch_refant[:,0], NFFT)
            ch_refant_fftd_y = do_npoint_spec(ch_refant[:,1], NFFT)
            # all the above will have shape = (NTIME//NFFT, NFFT)

            for ifft in range(NFFT):
                # note that np.vdot(a, b) = np.dot(np.conj(a), b)
                #                         = np.sum(np.conj(a) * b)

                # Here we are summing over every block. We will not subdivide each block
                # which is fine, because each block is ~16 ms

                # XX
                vis[iblock, iant, ichan*NFFT+ifft, 0] =\
                    np.vdot(ch_ant_fftd_x[:,ifft], ch_refant_fftd_x[:,ifft])

                # YY
                vis[iblock, iant, ichan*NFFT+ifft, 1] =\
                   np.vdot(ch_ant_fftd_y[:,ifft], ch_refant_fftd_y[:,ifft])

                # XY
                vis[iblock, iant, ichan*NFFT+ifft, 2] =\
                   np.vdot(ch_ant_fftd_x[:,ifft], ch_refant_fftd_y[:,ifft])

                # YX
                vis[iblock, iant, ichan*NFFT+ifft, 3] =\
                    np.vdot(ch_ant_fftd_y[:,ifft], ch_refant_fftd_x[:,ifft])


# Now we're done, write data to disk
vis.tofile(outname)