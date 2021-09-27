import numpy as np


# NOTE: the below numbers might change because of constraints on the FPGA resources. 
# We'll know better in the near future

NBEAMS=4 #number of beams to form on sky
NANTS=20 #number of antennas
NCHANS=384 #number of PFB channels being processed by node
NTIME=8750 #number of time samples per block to form ~35 ms of data (assuming 4k PFB)
NPOLS=2 #number of polarisations

TFACT=10 #number of integrations post beamforming
OUTPOLS=4 #output polarization (to send to CPU); 4=XX,YY,XY,YX

print("NPOLS: %i, NANTS: %i, NCHANS: %i, NTIME: %i, NPOLS: %i" 
        %(NBEAMS, NANTS, NCHANS, NTIME, NPOLS))
print("-"*79)

# simulate 8bit numbers
# I'm taking them to be 8bit + 8bit complex numbers = int16
input_data = np.random.randint(-int(2**16/2), int(2**16/2),
        size=(NANTS,NCHANS,NTIME,NPOLS)).astype(np.int16)


# simulate complex phasors
phasors = np.zeros(shape=(NBEAMS, NANTS, NCHANS, NPOLS),
        dtype=np.complex64) # assume a certain weights

phasors[:] = np.random.random(size=phasors.shape) +\
        1j*np.random.random(size=phasors.shape)


print("Dimension of input_data")
print(input_data.shape)
print("Dimension of phasors")
print(phasors.shape)

##############################################################################

# I assume the above to be what the hashpipe/phasor
# code provide me
# Now here the data is sent to the GPU

# I want to expand the data to 32bit complex
# not sure if tensor cores can do complex operations
input_data_expanded = (input_data.view(np.int8)[::2] +\
        1j*input_data.view(np.int8)[1::2]).astype(np.complex64)
input_data_expanded = input_data_expanded.reshape(input_data.shape)

print(input_data_expanded.shape)

output_data = np.zeros(shape=(NBEAMS+1, NCHANS, NTIME, NPOLS), dtype=np.complex64)

# Now beamforming begins
# The for loop can be replaced by a single cuBLAS tensor operation
for ibeam in range(NBEAMS):
    print("Doing beam: %i" %ibeam)
    phased = input_data_expanded * phasors[ibeam][..., np.newaxis,:] # add a new axis for the phasors... (20,384,8750,2) (20,384,2)
    # now we coherently sum across the NANT axis, and store the data in the output buffer:
    output_data[ibeam] = phased.sum(axis=0)

# This is for the incoherent sum
# stored as the last beam
print("Doing incoherent sum")
output_data[-1] = np.abs(input_data_expanded).sum(axis=0)


# Here we produced our beams, but need to send data back to the CPU
# We'll send detected "coherency" data product, i.e. [XX,YY,XY,YX]

print("Doing detection")
assert NTIME%TFACT == 0, "integration length must divide the time samples in block"
out_nsamp=NTIME//TFACT
output_data_detected = np.zeros(shape=(NBEAMS+1, NCHANS, out_nsamp, OUTPOLS), dtype=np.float32)
#output_data_detected is what will be sent to the CPU

for ibeam in range(NBEAMS):
    print("Doing beam: %i" %ibeam)
    for ichan in range(NCHANS):
        for isamp in range(out_nsamp):
            x = output_data[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 0] #just to make code more visible
            y = output_data[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 1] #just to make code more visible

            output_data_detected[ibeam, ichan, isamp, 0] = np.abs(np.sum(x * np.conj(x))) #XX
            output_data_detected[ibeam, ichan, isamp, 1] = np.abs(np.sum(y * np.conj(y))) #YY
            output_data_detected[ibeam, ichan, isamp, 2] = np.abs(np.sum(x * np.conj(y))) #XY
            output_data_detected[ibeam, ichan, isamp, 3] = np.abs(np.sum(y * np.conj(x))) #YX

# incoherent sum will be zeros at this point, we'll figure out what to do with it
