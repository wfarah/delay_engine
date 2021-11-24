import numpy as np

from phasing import compute_uvw, compute_antenna_gainphase
import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
from astropy.time import Time,TimeDelta
import pandas as pd
import time, os

import argparse

from SNAPobs import snap_control, snap_config


ANTNAMES = ['1C', '1K', '1H', '1E', '1G', 
        '2A', '2B', '2C', '2H', '2E', '2J', '2K', '2L', '2M',
        '3C', '3D', '3L', 
        '4G', '4J', '5B']
        
WD = os.path.realpath(os.path.dirname(__file__))
DEFAULT_ANT_ITRF = os.path.join(WD, "ant_itrf.txt")
DEFAULT_DELAYS = os.path.join(WD, "delays.txt")
DEFAULT_REF_ANT = "1C" #1c is a performant antenna
MAX_SAMP_DELAY = 16384
CLOCK_FREQ = 2.048e9 #Gsps
ADC_SAMP_TIME = 1/CLOCK_FREQ

MAX_DELAY = MAX_SAMP_DELAY * ADC_SAMP_TIME #seconds
ADVANCE_TIME = MAX_DELAY/2


def main():
    parser = argparse.ArgumentParser(description='print w coordinate')
    parser.add_argument('-source_ra', type=float, required=True, 
        help = 'Source RA [decimal hours]')
    parser.add_argument('-source_dec', type=float, required=True,
        help = 'Source Dec [degrees]')
    parser.add_argument('-lo', type=float, required=True,
        help = 'LO frequency [MHz]')
    parser.add_argument('-refant', type=str,
        default = DEFAULT_REF_ANT,
        help = 'Reference antenna')
    parser.add_argument('-itrf', type=str,
        default = DEFAULT_ANT_ITRF, required = False,
        help = 'ITRF file [default: %s]' %DEFAULT_ANT_ITRF)
    parser.add_argument('-fixed', required = False, type=str,
        default = DEFAULT_DELAYS,
        help = 'Delay file to use [default: %s]' %DEFAULT_DELAYS)
    parser.add_argument('-noadvance', action='store_true', default=False,
        help = 'Do not advance the delay engine by the fixed term')
    parser.add_argument('-zero', action='store_true', default=False,
        help = 'Simply apply zero delay/phase, ignore everything')

    # Parse cmd line arguments
    args = parser.parse_args()


    fixed_delays_all = pd.read_csv(args.fixed, sep=" ", index_col=None)


    # just use LO b for now
    rfsoc_tab = snap_config.ATA_SNAP_TAB[snap_config.ATA_SNAP_TAB.LO == "b"]
    rfsoc_hostnames = []
    fixed_delays = []

    # retrieve names of rfsocs
    for ant in np.char.lower(np.array(ANTNAMES)):
        if ant not in list(rfsoc_tab.ANT_name):
            raise RuntimeError("Antenna %s not in the rfsoc configuration!" %ant)
        if ant not in list(fixed_delays_all.values[:,0]):
            raise RuntimeError("Antenna %s not in the fixed delays list!" %ant)
        rfsoc_hostnames.append(
                rfsoc_tab[rfsoc_tab.ANT_name == ant].snap_hostname.values[0])
        fixed_delays.append(
                fixed_delays_all[fixed_delays_all.values[:,0] == ant].values[:,1][0])

    fixed_delays = np.array(fixed_delays)*1e-9


    # initialise the rfsoc feng objects
    rfsocs = snap_control.init_snaps(rfsoc_hostnames)
    for rfsoc in rfsocs:
        rfsoc.fpga.get_system_information(snap_config.ATA_CFG['RFSOCFPG'])


    # Get ITRF coordinates of the antennas
    itrf = pd.read_csv(args.itrf, names=['x', 'y', 'z'], header=None, skiprows=1)
    itrf_sub = itrf.loc[ANTNAMES]

    # Select reference antenna
    refant = args.refant.upper()
    irefant = itrf_sub.index.values.tolist().index(refant)

    # Parse phase center coordinates
    ra = args.source_ra * 360 / 24.
    dec = args.source_dec
    source = SkyCoord(ra, dec, unit='deg')

    while True:
        t = np.floor(time.time())
        tts = [3, 20+3] # Interpolate between t=3 sec and t=20 sec
        tts = np.array(tts) + t

        ts = Time(tts, format='unix')

        # perform coordinate transformation to uvw
        uvw1 = compute_uvw(ts[0],  source, itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])
        uvw2 = compute_uvw(ts[-1], source, itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])

        # "w" coordinate represents the goemetric delay in light-meters
        w1 = uvw1[...,2]
        w2 = uvw2[...,2]

        # Add fixed delays + convert to seconds
        delay1 = fixed_delays + (w1/const.c.value)
        delay2 = fixed_delays + (w2/const.c.value)

        #delay1 = delay1
        #delay2 = delay2

        # advance all the B-engines forward in time
        if not args.noadvance:
            delay1 += ADVANCE_TIME
            delay2 += ADVANCE_TIME

        # make sure we're not providing large delays
        assert np.all((delay1 < MAX_DELAY) & (delay1 > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY
        assert np.all((delay2 < MAX_DELAY) & (delay2 > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY

        # Compute the delay rate in s/s
        rate = (delay2 - delay1) / (tts[-1] - tts[0])
        
        print(ANTNAMES)
        print("")

        # Print values to screen, for now
        print("Delay [ns]")
        print(delay1*1e9)
        #print(delay2*1e9)
        print("")
        print("Delay rate [ns/s]")
        print(rate*1e9)

        phase      = -2 * np.pi * args.lo*1e6 * delay1
        phase_rate = -2 * np.pi * args.lo*1e6 * rate

        print("")
        print("Phase [rad]")
        print(phase)

        print("")
        print("Phase rate [rad/s]")
        print(phase_rate)

        print("="*79)

        if args.zero:
            print("Zeroing all delays/phase")
            delay1 = np.zeros_like(delay1)
            rate = np.zeros_like(rate)
            phase = np.zeros_like(phase)
            phase_rate = np.zeros_like(phase_rate)

        for i,rfsoc in enumerate(rfsocs):
            rfsoc.set_delay_tracking(
                    [delay1[i]*1e9,     delay1[i]*1e9], 
                    [rate[i]*1e9,       rate[i]*1e9],
                    [phase[i],      phase[i]],
                    [phase_rate[i], phase_rate[i]],
                    load_time = int(ts[0].unix)
                    )

        time.sleep(10)


if __name__ == "__main__":
    main()
