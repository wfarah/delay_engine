import numpy as np

from phasing import compute_uvw, compute_antenna_gainphase
import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
from astropy.time import Time,TimeDelta
import pandas as pd
import time, os

import argparse

from SNAPobs import snap_control, snap_config

import atexit


ANTNAMES = ['1C', '1K', '1H', '1E', '1G', 
        '2A', '2B', '2C', '2H', '2E', '2J', '2K', '2L', '2M',
        '3C', '3D', '3L', 
        '4G', '4J', '5B']
ALL_LO = ["a", "b", "c", "d"]
        
WD = os.path.realpath(os.path.dirname(__file__))
DEFAULT_ANT_ITRF = os.path.join(WD, "ant_itrf.txt")
DEFAULT_DELAYS = os.path.join(WD, "delays.txt")
DEFAULT_PHASES = os.path.join(WD, "phases.txt")
DEFAULT_REF_ANT = "1C" #1c is a performant antenna
MAX_SAMP_DELAY = 16384
CLOCK_FREQ = 2.048e9 #Gsps
ADC_SAMP_TIME = 1/CLOCK_FREQ

MAX_DELAY = MAX_SAMP_DELAY * ADC_SAMP_TIME #seconds
ADVANCE_TIME = MAX_DELAY/2

BANDWIDTH = CLOCK_FREQ/2. #Hz


def main():
    parser = argparse.ArgumentParser(
        description = 'Control and apply delay engine on RFSoC-boards')
    parser.add_argument('-source_ra', type=float, required=True, 
        help = 'Source RA [decimal hours]')
    parser.add_argument('-source_dec', type=float, required=True,
        help = 'Source Dec [degrees]')
    parser.add_argument('-lo', type=str, required=True,
        help = 'LO letter [a, b, c, d]')
    parser.add_argument('-lofreq', type=float, required=True,
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
    parser.add_argument('-phases', required = False, type=str,
        default = DEFAULT_PHASES,
        help = 'Frequency-dependent phases file to use [default: %s]'\
            %DEFAULT_PHASES)
    parser.add_argument('-noadvance', action='store_true', default=False,
        help = 'Do not advance the delay engine by the fixed term')
    parser.add_argument('-nophase', action='store_true', default=False,
        help = 'Do not apply phase solution')
    parser.add_argument('-zero', action='store_true', default=False,
        help = 'Simply apply zero delay/phase, ignore everything')

    # Parse cmd line arguments
    args = parser.parse_args()

    assert args.lo in ALL_LO,\
            "Input correct LO letter (input: %s)" %args.lo


    fixed_delays_all = pd.read_csv(args.fixed, sep=" ", index_col=None)
    phases_all = pd.read_csv(args.phases, sep=" ", index_col=False)


    # Select LO
    rfsoc_tab = snap_config.ATA_SNAP_TAB[
            snap_config.ATA_SNAP_TAB.LO == args.lo]
    rfsoc_hostnames = []
    fixed_delays_x = []
    fixed_delays_y = []

    phases_x = []
    phases_y = []

    # retrieve names of rfsoc instances
    for ant in np.char.lower(np.array(ANTNAMES)):
        if ant not in list(rfsoc_tab.ANT_name):
            raise RuntimeError("Antenna %s not in the rfsoc configuration!" %ant)
        if ant not in list(fixed_delays_all.values[:,0]):
            raise RuntimeError("Antenna %s not in the fixed delays list!" %ant)
        rfsoc_hostnames.append(
                rfsoc_tab[rfsoc_tab.ANT_name == ant].snap_hostname.values[0])
        fixed_delays_x.append(
                fixed_delays_all[fixed_delays_all.values[:,0] == ant].values[:,1][0])
        fixed_delays_y.append(
                fixed_delays_all[fixed_delays_all.values[:,0] == ant].values[:,2][0])

        phases_x.append(phases_all[ant+"x"])
        phases_y.append(phases_all[ant+"y"])

    fixed_delays_x = np.array(fixed_delays_x)*1e-9
    fixed_delays_y = np.array(fixed_delays_y)*1e-9


    # initialise the rfsoc feng objects
    rfsocs = snap_control.init_snaps(rfsoc_hostnames)
    for rfsoc in rfsocs:
        rfsoc.fpga.get_system_information(snap_config.ATA_CFG['RFSOCFPG'])

    if not args.nophase:
        for rfsoc, phase_calx, phase_caly in zip(rfsocs, phases_x, phases_y):
            rfsoc.set_phase_calibration(0, -phase_calx)
            rfsoc.set_phase_calibration(1, -phase_caly)


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

    log = open("delay_engine.log", "a")
    log.write("rfsoc_engine unix delay delay_rate phase phase_rate\n")
    atexit.register(log.close)

    lo_freq = args.lofreq

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
        delay1_x = fixed_delays_x + (w1/const.c.value)
        delay2_x = fixed_delays_x + (w2/const.c.value)
        delay1_y = fixed_delays_y + (w1/const.c.value)
        delay2_y = fixed_delays_y + (w2/const.c.value)


        # advance all the B-engines forward in time
        if not args.noadvance:
            delay1_x += ADVANCE_TIME
            delay2_x += ADVANCE_TIME
            delay1_y += ADVANCE_TIME
            delay2_y += ADVANCE_TIME

        # make sure we're not providing large delays
        assert np.all((delay1_x < MAX_DELAY) & (delay1_x > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY
        assert np.all((delay2_x < MAX_DELAY) & (delay2_x > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY
        assert np.all((delay1_y < MAX_DELAY) & (delay1_y > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY
        assert np.all((delay2_y < MAX_DELAY) & (delay2_y > 0)),\
                "Delays are not within 0 and max_delay: %.2e" %MAX_DELAY

        # Compute the delay rate in s/s
        rate_x = (delay2_x - delay1_x) / (tts[-1] - tts[0])
        rate_y = (delay2_y - delay1_y) / (tts[-1] - tts[0])
        
        print(ANTNAMES)
        print("")

        # Print values to screen, for now
        print("Delay [ns]")
        print(delay1_x*1e9)
        print(delay1_y*1e9)
        print("")
        print("Delay rate [ns/s]")
        print(rate_x*1e9)
        print(rate_y*1e9)

        # Using LO - BW/2 for fringe rate
        phase_x      = -2 * np.pi * (lo_freq*1e6 - BANDWIDTH/2.) * delay1_x
        phase_rate_x = -2 * np.pi * (lo_freq*1e6 - BANDWIDTH/2.) * rate_x
        phase_y      = -2 * np.pi * (lo_freq*1e6 - BANDWIDTH/2.) * delay1_y
        phase_rate_y = -2 * np.pi * (lo_freq*1e6 - BANDWIDTH/2.) * rate_y

        print("")
        print("Phase [rad]")
        print(phase_x)
        print(phase_y)

        print("")
        print("Phase rate [rad/s]")
        print(phase_rate_x)
        print(phase_rate_y)

        print("="*79)

        if args.zero:
            print("Zeroing all delays/phase")
            delay1_x = np.zeros_like(delay1_x)
            rate_x = np.zeros_like(rate_x)
            phase_x = np.zeros_like(phase_x)
            phase_rate_x = np.zeros_like(phase_rate_x)
            delay1_y = np.zeros_like(delay1_y)
            rate_y = np.zeros_like(rate_y)
            phase_y = np.zeros_like(phase_y)
            phase_rate_y = np.zeros_like(phase_rate_y)

        for i,rfsoc in enumerate(rfsocs):
            rfsoc.set_delay_tracking(
                    [delay1_x[i]*1e9,     delay1_y[i]*1e9], 
                    [rate_x[i]*1e9,       rate_y[i]*1e9],
                    [phase_x[i],      phase_y[i]],
                    [phase_rate_x[i], phase_rate_y[i]],
                    load_time = int(ts[0].unix),
                    invert_band=False
                    )
            log.write("%s %i %.6f %.6f %.6f %.6f\n" \
                    %(rfsoc.host, int(ts[0].unix),
                        delay1_x[i]*1e9, rate_x[i]*1e9, phase_x[i], phase_rate_x[i]))

        time.sleep(10)


if __name__ == "__main__":
    main()
