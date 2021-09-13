import numpy as np

from phasing import compute_uvw, compute_antenna_gainphase
import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
from astropy.time import Time,TimeDelta
import pandas as pd
import time, os

import argparse


ANTNAMES = ['1A', '1F', '1C', '1K', '1H', '1E', '1G', 
        '2A', '2B', '2C', '2H', '2E', '2J', '2K', '2L', '2M',
        '3C', '3D', '3L', 
        '4G', '4J',
        '5B', '5C']
        
WD = os.path.realpath(os.path.dirname(__file__))
DEFAULT_ANT_ITRF = os.path.join(WD, "ant_itrf.txt")
DEFAULT_REF_ANT = "1C"


def main():
    parser = argparse.ArgumentParser(description='print w coordinate')
    parser.add_argument('-source_ra', type=float, required=True, 
        help = 'Source RA [decimal hours]')
    parser.add_argument('-source_dec', type=float, required=True,
        help = 'Source Dec [degrees]')
    parser.add_argument('-refant', type=str,
        default = DEFAULT_REF_ANT,
        help = 'Reference antenna')
    parser.add_argument('-itrf', type=str,
        default = DEFAULT_ANT_ITRF, required = False,
        help = 'ITRF file [default: %s]' %DEFAULT_ANT_ITRF)

    args = parser.parse_args()

    # Get ITRF coordinates of the antennas
    itrf = pd.read_csv(args.itrf, names=['x', 'y', 'z'], header=None, skiprows=1)
    itrf_sub = itrf.loc[ANTNAMES]

    refant = args.refant.upper()
    irefant = itrf_sub.index.values.tolist().index(refant)

    ra = args.source_ra * 360 / 24.
    dec = args.source_dec
    source = SkyCoord(ra, dec, unit='deg')

    while True:
        t = np.floor(time.time())
        tts = np.arange(-2, 10) + t

        ts = Time(tts, format='unix')

        uvw1 = compute_uvw(ts[0],  source, itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])
        uvw2 = compute_uvw(ts[-1], source, itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])

        w1 = uvw1[...,2]
        w2 = uvw2[...,2]

        delay1 = w1/const.c.value
        delay2 = w2/const.c.value

        rate = (delay2 - delay1) / (tts[-1] - tts[0])
        print("Time 0: ", ts[0], "Time 1: ", ts[-1])
        print(delay1)
        print(delay2)
        print(rate)

        print("")
        time.sleep(5)


if __name__ == "__main__":
    main()
