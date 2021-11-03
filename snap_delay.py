#from . import snap_control
#from ..ATATools import ata_control as ac

from SNAPobs import snap_control, snap_config
from ATATools import ata_control as ac

import numpy as np
import pandas as pd
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
import astropy.units as u
from astropy.time import Time
import datetime,time
from pytz import timezone
import pymap3d


# location of ATA
LAT0 = 40+49.0/60+3.0/3600
LON0 = -(121+28.0/60+24.0/3600)
H0   = 1008

C = 299792458 #m/s

MAX_SAMPLE_DELAY = 16384 - 1
MID_SAMPLE_DELAY = MAX_SAMPLE_DELAY // 2

LOCATION = EarthLocation(lon=LON0 * u.deg, lat=LAT0 * u.deg, height=H0 * u.m)

ATA_CFG      = snap_config.get_ata_cfg()
ATA_SNAP_TAB = snap_config.get_ata_snap_tab()


def check_if_valid_ants(ant_list):
    valid_ants = ATA_CFG['ACTIVE_ANTS']
    mask = [ant in valid_ants for ant in ant_list]
    if not(all(mask)):
        raise RuntimeError("Antennas provided: %s\n"\
                "not all included in valid antennas: %s"\
                %(ant_list, valid_ants))


def get_fengs(ant_list):
    # initialise snaps
    check_if_valid_ants(ant_list)
    snap_names = []

    sub_tab = ATA_SNAP_TAB[ATA_SNAP_TAB.ANT_name.isin(ant_list)]
    snap_names = list(sub_tab.snap_hostname)

    # better put them in a dictionary
    snaps = {}
    s = snap_control.init_snaps(snap_names)
    #for iant,ant in enumerate(ant_list):
    #    snaps[ant] = s[iant]
    for isnap_name, snap_name in enumerate(snap_names):
        ant = (sub_tab.ANT_name[sub_tab.snap_hostname == snap_name]).values[0]
        snaps[ant] = s[isnap_name]

    return snaps


def delay(ant_list, ref_ant, fixed_delays, source_ra, source_dec,
        delay_update_time=10, look_forward_time=7):
    """
    Basic snap delay engine. Goes in infinite loop, applying delays

    Parameters
    ----------
    ant_list : array-like
        list of antennas to apply delay on
    ref_ant : str
        reference antenna
    fixed_delays : array-like
        list of fixed delays to be applied. Values in nanoseconds
    source_ra : float
        right ascension of source, in hours
    source_dec : float
        declination of source, in degrees
    delay_update_time : float
        update time in seconds of delay engine [default: 10]
    look_forward_time : float
        time in seconds at which the delays are applied in the future 
        [default: 10]
    """

    snaps = get_fengs(ant_list)

    # get antenna geographical positions
    ant_pos = ac.get_ant_pos(ant_list)
    ant_pos_df = pd.DataFrame.from_dict(ant_pos, orient='index', columns=['N', 'E', 'U'])

    ecef = pymap3d.ned2ecef(ant_pos_df['N'], ant_pos_df['E'], -ant_pos_df['U'],
        LAT0, LON0, H0)
    ecef = pd.DataFrame(np.row_stack(ecef), columns=ant_list, index=None)

    blines = ecef[ref_ant].values - ecef.values.T

    fixed_delays = [float(d) * 1e-9 for d in fixed_delays]

    
    ra  = source_ra * 360 / 24.
    dec = source_dec

    source = SkyCoord(ra, dec, unit="deg")



    while True:
        #for ant, snap in snaps.items():
        #    print(ant)
        #    print(snap.get_delay(0))
        #    print(snap.get_pending_delay(0))
        tnow = time.time() # unix time now
        t_delay = int(tnow + look_forward_time) # unix time to apply delays

        astro_time = Time(t_delay, format='unix', 
                location=LOCATION)

        source_xyz = source.transform_to(ITRS(obstime = astro_time)).cartesian
        delay_lengths = np.dot(blines, source_xyz.get_xyz()) #in meters
        delay_seconds = delay_lengths / C # in seconds
        delay_seconds += np.array(fixed_delays)

        delay_snaps = delay_seconds * 2.048e9 # in units of FPG samples
        delay_snaps = np.array(delay_snaps)
        delay_snaps_int = np.round(delay_snaps).astype(np.int)
        # now compensate for delays
        delay_snaps_int = MID_SAMPLE_DELAY - delay_snaps_int

        print("Unix time to apply delays:", t_delay)
        print("Delays (in ADC units):", delay_snaps)
        print("Delays (in ADC units) (rounded):", delay_snaps_int, flush=True)

        # setting delays:
        for iant, ant in enumerate(ant_list):
            xpol, ypol = [delay_snaps_int[iant]]*2
            snaps[ant].set_delays((xpol,ypol), load_time=t_delay)

        time.sleep(delay_update_time)


def reset_delays(ant_list):
    snaps = get_fengs(ant_list)
    for snap in snaps.values():
        snap.set_delays((0,0))


if __name__ == "__main__":
    import atexit
    #ant_list = ["1a", "1f", "1c", "3c", "2b", "5c"]
    #ant_list = ["2b", "3c", "4g"]
    #ant_list = ["1a", "1f", "4g", "5c", "1c"]
    ant_list = ["1a", "1f", "4g", "5c", "1c", "2b", "2h", "1h", "1k", "4j", "2a", "3c"]
    print(ant_list)
    ref_ant = "1c"
    print("Reference antenna: %s" %ref_ant)

    #ra, dec = 5.576, 22.014
    ra, dec = 3.33, 41.512 #3c84
    print("Source coordinates: ra, dec = %.3f, %.3f"
            %(ra, dec))
    print("base delay applied to all:", MID_SAMPLE_DELAY)

    fixed_delays = [0]*len(ant_list)
    #fixed_delays = [-149, -171, -904, -1650, 0, -233, -244, -178, -260, -984]
    fixed_delays = [-149, -171, -904, -1650, 0, -233, -244, -178, -260. -984, -347, -250]
    print("Fixed delays:", fixed_delays)

    atexit.register(reset_delays, ant_list)
    #delay(ant_list, ref_ant, fixed_delays, ra, dec) 
