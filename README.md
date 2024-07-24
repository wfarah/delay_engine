# delay_engine
First go at delay engine for the ATA

# Requirements
- [phasing.py](https://github.com/daniestevez/ata_interferometry/blob/main/postprocess/phasing.py) is written by Dani Estevez
- astropy
- pandas
- numpy
- [guppi package](https://github.com/wfarah/guppi) for the correlator

# Example:
The below will print to screen delay, delay_rate, phase and phase_rate for the source CasA

`python print_t_dt.py -source_ra 23.391 -source_dec 58.808 -lo 1400`

# NOTE:
The delay engine has now officially been placed in the `ATA-Utils` repository

===    This repo is deprecated    ===
-----------------------------------------
