#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:31:45 2021

@author: dubay.11
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
from astropy.table import Table

apokasc_file = 'APOKASC_cat_v6.7.2.fits'
starhorse_file = 'APOGEE_DR17_EDR3_STARHORSE_v2.fits'
astroNN_file = 'apogee_astroNN-DR17.fits'
bacchus_file = 'dr17_nc_abund_v1_0.fits'


def decode(df):
    """
    Decode DataFrame with byte strings into ordinary strings.

    """
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def rms(array, axis=None):
    """
    Return the root-mean-square of a given array.

    Parameters
    ----------
    axis : int or NoneType
        Axis along which to calculate rms (passed to numpy.mean()).

    """
    return np.sqrt(np.mean(array**2, axis=axis))


def quad_add(arr1, arr2):
    """
    Add input arrays in quadrature.

    """
    return np.sqrt(arr1**2 + arr2**2)


def import_catalogs(data_path, verb=True):
    """
    Import and combine APOKASC-2 and APOGEE DR17 value added catalogs.

    """
    data_path = Path(data_path)
    # APOKASC-2 catalog
    if verb:
        print('Importing APOKASC catalog...')
    data = Table.read(data_path / apokasc_file, format='fits')
    apokasc_df = decode(data.to_pandas())
    # Relevant columns
    apokasc_cols = ['2MASS_ID', 'APOKASC2_AGE', 'APOKASC2_AGE_MERR',
                    'APOKASC2_AGE_PERR', 'DR16_ALP_M_COR', 'DR16_ALP_M_COR_ERR',
                    'DR16_FE_H', 'DR16_FE_H_ERR', 'DR16_M_H_COR',
                    'DR16_M_H_COR_ERR', 'DR16_O_FE', 'DR16_O_FE_ERR',
                    'DR16_C_FE', 'DR16_C_FE_ERR', 'DR16_N_FE', 'DR16_N_FE_ERR',
                    'DR16_ASPCAP_SNR', 'APOKASC2_LOGG', 'APOKASC2_LOGG_RANERR',
                    'GAIA_PARALLAX_DR2', 'GAIA_PARALLAX_ERROR_DR2',
                    'GAIA_PHOT_G_MEAN_MAG_DR2', 'GAIA_PHOT_BP_MEAN_MAG_DR2',
                    'GAIA_PHOT_RP_MEAN_MAG_DR2', 'APOKASC2_AV',
                    'DR16_LOGG_COR', 'DR16_LOGG_COR_ERR', 'DR16_TEFF_COR',
                    'DR16_TEFF_COR_ERR', 'DR16_TI_FE', 'DR16_TI_FE_ERR']

    # astroNN DR17 catalog
    if verb:
        print('Importing astroNN DR17 catalog...')
    data = Table.read(data_path / astroNN_file, format='fits')
    astroNN_df = decode(data.to_pandas())
    astroNN_cols = ['APOGEE_ID', 'age_lowess_correct', 'age_total_error',
                    'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'C_H', 'C_H_ERR',
                    'N_H', 'N_H_ERR', 'O_H', 'O_H_ERR', 'TI_H', 'TI_H_ERR',
                    'FE_H', 'FE_H_ERR']

    # StarHorse DR17 catalog
    if verb:
        print('Importing StarHorse DR17 catalog...')
    data = Table.read(data_path / starhorse_file, format='fits')
    starhorse_df = decode(data.to_pandas())
    starhorse_cols = ['APOGEE_ID', 'met16', 'met50', 'met84']

    # BACCHUS neutron capture abundance catalog
    # if verb:
    #     print('Importing BACCHUS catalog...')
    # data = Table.read(data_path / bacchus_file, format='fits', hdu=1)
    # multd_names = [name for name in data.colnames if len(data[name].shape) <= 1]
    # bacchus_df = decode(data[multd_names].to_pandas())
    # bacchus_cols = ['APOGEE_ID', 'C12C13', 'C12C13_ERR_MEAS', 'C12C13_ERR_EMP']

    # Consolidate into single DataFrame
    if verb:
        print('Combining datasets...')
    cat = apokasc_df[apokasc_cols].copy()
    cat.rename(columns={'2MASS_ID': 'APOGEE_ID'}, inplace=True)
    cat = cat.join(astroNN_df[astroNN_cols].set_index('APOGEE_ID'),
                   on='APOGEE_ID', how='outer', rsuffix='_astroNN')
    cat = cat.join(starhorse_df[starhorse_cols].set_index('APOGEE_ID'),
                   on='APOGEE_ID', how='outer', rsuffix='_StarHorse')
    # cat = cat.join(bacchus_df[bacchus_cols].set_index('APOGEE_ID'),
    #                      on='APOGEE_ID', how='outer', rsuffix='_BACCHUS')

    # Clean up
    if verb:
        print('Cleaning up...')
    cat.replace([np.inf, -np.inf, -9999., -9999.99, -999., -999.99],
                np.nan, inplace=True)
    cat.dropna(how='all', inplace=True)
    # TODO better way to deal with astroNN duplicates
    # (same APOGEE_ID, different LOCATION_ID)
    cat.drop_duplicates(subset='APOGEE_ID', inplace=True)
    cat.set_index('APOGEE_ID', inplace=True)
    # Rename columns
    mapper = dict([(col, 'ASTRONN_'+col) for col in astroNN_cols[3:]])
    cat.rename(columns=mapper, inplace=True)
    cat.rename(columns={'age_lowess_correct': 'ASTRONN_AGE',
                        'age_total_error': 'ASTRONN_AGE_ERR',
                        'met50': 'STARHORSE_M_H'}, inplace=True)
    # Combine columns
    if verb:
        print('Combining columns...')
    cat['STARHORSE_M_H_MERR'] = cat['STARHORSE_M_H'] - cat['met16']
    cat['STARHORSE_M_H_PERR'] = cat['met84'] - cat['STARHORSE_M_H']
    cat.drop(['met16', 'met84'], axis='columns', inplace=True)
    cat['ASTRONN_C_N'] = cat['ASTRONN_C_H'] - cat['ASTRONN_N_H']
    cat['ASTRONN_C_N_ERR'] = quad_add(cat['ASTRONN_C_H_ERR'],
                                      cat['ASTRONN_N_H_ERR'])
    cat['ASTRONN_O_FE'] = cat['ASTRONN_O_H'] - cat['ASTRONN_FE_H']
    cat['ASTRONN_O_FE_ERR'] = quad_add(cat['ASTRONN_O_H_ERR'],
                                       cat['ASTRONN_FE_H_ERR'])
    cat['ASTRONN_TI_FE'] = cat['ASTRONN_TI_H'] - cat['ASTRONN_FE_H']
    cat['ASTRONN_TI_FE_ERR'] = quad_add(cat['ASTRONN_TI_H_ERR'],
                                        cat['ASTRONN_FE_H_ERR'])
    cat['DR16_C_N'] = cat['DR16_C_FE'] - cat['DR16_N_FE']
    cat['DR16_C_N_ERR'] = quad_add(cat['DR16_C_FE_ERR'], cat['DR16_N_FE_ERR'])
    if verb:
        print('Done!')

    return cat


def get_discrepant_ages(cat, diff_cut=-5):
    """
    Identify targets where the astroNN reported age differs substantially
    from APOKASC-2.

    Parameters
    ----------
    cat : DataFrame
        Combine catalog of at least APOKASC-2 and astroNN-DR17
    diff_cut : float
        Age difference between catalogs defined as discrepant

    """
    # Select targets with ages from both APOKASC and astroNN
    ages = cat[(pd.notna(cat['APOKASC2_AGE'])) &
               (pd.notna(cat['APOKASC2_AGE_PERR'])) &
               (pd.notna(cat['ASTRONN_AGE'])) &
               (pd.notna(cat['ASTRONN_AGE_ERR']))].copy()
    ages['AGE_DIFF'] = ages['ASTRONN_AGE'] - ages['APOKASC2_AGE']
    ages['AGE_DIFF_MERR'] = quad_add(ages['ASTRONN_AGE_ERR'],
                                     ages['APOKASC2_AGE_MERR'])
    ages['AGE_DIFF_PERR'] = quad_add(ages['ASTRONN_AGE_ERR'],
                                     ages['APOKASC2_AGE_PERR'])
    low_age = ages[ages['AGE_DIFF'] < diff_cut].copy()
    return ages, low_age


def get_discrepant_metallicities(cat, diff_cut=-0.5):
    """
    Identify targets where the astroNN reported [Fe/H] differs substantially
    from ASPCAP DR16

    """
    # Select targets with metallicities from both ASPCAP-DR16 and astroNN-DR17
    metals = cat[(pd.notna(cat['ASTRONN_FE_H'])) &
                 (pd.notna(cat['ASTRONN_FE_H_ERR'])) &
                 (pd.notna(cat['DR16_M_H_COR'])) &
                 (pd.notna(cat['DR16_M_H_COR_ERR']))].copy()
    metals['M_H_DIFF'] = metals['ASTRONN_FE_H'] - metals['DR16_M_H_COR']
    metals['M_H_DIFF_ERR'] = quad_add(metals['ASTRONN_FE_H_ERR'],
                                      metals['DR16_M_H_COR_ERR'])
    low_fe = metals[metals['M_H_DIFF'] < diff_cut].copy()
    return metals, low_fe


def get_gaia_cmd(cat):
    # Select only positive parallaxes
    cmd = cat[cat['GAIA_PARALLAX_DR2'] > 0].copy()
    # Use APOKASC2 A_V extinction where possible
    # cmd['APOKASC2_AV'] = cmd['APOKASC2_AV'].replace(np.nan, 0)
    # Calculate distance modulus from Gaia parallaxes (given in mas)
    cmd['GAIA_DIST_MOD'] = 5*np.log10((1e-3 * cmd['GAIA_PARALLAX_DR2'])**-1) - 5 #+ cmd['APOKASC2_AV']
    # Absolute Gaia G-band magnitude
    cmd['GAIA_ABS_MAG'] = cmd['GAIA_PHOT_G_MEAN_MAG_DR2'] - cmd['GAIA_DIST_MOD']
    # Gaia BP - RP color
    cmd['GAIA_COLOR'] = cmd['GAIA_PHOT_BP_MEAN_MAG_DR2'] - cmd['GAIA_PHOT_RP_MEAN_MAG_DR2']
    return cmd

def scatter_hist(ax, x, y, xlim=None, ylim=None, log_norm=True, cmap='gray',
                 cmin=10, vmin=None, vmax=None):
    """
    Combination scatter plot + 2D histogram for denser regions.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    xlim : TYPE, optional
        DESCRIPTION. The default is None.
    ylim : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    if not xlim:
        xlim = (np.min(x), np.max(x))
    if not ylim:
        ylim = (np.min(y), np.max(y))
    xbins = np.linspace(xlim[0], xlim[1], num=50, endpoint=True)
    ybins = np.linspace(ylim[0], ylim[1], num=50, endpoint=True)
    if log_norm:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
    ax.scatter(x, y, c='k', s=0.5)
    ax.hist2d(x, y, bins=[xbins, ybins], cmap=cmap, norm=norm, cmin=cmin)
        
    dx = xbins[1] - xbins[0]
    dy = ybins[1] - ybins[0]
    ax.set_xlim((xlim[0] - 2*dx, xlim[1] + 2*dx))
    ax.set_ylim((ylim[0] - 2*dy, ylim[1] + 2*dy))
    return ax


def plot_rms_err(ax, xerrs, yerrs, loc='upper left'):
    # RMS of error arrays
    # If array is 2D, that means the positive and negative errors are separate
    xerrs = np.array(xerrs)
    yerrs = np.array(yerrs)
    # drop nan
    xerrs = xerrs[~np.isnan(xerrs)]
    yerrs = yerrs[~np.isnan(yerrs)]
    if len(xerrs.shape) > 1:
        xerr = rms(xerrs, axis=0)[:,np.newaxis]
    else:
        xerr = np.array([rms(xerrs)])
    if len(yerrs.shape) > 1:
        yerr = rms(yerrs, axis=0)[:,np.newaxis]
    else:
        yerr = np.array([rms(yerrs)])
    
    # Set error bar location
    xlim = ax.get_xlim()
    xpad = 0.05 * (xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    ypad = 0.05 * (ylim[1] - ylim[0])
    if type(loc) == tuple or type(loc) == list or type(loc) == np.ndarray:
        x, y = loc
    elif loc == 'upper right':
        x = xlim[1] - xpad - xerr[-1]
        y = ylim[1] - ypad - yerr[-1]
    elif loc == 'lower right':
        x = xlim[1] - xpad - xerr[-1]
        y = ylim[0] + ypad + yerr[0]
    elif loc == 'lower left':
        x = xlim[0] + xpad + xerr[0]
        y = ylim[0] + ypad + yerr[0]
    else:
        x = xlim[0] + xpad + xerr[0]
        y = ylim[1] - ypad - yerr[-1]
    
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, color='k', capsize=2)
    return ax
