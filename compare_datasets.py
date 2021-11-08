from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.table import Table

# =============================================================================
# DEFINITIONS AND SETTINGS
# =============================================================================

# Plot settings
plt.rc(('xtick', 'ytick'), direction='in')
plt.rc('xtick', top=True)
plt.rc('ytick', right=True)
plt.rc('font', family='STIXgeneral')

# Paths
data_dir = 'C:\\Users\\dubay.11\\Data\\APOGEE'
data_path = Path(data_dir)
apokasc_file = 'APOKASC_cat_v6.7.2.fits'
starhorse_file = 'APOGEE_DR17_EDR3_STARHORSE_v2.fits'
astroNN_file = 'apogee_astroNN-DR17.fits'


def decode(df):
    """
    Decode DataFrame with byte strings into ordinary strings.

    """
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def rms(array, arrmin=None, arrmax=None):
    """
    Return the root-mean-square of a given array

    """
    if arrmin:
        array = array[array > arrmin]
    if arrmax:
        array = array[array < arrmax]
    return np.sqrt(np.mean(array**2))

# =============================================================================
# IMPORT DATA
# =============================================================================

# APOKASC catalog
print('Importing APOKASC catalog...')
data = Table.read(data_path / apokasc_file, format='fits')
apokasc_df = decode(data.to_pandas())
apokasc_df['LOC_ID'] = apokasc_df['LOC_ID'].astype(int)
# Relevant columns
apokasc_cols = ['2MASS_ID', 'LOC_ID', 'APOKASC2_AGE', 'APOKASC2_AGE_MERR', 
                'APOKASC2_AGE_PERR', 'DR16_ALP_M_COR', 'DR16_ALP_M_COR_ERR', 
                'DR16_FE_H', 'DR16_FE_H_ERR', 'DR16_M_H_COR', 
                'DR16_M_H_COR_ERR']

# astroNN DR17 catalog
print('Importing astroNN DR17 catalog...')
data = Table.read(data_path / astroNN_file, format='fits')
astroNN_df = decode(data.to_pandas())
# astroNN_df.drop(astroNN_df[astroNN_df['LOCATION_ID'] < 0].index, inplace=True)
astroNN_cols = ['APOGEE_ID', 'LOCATION_ID', 'TEFF', 'TEFF_ERR', 'LOGG', 
                'LOGG_ERR', 'C_H', 'C_H_ERR', 'N_H', 'N_H_ERR', 'O_H', 
                'O_H_ERR', 'TI_H', 'TI_H_ERR', 'FE_H', 'FE_H_ERR', 
                'age_lowess_correct', 'age_total_error']

# StarHorse DR17 catalog
print('Importing StarHorse DR17 catalog...')
data = Table.read(data_path / starhorse_file, format='fits')
starhorse_df = decode(data.to_pandas())
starhorse_cols = ['APOGEE_ID', 'ASPCAP_ID', 'met16', 'met50', 'met84']

# Consolidate into single DataFrame
print('Combining datasets...')
main_df = apokasc_df[apokasc_cols].rename(columns={'2MASS_ID': 'APOGEE_ID', 
                                                   'LOC_ID': 'LOCATION_ID'})
main_df = main_df.join(astroNN_df[astroNN_cols].set_index(['APOGEE_ID', 'LOCATION_ID']), 
                       on=['APOGEE_ID', 'LOCATION_ID'], how='outer', rsuffix='_astroNN')
main_df = main_df.join(starhorse_df[starhorse_cols].set_index('APOGEE_ID'), 
                       on='APOGEE_ID', how='outer', rsuffix='_StarHorse')
# Clean up
main_df.dropna(how='all', inplace=True)
main_df.replace([np.inf, -np.inf, -9999., -9999.99, -999., -999.99], np.nan, 
                inplace=True)
main_df.set_index('APOGEE_ID', inplace=True)

# =============================================================================
# AGE COMPARISON
# =============================================================================

# Plot APOKASC age vs astroNN age
print('Plotting age vs age...')
# Select targets with ages from both APOKASC and astroNN
ages = main_df[(pd.notna(main_df['APOKASC2_AGE'])) & (pd.notna(main_df['age_lowess_correct']))]
fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
ax = axs[0]
ax.plot([0, 12], [0, 12], linestyle='--')
ax.scatter(ages['APOKASC2_AGE'], ages['age_lowess_correct'], s=0.5, c='k')
# plot RMS error
ax.errorbar(13, 2, 
            xerr=[[rms(ages['APOKASC2_AGE_MERR'])],
                  [rms(ages['APOKASC2_AGE_PERR'])]], 
            yerr=rms(ages['age_total_error']),
            color='k', capsize=2)
ax.set_xlabel('APOKASC2 Age [Gyr]')
ax.set_ylabel('astroNN Age [Gyr]')

# Histrogram of age differences
ax = axs[1]
xmin = -15
xmax = 10
bins = np.linspace(xmin, xmax, 26)
ax.hist(ages['age_lowess_correct'] - ages['APOKASC2_AGE'], 
        color='k', bins=bins, rwidth=0.9)
ax.grid(which='major', axis='y', color='w')
ax.set_xlim((xmin, xmax))
ax.set_yscale('log')
ax.set_xlabel('astroNN Age - APOKASC2 Age [Gyr]')
ax.set_ylabel('Count')
plt.savefig('age_comparison.png', dpi=300)
plt.show()

# List of IDs for stars with big age differences
age_discrep = ages[ages['age_lowess_correct'] - ages['APOKASC2_AGE'] < -5]
age_discrep.to_csv('age_discrep.csv')

# Age difference vs C/N
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.scatter(ages['age_lowess_correct'] - ages['APOKASC2_AGE'], 
           ages['C_H'] - ages['N_H'], c='k', s=1)
ax.set_xlabel('astroNN Age - APOKASC Age[Gyr]')
ax.set_ylabel('astroNN [C/N]')
plt.savefig('agediff_CN.png')
plt.show()

# =============================================================================
# ALPHA VS AGE
# =============================================================================

apokasc_alpha = apokasc_df[(apokasc_df['APOKASC2_AGE'] > 0) &
                           (apokasc_df['DR16_ALP_M_COR'] > -9999)]

alpha_rms_err = np.sqrt(np.mean(apokasc_alpha['DR16_ALP_M_COR']**2))
age_rms_err = [[np.sqrt(np.mean(apokasc_alpha['APOKASC2_AGE_MERR']**2))], 
               [np.sqrt(np.mean(apokasc_alpha['APOKASC2_AGE_PERR']**2))]]

zoom_xmin = 9
zoom_xmax = 18
zoom_ymin = -0.035
zoom_ymax = 0.005

print('Plotting alpha vs age...')
fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300, tight_layout=True)
ax = axs[0]
ax.scatter(apokasc_alpha['APOKASC2_AGE'], apokasc_alpha['DR16_ALP_M_COR'],
           c='k', s=1)
ax.errorbar(16, 0.35, xerr=age_rms_err, yerr=alpha_rms_err, 
            color='k', capsize=2)
rect = patches.Rectangle([zoom_xmin, zoom_ymin], zoom_xmax - zoom_xmin, 
                         zoom_ymax - zoom_ymin, color='k', fill=False)
ax.add_patch(rect)
ax.set_xlabel('APOKASC2 Age [Gyr]')
ax.set_ylabel('APOKASC DR16 [α/M]')

old_low_alpha = apokasc_alpha[(apokasc_alpha['APOKASC2_AGE'] > zoom_xmin) &
                              (apokasc_alpha['APOKASC2_AGE'] < zoom_xmax) &
                              (apokasc_alpha['DR16_ALP_M_COR'] > zoom_ymin) &
                              (apokasc_alpha['DR16_ALP_M_COR'] < zoom_ymax)]
ax = axs[1]
ax.scatter(old_low_alpha['APOKASC2_AGE'], old_low_alpha['DR16_ALP_M_COR'],
           c='k', s=6)
ax.errorbar(old_low_alpha['APOKASC2_AGE'], old_low_alpha['DR16_ALP_M_COR'],
            xerr=[old_low_alpha['APOKASC2_AGE_MERR'], 
                  old_low_alpha['APOKASC2_AGE_PERR']],
            yerr=old_low_alpha['DR16_ALP_M_COR_ERR'],
            c='k', elinewidth=1, linestyle='none', alpha=0.2)
ax.set_xlim((zoom_xmin, zoom_xmax))
ax.set_ylim((zoom_ymin, zoom_ymax))
ax.set_xlabel('APOKASC2 Age [Gyr]')
ax.set_ylabel('APOKASC DR16 [α/M]')
plt.savefig('alpha_age.png', dpi=300)
plt.show()

# =============================================================================
# METALLICITY COMPARISON
# =============================================================================

print('Joining metallicity datasets...')
metallicities = apokasc_df[(apokasc_df['DR16_FE_H'] > -9999) |
                           (apokasc_df['DR16_M_H_COR'] > -9999)]
metallicities = metallicities[['2MASS_ID', 'LOC_ID', 'DR16_FE_H',
                               'DR16_FE_H_ERR', 'DR16_M_H_COR',
                               'DR16_M_H_COR_ERR']]
metallicities.columns = ['id', 'loc_id', 'apokasc_Fe_H', 'apokasc_Fe_H_err',
                         'apokasc_M_H', 'apokasc_M_H_err']
metallicities['loc_id'] = metallicities['loc_id'].astype(int)
metallicities.set_index(['id', 'loc_id'], inplace=True)

astroNN_fe = astroNN_df[['APOGEE_ID',
                         'LOCATION_ID', 'FE_H', 'FE_H_ERR']].copy()
astroNN_fe.dropna(how='any', inplace=True)
astroNN_fe.columns = ['id', 'loc_id', 'astroNN_Fe_H', 'astroNN_Fe_H_err']
astroNN_fe['loc_id'] = astroNN_fe['loc_id'].astype(int)
astroNN_fe.set_index(['id', 'loc_id'], inplace=True)
metallicities = metallicities.join(astroNN_fe, how='outer')

metallicities.reset_index(inplace=True)
metallicities.set_index('id', inplace=True)
starhorse_fe = starhorse_df[['APOGEE_ID', 'met50']].copy()
starhorse_fe.columns = ['id', 'starhorse_M_H']
starhorse_fe['starhorse_M_H_err1'] = starhorse_df['met50'] - \
    starhorse_df['met16']
starhorse_fe['starhorse_M_H_err2'] = starhorse_df['met84'] - \
    starhorse_df['met50']
starhorse_fe.set_index('id', inplace=True)
metallicities = metallicities.join(starhorse_fe, how='outer')

print('Plotting Fe vs Fe...')
fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
# APOKASC vs astroNN
ax = axs[0, 0]
ax.scatter(metallicities['apokasc_Fe_H'], metallicities['astroNN_Fe_H'],
           c='k', s=1)
ax.errorbar(-3, 0,
            xerr=rms(metallicities['apokasc_Fe_H_err'], arrmin=-4, arrmax=1),
            yerr=rms(metallicities['astroNN_Fe_H_err'], arrmin=-4, arrmax=1),
            color='k', capsize=2)
ax.plot([-2, 0.5], [-2, 0.5], linestyle='--')
# Weirdos cut: APOKASC - astroNN >= 0.5
ax.plot([-0.7, 0.2], [-1.2, -0.3], linestyle='-', color='r')
ax.set_xlim((-4, 1))
ax.set_ylim((-4, 1))
ax.set_xlabel('APOKASC DR16 [Fe/H]')
ax.set_ylabel('astroNN [Fe/H]')

# APOKASC vs StarHorse
ax = axs[0, 1]
ax.scatter(metallicities['apokasc_M_H'], metallicities['starhorse_M_H'],
           c='k', s=1)
ax.errorbar(-2, 0,
            xerr=rms(metallicities['apokasc_M_H_err'], arrmin=-3, arrmax=1),
            yerr=[[rms(metallicities['starhorse_M_H_err1'], arrmin=-3, arrmax=1)],
                  [rms(metallicities['starhorse_M_H_err2'], arrmin=-3, arrmax=1)]],
            color='k', capsize=2)
ax.plot([-2, 0.5], [-2, 0.5], linestyle='--')
ax.set_xlim((-3, 1))
ax.set_ylim((-3, 1))
ax.set_xlabel('APOKASC DR16 [M/H]')
ax.set_ylabel('StarHorse [M/H]')

# APOKASC vs APOKASC
ax = axs[1, 0]
ax.scatter(metallicities['apokasc_Fe_H'], metallicities['apokasc_M_H'],
           c='k', s=1)
# Reported errors are too small to be noticeable
# ax.errorbar(-2, 0,
#             xerr=rms(metallicities['apokasc_Fe_H_err'], arrmin=-3, arrmax=1),
#             yerr=rms(metallicities['apokasc_M_H_err'], arrmin=-3, arrmax=1),
#             color='k', capsize=2)
ax.plot([-2, 0.5], [-2, 0.5], linestyle='--')
ax.set_xlim((-3, 1))
ax.set_ylim((-3, 1))
ax.set_xlabel('APOKASC DR16 [Fe/H]')
ax.set_ylabel('APOKASC DR16 [M/H]')

ax = axs[1, 1]
ax.scatter(metallicities['astroNN_Fe_H'], metallicities['starhorse_M_H'],
           c='k', s=1)
ax.errorbar(-3, 0,
            xerr=rms(metallicities['astroNN_Fe_H_err'], arrmin=-4, arrmax=1),
            yerr=[[rms(metallicities['starhorse_M_H_err1'], arrmin=-4, arrmax=1)],
                  [rms(metallicities['starhorse_M_H_err2'], arrmin=-4, arrmax=1)]],
            color='k', capsize=2)
ax.plot([-2, 0.5], [-2, 0.5], linestyle='--')
ax.set_xlim((-4, 1))
ax.set_ylim((-4, 1))
ax.set_xlabel('astroNN [Fe/H]')
ax.set_ylabel('StarHorse [M/H]')

plt.show()

# =============================================================================
# ALPHA VS ALPHA
# =============================================================================
