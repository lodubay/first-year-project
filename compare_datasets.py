from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

# Plot settings
plt.rc(('xtick', 'ytick'), direction='in')
plt.rc('xtick', top=True)
plt.rc('ytick', right=True)
plt.rc('font', family='STIXgeneral')

# Paths
data_dir = 'C:\\Users\\dubay.11\\Data'
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

### APOKASC catalog
print('Importing APOKASC catalog...')
data = Table.read(data_path / apokasc_file, format='fits')
apokasc_df = decode(data.to_pandas())

# Relevant data columns
id_col = '2MASS_ID'
loc_col = 'LOC_ID'
alpha_col = 'DR16_ALP_M_COR'
alpha_err_col = 'ALP_M_COR_ERR'
fe_col = 'DR16_FE_H'
fe_err_col = 'DR16_FE_H_ERR'
metal_col = 'DR16_M_H_COR'
metal_err_col = 'DR16_M_H_COR_ERR'
age_col = 'APOKASC2_AGE'
age_err1_col = 'APOKASC2_AGE_MERR' # error in negative direction
age_err2_col = 'APOKASC2_AGE_PERR' # error in positive direction

# DataFrame with only ages
ages = apokasc_df[apokasc_df[age_col] > 0]
ages = ages[[id_col, loc_col, age_col, age_err1_col, age_err2_col]]
ages.columns = ['id', 'loc_id', 'apokasc2_age', 'apokasc2_age_err1', 
                'apokasc2_age_err2']
ages['loc_id'] = ages['loc_id'].astype(int)
ages.set_index(['id', 'loc_id'], inplace=True)

### astroNN DR17 catalog
print('Importing astroNN DR17 catalog...')
data = Table.read(data_path / astroNN_file, format='fits')
astroNN_df = decode(data.to_pandas())

id_col = 'APOGEE_ID'
loc_col = 'LOCATION_ID'
age_col = 'age_lowess_correct'
age_err_col = 'age_total_error'
fe_col = 'FE_H'
fe_err_col = 'FE_H_ERR'

astroNN_ages = astroNN_df[astroNN_df[age_col] > 0]
astroNN_ages = astroNN_ages[[id_col, loc_col, age_col, age_err_col]]
astroNN_ages.columns = ['id', 'loc_id', 'astroNN_age', 'astroNN_age_err']
astroNN_ages['loc_id'] = astroNN_ages['loc_id'].astype(int)
astroNN_ages.set_index(['id', 'loc_id'], inplace=True)

# Combined age data
print('Joining datasets...')
ages = ages.join(astroNN_ages, how='inner')
# Remove infinities
ages.replace([np.inf, -np.inf], np.nan, inplace=True)
ages = ages.dropna(how='any')
print(ages)

# Average age error
apokasc_rms_err = [[np.sqrt(np.mean(ages['apokasc2_age_err1']**2))], 
                   [np.sqrt(np.mean(ages['apokasc2_age_err2']**2))]]
astroNN_rms_err = np.sqrt(np.mean(ages['astroNN_age_err']**2))

### Plot APOKASC age vs astroNN age
fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
ax = axs[0]
ax.plot([0, 14], [0, 14], linestyle='--')
ax.scatter(ages['apokasc2_age'], ages['astroNN_age'], s=0.5, c='k')
# ax.errorbar(ages['apokasc2_age'], ages['astroNN_age'],
#             xerr=[ages['apokasc2_age_err1'], ages['apokasc2_age_err2']],
#             yerr=ages['astroNN_age_err'],
#             linestyle='none', elinewidth=1, color='k', alpha=0.05)
# plot RMS error
ax.errorbar(12, 1, xerr=apokasc_rms_err, yerr=astroNN_rms_err,
            color='k', capsize=2)
ax.set_xlabel('APOKASC2 Age [Gyr]')
ax.set_ylabel('astroNN Age [Gyr]')
ax.set_xlim((-1, 15))
ax.set_ylim((-1, 15))

# Histrogram of age differences
ax = axs[1]
# xmin = -1
# xmax = 1.
# bins = np.logspace(xmin, xmax, 20)
# ax.hist(ages['astroNN_age'] / ages['apokasc2_age'], color='k', bins=bins)
# ax.set_xscale('log')
# ax.set_xlim((10**xmin, 10**xmax))
# ax.set_xlabel('astroNN Age / APOKASC2 Age')
xmin = -14
xmax = 14
bins = np.linspace(xmin, xmax, 29)
ax.hist(ages['astroNN_age'] - ages['apokasc2_age'], 
        color='k', bins=bins, rwidth=0.9)
ax.grid(which='major', axis='y', color='w')
ax.set_xlim((xmin, xmax))
ax.set_yscale('log')
ax.set_xlabel('astroNN Age - APOKASC2 Age [Gyr]')
ax.set_ylabel('Count')
plt.savefig('age_comparison.png', dpi=300)
plt.show()

### StarHorse DR17 catalog
# data = Table.read(data_path / starhorse_file, format='fits')

id_col = 'APOGEE_ID'
metal_col = 'met50'
metal_err1_col = 'met16'
metal_err2_col = 'met84'