from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io.fits import getdata

# Plot settings
plt.rc(('xtick', 'ytick'), direction='in')
plt.rc('xtick', top=True)
plt.rc('ytick', right=True)
plt.rc('font', family='STIXgeneral')

# Paths
data_dir = 'C:\\Users\\dubay.11\\Data'
data_path = Path(data_dir)
apokasc_file = 'APOKASC_cat_v6.7.2.fits'
astronn_file = 'apogee_astroNN-DR17'
starhorse_file = 'APOGEE_DR17_EDR3_STARHORSE_v2'

data, hdr = getdata(data_path / apokasc_file, 1, header=True)

# Relevant data columns
alpha_col = 'DR16_CA_FE'
fe_col = 'DR16_FE_H'
age_col = 'AGE_DW'
age_alt_col = 'AGE_DW_SDSS'

# Parameter cuts
alpha_cut = 0. # below
fe_cut = -1.1 # below
age_cut = 9. # Gyr, above

cuts = np.where((data[fe_col] > -9999) & (data[alpha_col] > -9999))
data = data[cuts]

# Alpha vs Fe plot
fig, ax = plt.subplots(dpi=300)
ax.scatter(data[fe_col], data[alpha_col], c='k', s=1)
# mean error bar
# ax.errorbar(-2.2, -0.8, xerr=iron_mean_error, yerr=alpha_mean_error, color='k')
ax.set_xlabel(fe_col)
ax.set_ylabel(alpha_col)
plt.show()

# Inset in low-alpha, low-Fe regime
low_fe_low_alpha = data[np.where((data[fe_col] < fe_cut) & (data[alpha_col] < alpha_cut))]

fig, ax = plt.subplots(dpi=300)
ax.errorbar(low_fe_low_alpha[fe_col], low_fe_low_alpha[alpha_col],
            xerr=low_fe_low_alpha[fe_col + '_ERR'],
            yerr=low_fe_low_alpha[alpha_col + '_ERR'],
            color='k', linestyle='none')
ax.set_xlabel(fe_col)
ax.set_ylabel(alpha_col)
plt.show()

print(f'[Fe/H] < {fe_cut} and [alpha/Fe] < {alpha_cut}:\n', 
      low_fe_low_alpha['2MASS_ID'])

# print(len(data[data[age_col] > -9999]))
# print(len(data[data[age_alt_col] > -9999]))
data = data[data[age_col] > -9999]
# Add errors in quadrature - is this right?
age_err = np.sqrt(data[age_col + '_PERR']**2 + data[age_col + '_SYSERR']**2)
age_alt_err = np.sqrt(data[age_alt_col + '_PERR']**2 + data[age_alt_col + '_SYSERR']**2)

# Age comparison plot
fig, ax = plt.subplots(dpi=300)
ax.plot([0, 16], [0, 16], linestyle='--', color='k')
ax.scatter(data[age_col], data[age_alt_col], c='k', s=1)
ax.errorbar(data[age_col], data[age_alt_col], xerr=age_err, yerr=age_alt_err,
            linestyle='none', color='k', alpha=0.1, elinewidth=1)
ax.set_xlim((0, 16))
ax.set_ylim((0, 16))
ax.set_xlabel(age_col)
ax.set_ylabel(age_alt_col)
plt.show()

# Age vs alpha plot
fig, ax = plt.subplots(dpi=300)
ax.scatter(data[age_col], data[alpha_col], c='k', s=1)
ax.set_xlabel(age_col)
ax.set_ylabel(alpha_col)
plt.show()

# Age vs Fe plot
fig, ax = plt.subplots(dpi=300)
ax.scatter(data[age_col], data[fe_col], c='k', s=1)
ax.set_xlabel(age_col)
ax.set_ylabel(fe_col)
plt.show()

# Select oldest stars
oldest = data[data[age_col] > age_cut]
age_err = np.sqrt(oldest[age_col + '_PERR']**2 + oldest[age_col + '_SYSERR']**2)

fig, ax = plt.subplots(dpi=300)
ax.scatter(oldest[age_col], oldest[alpha_col], c='k', s=4)
ax.errorbar(oldest[age_col], oldest[alpha_col], 
            xerr=age_err, yerr=oldest[alpha_col + '_ERR'],
            color='k', linestyle='none', elinewidth=1, alpha=0.2)
ax.set_xlim((age_cut - 0.5, 16))
ax.set_xlabel(age_col)
ax.set_ylabel(alpha_col)
plt.show()

oldest_low_alpha = oldest[oldest[alpha_col] < alpha_cut]
print(f'Age > {age_cut} Gyr and [alpha/Fe] < {alpha_cut}:\n', 
      oldest_low_alpha['2MASS_ID'])
# np.savetxt('oldest_low_alpha.txt', oldest_low_alpha)

print(f'Age > {age_cut} Gyr, [alpha/Fe] < {alpha_cut} and [Fe/H] < {fe_cut}:\n', 
      oldest_low_alpha[oldest_low_alpha[fe_col] < fe_cut]['2MASS_ID'])