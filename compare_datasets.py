from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io.fits import getdata
from astropy.table import Table, join

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
ages.columns = ['id', 'loc_id', 'apokasc2_age', 'apokask2_age_err1', 
                'apokask2_age_err2']
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
ages = ages.join(astroNN_ages, how='inner')
print(ages)
# ages = ages.rename(columns={'age_lowess_correct': 'astroNN_age', 
#                             'age_total_error': 'astroNN_age_err'})
# print(ages)

# for i, id in enumerate(ages.index):
#     if id in ages.index[:i]:
#         print(ages.loc[id, ['apokasc2_age', 'astroNN_age', 'astroNN_age_err']])

# astroNN_ages = data[data[age_col] > 0][[id_col, age_col, age_err_col]]
# astroNN_ages.rename_column(id_col, 'ID')
# print(astroNN_ages)

# combined_ages = join(apokasc_ages, astroNN_ages, keys='ID')
# print(combined_ages)

### StarHorse DR17 catalog
# data = Table.read(data_path / starhorse_file, format='fits')

id_col = 'APOGEE_ID'
metal_col = 'met50'
metal_err1_col = 'met16'
metal_err2_col = 'met84'