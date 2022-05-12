import os 
import pandas as pd
import numpy as np

df_input = pd.read_feather('output/data/input.feather')

# Change variable names 
for n in range(1,11):
    df_input = df_input.rename(columns={f'derived_bmi{n}_date_measured':f'derived_bmi_date{n}',
                                        f'recorded_bmi{n}_date':f'recorded_bmi_date{n}',
                                        f'weight{n}_date':f'weight_date{n}',
                                        f'weight_backend{n}_date':f'weight_backend_date{n}',
                                        f'height{n}_date':f'height_date{n}',
                                        f'height_backend{n}_date':f'height_backend_date{n}',
                                       })

# Reshape wide to long
filter_col = [col for col in df_input if col.startswith('derived') | 
              col.startswith('recorded') | 
              col.startswith('height')   |
              col.startswith('weight')
             ]
id_col = [col for col in df_input if col not in filter_col]

df_output = pd.wide_to_long(df_input, ['derived_bmi', 'derived_bmi_date', 
                                       'recorded_bmi', 'recorded_bmi_date',
                                       'weight', 'weight_date', 'weight_age',
                                       'weight_backend', 'weight_backend_date', 
                                       'weight_backend_age', 
                                       'height', 'height_date', 'height_age',
                                       'height_backend', 'height_backend_date',
                                       'height_backend_age'
                                      ], i='patient_id', j="number").reset_index()

# Delete weight and height if age < 18
for var in ['weight', 'weight_backend', 'height', 'height_backend']:
    df_output.loc[df_output[f'{var}_age'] < 18, var] = 0
    df_output.loc[df_output[f'{var}_age'] < 18, f'{var}_date'] = np.nan
    df_output.loc[df_output[f'{var}_age'] < 18, f'{var}_age'] = 0

# Frontfill height if adult
# SNOMED
df_output.sort_values(by=['patient_id','height_date'])
for var in ['height','height_date','height_age']:
    df_output[var] = df_output.groupby('patient_id')[var].transform(lambda v: v.ffill())
# CTV3 backend definition
df_output.sort_values(by=['patient_id','height_backend_date'])
for var in ['height_backend','height_backend_date','height_backend_age']:
    df_output[var] = df_output.groupby('patient_id')[var].transform(lambda v: v.ffill())

# Create computed BMI if height and weight both populated
df_output.loc[(df_output['height'] != 0) & (df_output['weight'] != 0) & (~df_output['height'].isna()) & (~df_output['weight'].isna()),
              'computed_bmi'] = df_output['weight']/(df_output['height']**2)
df_output.loc[(~df_output['computed_bmi'].isna()), 'computed_bmi_date'] = df_output['weight_date']
# Adjust for division by 0 (return 0)
df_output.loc[~np.isfinite(df_output['computed_bmi']), 'computed_bmi'] = 0

# Create backend BMI if height and weight both populated
df_output.loc[(df_output['height_backend'] != 0) & (df_output['weight_backend'] != 0) & (~df_output['height_backend'].isna()) & (~df_output['weight_backend'].isna()),
              'backend_computed_bmi'] = df_output['weight_backend']/(df_output['height_backend']**2)
df_output.loc[(~df_output['backend_computed_bmi'].isna()), 'backend_computed_bmi_date'] = df_output['weight_backend_date']
# Adjust for division by 0 (return 0)
df_output.loc[~np.isfinite(df_output['backend_computed_bmi']), 'backend_computed_bmi'] = 0

df_output.to_feather('output/data/input_processed.feather')