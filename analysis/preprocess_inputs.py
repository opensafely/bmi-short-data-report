import os 
import pandas as pd
import numpy as np

df_input = pd.read_feather('output/data/input.feather')

# Change variable names 
for n in range(1,11):
    df_input = df_input.rename(columns={f'derived_bmi{n}_date_measured':f'derived_bmi_date{n}',
                                        f'recorded_bmi{n}_date':f'recorded_bmi_date{n}',
                                        f'weight{n}_date':f'weight_date{n}',
                                        f'height{n}_date':f'height_date{n}'
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
                                       'height', 'height_date', 'height_age'
                                      ], i='patient_id', j="number").reset_index()

# Delete weight and height if age < 18
df_output.loc[df_output['weight_age'] < 18, ['weight','weight_date','weight_age']] = np.nan
df_output.loc[df_output['weight'] == 0, 'weight'] = np.nan

df_output.loc[df_output['height_age'] < 18, ['height','height_date','height_age']] = np.nan
df_output.loc[df_output['height'] == 0, 'height'] = np.nan

# Frontfill height if adult
df_output.sort_values(by=['patient_id','height_date'])
df_output['height'] = df_output.groupby('patient_id')['height'].transform(lambda v: v.ffill())
df_output['height_date'] = df_output.groupby('patient_id')['height_date'].transform(lambda v: v.ffill())
df_output['height_age'] = df_output.groupby('patient_id')['height_age'].transform(lambda v: v.ffill())

# Create computed BMI
df_output['computed_bmi'] = df_output['weight']/(df_output['height']**2)
df_output['computed_bmi_date'] = df_output['weight_date']

# Adjust for division by 0 (return 0)
df_output.loc[~np.isfinite(df_output['computed_bmi']), 'computed_bmi'] = 0

df_output.to_feather('output/data/input_processed.feather')