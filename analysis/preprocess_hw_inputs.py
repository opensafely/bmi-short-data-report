import sys 
import pandas as pd
import numpy as np

height = sys.argv[1]
weight = sys.argv[2]
code = sys.argv[3]
bmi = sys.argv[4]

df_input = pd.read_feather(f'output/data/input_{code}_hw.feather')

# Change variable names 
for n in range(1,11):
    df_input = df_input.rename(
        columns={
            f'{height}{n}_date':f'{height}_date{n}', 
            f'{weight}{n}_date':f'{weight}_date{n}',
        }
    )

# Reshape wide to long
df_output = pd.wide_to_long(
    df_input, 
    [
        f'{height}',f'{height}_date',f'{height}_age', 
        f'{weight}',f'{weight}_date',f'{weight}_age'
    ], 
    i='patient_id', 
    j="number"
).reset_index()

# Delete weight and height if age < 18
for var in [f'{height}', f'{weight}']:
    df_output.loc[df_output[f'{var}_age'] < 18, var] = 0
    df_output.loc[df_output[f'{var}_age'] < 18, f'{var}_date'] = np.nan
    df_output.loc[df_output[f'{var}_age'] < 18, f'{var}_age'] = 0

# Frontfill height if adult
df_output.sort_values(by=['patient_id',f'{height}_date'])
for var in [f'{height}',f'{height}_date',f'{height}_age']:
    df_output[var] = df_output.groupby('patient_id')[var].transform(lambda v: v.ffill())

# Create computed BMI if height and weight both populated
df_output.loc[
    (df_output[f'{height}'] != 0) & (df_output[f'{weight}'] != 0) & 
    (~df_output[f'{height}'].isna()) & (~df_output[f'{weight}'].isna()),
    f'{bmi}'] = df_output[f'{weight}']/(df_output[f'{height}']**2)

df_output.loc[
    (~df_output[f'{bmi}'].isna()), 
    f'{bmi}_date'] = df_output[f'{weight}_date']

# Adjust for division by 0 (return 0)
df_output.loc[~np.isfinite(df_output[f'{bmi}']), f'{bmi}'] = 0

df_output.to_feather(f'output/data/input_{bmi}_processed.feather')