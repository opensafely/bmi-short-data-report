import sys
import pandas as pd

bmi = sys.argv[1]

df_input = pd.read_feather(f'output/data/input_{bmi}.feather')

# Change variable names 
for n in range(1,11):
    if bmi == 'derived_bmi':
        df_input = df_input.rename(
            columns={f'{bmi}{n}_date_measured':f'{bmi}_date{n}',}
        )
    elif bmi == 'recorded_bmi':
        df_input = df_input.rename(
            columns={f'{bmi}{n}_date':f'{bmi}_date{n}',}
        )

# Reshape wide to long
df_output = pd.wide_to_long(
    df_input, 
    [f'{bmi}', f'{bmi}_date',], 
    i='patient_id', 
    j="number"
)

# Delete if value is null
df_output = df_output.loc[df_output[f'{bmi}'] != 0].reset_index()

df_output.to_feather(f'output/data/input_processed_{bmi}.feather')