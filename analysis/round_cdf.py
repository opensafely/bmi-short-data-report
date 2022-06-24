import pandas as pd
import numpy as np

definitions = [
    'derived_bmi',
    'recorded_bmi',
    'backend_computed_bmi',
    'computed_bmi'
]

def cdf_round(definition, suffix):
    file = f"output/validation/tables/{definition}/{definition}_{suffix}.csv"
    df_in = pd.read_csv(file, index_col=0).drop(columns=['pdf','cdf'])
    df_in['frequency'] = np.ceil(df_in['frequency']/5)*(5-np.floor(5/2))
    # Drop 0s
    df_in = df_in.loc[df_in['frequency'] > 0]
    # Compute PDF
    df_in["pdf"] = df_in["frequency"] / sum(df_in["frequency"])
    # Compute CDF
    df_in["cdf"] = df_in["pdf"].cumsum()
    df_out = df_in.reset_index(drop=True)
    df_out.to_csv(f'output/validation/formatted_tables/{definition}_{suffix}.csv')

for definition in definitions:
    for suffix in ['cdf_data','date_diff_cdf_data']:
        cdf_round(definition, suffix)