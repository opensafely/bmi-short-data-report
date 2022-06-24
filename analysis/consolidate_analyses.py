import os
import pandas as pd
import numpy as np

from functools import reduce
from glob import glob

definitions = [
    'derived_bmi',
    'recorded_bmi',
    'backend_computed_bmi',
    'computed_bmi'
]

################################# FUNCTIONS #####################################


def format_distribution(bmi):
    files = glob(f"output/validation/tables/{bmi}/*_distribution.csv") 
    li_df = []
    for f in files:
        df_temp = pd.read_csv(f)
        df_temp['category'] = df_temp.columns[0]
        if df_temp.columns[0] == 'Unnamed: 0':
            df_temp['category'] = 'population'
            df_temp['subcategory'] = 'all'
            df_temp = df_temp.drop(columns=['Unnamed: 0'])
        else:
            df_temp = df_temp.rename(columns={df_temp.columns[0]:'subcategory'})
        df_temp = df_temp.set_index(['category','subcategory'])
        li_df.append(df_temp)
    df_out = pd.concat(li_df)
    df_out.to_csv(f"output/validation/formatted_tables/{bmi}_distribution.csv")


def format_counts(definitions, units):
    if units == "patient_counts":
        col = "num_patients"
    else:
        col = "num_measurements"
    li_all_df = []
    for definition in definitions:
        files = glob(f"output/validation/tables/{definition}/*_{units}.csv") 
        li_df = []
        for f in files:
            df_temp = pd.read_csv(f)
            df_temp['category'] = df_temp.columns[0]
            if df_temp['category'][0] == 'Unnamed: 0':
                df_temp['category'] = 'population'
                df_temp['subcategory'] = 'all'
                df_temp = df_temp.drop(columns=['Unnamed: 0'])
            else:
                df_temp = df_temp.rename(columns={df_temp.columns[0]:'subcategory'})
            li_df.append(df_temp)
        df_combined = pd.concat(li_df)
        df_combined = df_combined.rename(
                columns={col:f'{definition}_{col}'}
        )
        li_all_df.append(df_combined)
    # Merge files
    df_out = reduce(lambda df1, df2: pd.merge(
        df1, df2, on=['category','subcategory'], how='outer'
    ), li_all_df).set_index(['category','subcategory'])
    df_out.to_csv(f"output/validation/formatted_tables/{units}.csv")


def format_over_time(bmi, units):
    files = glob(f"output/validation/tables/{bmi}/*_{units}_over_time.csv") 
    li_df = []
    for f in files:
        df_temp = pd.read_csv(f, index_col = 0)
        df_temp['category'] = df_temp.columns[1]
        # Format all population
        if (df_temp['category'][1] == 'count') | (df_temp['category'][1] == f'{bmi}'):
            df_temp['category'] = 'population'
            df_temp['subcategory'] = 'all'
        # Format by group
        else:
            df_temp = df_temp.rename(
                columns={df_temp.columns[1]:'subcategory'}
            )
        # Only keep 2015 on
        df_temp = df_temp.loc[df_temp.date > '2014-12-01']
        df_temp = df_temp.set_index(['category','subcategory','date'])
        li_df.append(df_temp)

    df_out = pd.concat(li_df)
    df_out.to_csv(f"output/validation/formatted_tables/{bmi}_{units}_over_time.csv")


def format_out_of_range(definitions, specification):
    li_all_df = []
    for definition in definitions:
        files = glob(f"output/validation/tables/{definition}/*_{specification}.csv") 
        li_df = []
        # Join files by definition
        for f in files:
            df_temp = pd.read_csv(f)
            # Format all population
            if df_temp.columns[0] == 'Unnamed: 0':
                df_temp = df_temp.rename(
                    columns={'Unnamed: 0':'population',f'{definition}':'all'}
                )
                df_temp = df_temp.T.reset_index()
                df_temp.columns = df_temp.iloc[0]
                df_temp = df_temp.iloc[1: , :].rename(
                    columns={'population':'subcategory'}
                )
                df_temp['category'] = 'population'
            # Format by group
            else:
                df_temp['category'] = df_temp.columns[0]
                df_temp = df_temp.rename(
                    columns={df_temp.columns[0]:'subcategory'}
                )
            df_temp = df_temp.set_index(
                ['category',df_temp.iloc[:,0]]
            ).drop(columns=[df_temp.columns[0]])
            li_df.append(df_temp)
        df_combined = pd.concat(li_df).reset_index()
        df_combined = df_combined.rename(
            columns={'count':f'count_{definition}', 'mean':f'mean_{definition}'}
        )
        li_all_df.append(df_combined)
    # Merge files
    df_out = reduce(lambda df1, df2: pd.merge(
        df1, df2, on=['category','subcategory'], how='outer'
    ), li_all_df)
    df_out.to_csv(f"output/validation/formatted_tables/{specification}.csv")


def main():

    # Check whether output paths exist or not, create if missing
    filepath = f'output/validation/formatted_tables'
    exists = os.path.exists(filepath)
    if not exists:
        os.makedirs(filepath)

    # Counts
    format_counts(definitions, 'patient_counts')
    format_counts(definitions, 'measurement_counts')

    for definition in definitions: 
        # Distribution
        format_distribution(definition)
        # Over time
        format_over_time(definition, 'means')
        format_over_time(definition, 'records')

    # Out-of-range analyses
    format_out_of_range(definitions, 'greater_than_max')
    format_out_of_range(definitions, 'less_than_min')

########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()