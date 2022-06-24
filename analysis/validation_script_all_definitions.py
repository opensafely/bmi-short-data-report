import os
import numpy as np
import pandas as pd

definitions = [
    'derived_bmi',
    'recorded_bmi',
    'backend_computed_bmi',
    'computed_bmi'
]

demographic_covariates = [
    'age_band', 
    'sex', 
    'ethnicity', 
    'region', 
    'imd'
]

clinical_covariates = [
    'dementia', 
    'diabetes', 
    'learning_disability'
]

filepath = 'output/validation/tables/comparison'

################################# FUNCTIONS #####################################

def redact(df_in):
    df_out = df_in.where(
        df_in > 5, np.nan
    ).apply(lambda x: 5 * round(x / 5))
    return df_out

def import_clean(filepath, definitions, demographic_covariates,
                clinical_covariates):

    # Check whether output paths exist or not, create if missing
    exists = os.path.exists(filepath)
    if not exists:
        os.makedirs(filepath)

    # Import and concatenate
    li_dfs = []
    for d in definitions:
        print(f'Importing {d} file...')
        df_input = pd.read_feather(
            f'output/joined/input_processed_{d}.feather'
        ).drop(columns=['number'])
        df_input['bmi_type'] = f'{d}'
        df_input = df_input.rename(
            columns={f'{d}':'bmi', f'{d}_date':'bmi_date'}
        )
        print(f'Imported {d} file')
        li_dfs.append(df_input)
    df_bmi = pd.concat(li_dfs)
    print('Concatenated dataframes')
    del li_dfs

    # Drop unnecessary columns
    li_drop_cols = []
    for col in df_bmi.columns:
        if col.startswith('height') | col.startswith('weight'):
            li_drop_cols.append(col)
    df_bmi = df_bmi.drop(columns=li_drop_cols)
    print('Dropped unnecessary columns')
    del li_drop_cols

    # Create order for categorical variables
    for group in demographic_covariates + clinical_covariates:
        if df_bmi[group].dtype.name == 'category':
            li_order = sorted(df_bmi[group].dropna().unique().tolist())
            df_bmi[group] = pd.Categorical(df_bmi[group], categories=li_order)
            print(f'Ordered {group} variables')

    print("Successfully cleaned import variables")
    print(df_bmi.memory_usage())
    return df_bmi

def all_counts(df_bmi, filepath):

    df_bmi.loc[df_bmi['bmi'] == 0, 'missing'] = True
    df_bmi.loc[df_bmi['bmi'] > 0, 'filled'] = True

    df_bmi = df_bmi.sort_values(by='patient_id')
    df_all = df_bmi.drop_duplicates(subset='patient_id')
    pop_ct = df_all['patient_id'].count()
    print('Counted population')
    del df_all

    df_filled = df_bmi.drop_duplicates(
        subset=['patient_id','bmi_type','filled']
    )[['patient_id','filled']]
    df_filled_sum = pd.DataFrame(
        df_filled.groupby('patient_id')['filled'].sum()
    ).reset_index()
    filled_ct = df_filled_sum.loc[df_filled_sum['filled'] == 3]['patient_id'].count()
    print('Counted filled')
    del df_filled
    del df_filled_sum

    df_missing = df_bmi.drop_duplicates(
        subset=['patient_id','bmi_type','missing']
    )[['patient_id','missing']]
    df_missing_sum = pd.DataFrame(
        df_missing.groupby('patient_id')['missing'].sum()
    ).reset_index()
    missing_ct = df_missing_sum.loc[df_missing_sum['missing'] == 3]['patient_id'].count()
    print('Counted missing')
    del df_missing
    del df_missing_sum

    df_counts = pd.DataFrame(
        [pop_ct,filled_ct,missing_ct], 
        index=['population','all_filled','all_missing'], 
        columns=['total counts']
    ).T

    # Export
    df_counts.to_csv(
        f'{filepath}/total_filled_missing_counts.csv'
    )

def count_by_group(df_bmi, filepath, demographic_covariates,
                   clinical_covariates):
    for group in demographic_covariates + clinical_covariates:
        # All
        df_bmi = df_bmi.sort_values(by=['patient_id',group])
        df_all = df_bmi.drop_duplicates(subset=['patient_id',group])
        df_all_ct = df_all[['patient_id',group]].groupby(
            group).count().rename(columns={'patient_id':'population'})
        df_all_ct.to_csv(f'{filepath}/total_counts_{group}.csv')
        print(f"Counted population by {group}")

        # All filled
        df_filled = df_bmi.drop_duplicates(
            subset=['patient_id','bmi_type','filled',group]
        )[['patient_id','filled',group]]
        df_filled_sum = pd.DataFrame(
            df_filled.groupby(['patient_id',group]
        )['filled'].sum()).reset_index()
        df_filled_ct = pd.DataFrame(
            df_filled_sum.loc[df_filled_sum['filled'] == 3].groupby(
                group
            )['patient_id'].count()).rename(columns={'patient_id':'all_filled'})
        df_filled_ct.to_csv(f'{filepath}/filled_counts_{group}.csv')
        print(f"Counted filled by {group}")

        # All missing
        df_missing = df_bmi.drop_duplicates(
            subset=['patient_id','bmi_type','missing',group]
        )[['patient_id','missing',group]]
        df_missing_sum = pd.DataFrame(
            df_missing.groupby(['patient_id',group]
            )['missing'].sum()).reset_index()
        df_missing_ct = pd.DataFrame(
            df_missing_sum.loc[df_missing_sum['missing'] == 3].groupby(
                group
            )['patient_id'].count()).rename(columns={'patient_id':'all_missing'})
        df_missing_ct.to_csv(f'{filepath}/missing_counts_{group}.csv')
        print(f"Counted missing by {group}")


def upset_crosstab(df_bmi, definitions):
    df_filled = df_bmi.drop_duplicates(
        subset=['patient_id','bmi_type','filled']
    )[['patient_id','filled','bmi_type']]
    df_filled = df_filled.loc[df_filled.filled == 1]
    df_filled_pivot = df_filled.pivot(
        index='patient_id', 
        columns='bmi_type', 
        values='filled'
    ).reset_index(drop=True).fillna(False)
    df_crosstab = pd.DataFrame(
        df_filled_pivot.groupby(definitions[1:])[definitions[0]].value_counts()
    )
    df_crosstab = redact(df_crosstab)
    print("Generated crosstab table")
    df_crosstab.to_csv(f'{filepath}/upset_crosstab.csv')

########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_bmi = import_clean(
        filepath, 
        definitions,
        demographic_covariates,
        clinical_covariates
    )
    # Run counts for all
    all_counts(df_bmi, filepath)
    # Run counts by group
    count_by_group(
        df_bmi, 
        filepath, 
        demographic_covariates,
        clinical_covariates
    )
    # Run crosstab for upset plot data
    upset_crosstab(df_bmi, definitions)
    
########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    

