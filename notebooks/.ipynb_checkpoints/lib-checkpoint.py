import itertools
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from ebmdatalab import charts
from functools import reduce
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

def redact_round_table(df_in):
    """Redacts counts <= 5 and rounds counts to nearest 5"""
    df_out = df_in.where(df_in > 5, np.nan).apply(lambda x: 5 * round(x/5))
    return df_out

def import_clean(input_path, definitions, other_vars, demographic_covariates, clinical_covariates, null, dates=False):
    # Import
    df_import = pd.read_feather(input_path)
    if dates==True:
        date_vars = [definition+'_date' for definition in definitions]
        for definition in definitions:
            df_import[definition+'_date'] = df_import[definition+'_date'].dt.to_period('M').dt.to_timestamp()
    else: 
        date_vars = []
    # Subset to relevant columns
    df_clean = df_import[['patient_id'] + definitions + other_vars + date_vars + demographic_covariates + clinical_covariates]
    # Limit to relevant date range
    df_clean = df_clean.sort_values(by='patient_id').reset_index(drop=True)
    # Set null values to nan
    for definition in definitions: 
        df_clean.loc[df_clean[definition] == null, definition] = np.nan
     # Create order for categorical variables
    for group in demographic_covariates + clinical_covariates:
        if df_clean[group].dtype.name == 'category':
            li_order = sorted(df_clean[group].dropna().unique().tolist())
            df_clean[group] = df_clean[group].cat.reorder_categories(li_order, ordered=True)
    # Mark patients with value filled/missing for each definition
    # Mark filled
    li_filled = []
    for definition in definitions:
        df_fill = pd.DataFrame(df_clean.groupby("patient_id")[definition].any().astype('int')).rename(
            columns={definition:definition+'_filled'}
        )
        df_fill[definition+'_missing'] = 1-df_fill[definition+'_filled']
        li_filled.append(df_fill)

    df_filled = pd.concat(li_filled, axis=1)
    df_clean = df_clean.merge(df_filled, on='patient_id')
    return df_clean

def patient_counts(df_clean, definitions, demographic_covariates, clinical_covariates, missing=False):
    suffix = '_filled'
    subgroup = 'with records'
    if missing == True:
        suffix = '_missing'
        subgroup = 'missing records'
    # All population
    li_pop = []
    for definition in definitions:
        df_temp = df_clean[['patient_id']].drop_duplicates().set_index('patient_id')
        df_temp[definition+suffix] = 1
        li_pop.append(df_temp)

    df_temp0 = pd.concat(li_pop)
    df_pop = pd.DataFrame(df_temp0.sum()).T
    df_pop['group'],df_pop['subgroup'] = ['population','N']
    df_pop = df_pop.set_index(['group','subgroup'])

    # All with measurement
    li_filled = []
    for definition in definitions:
        df_temp = df_clean[['patient_id', definition+suffix]].drop_duplicates().dropna().set_index('patient_id')
        li_filled.append(df_temp)

    df_temp2 = pd.concat(li_filled, axis=1)
    df_all = pd.DataFrame(df_temp2.sum()).T
    df_all['group'],df_all['subgroup'] = ['population',subgroup]
    df_all = df_all.set_index(['group','subgroup'])

    # By group
    li_group = []
    for group in demographic_covariates + clinical_covariates:
        li_filled_group = []
        for definition in definitions:
            df_temp = df_clean[['patient_id', definition+suffix, group]].drop_duplicates().dropna().reset_index(drop=True)
            li_filled_group.append(df_temp)
        df_reduce = reduce(lambda df1, df2: pd.merge(df1, df2,on=['patient_id',group],how='outer'), li_filled_group)
        df_reduce2 = df_reduce.sort_values(by=group).drop(columns=['patient_id']).groupby(group).sum().reset_index()
        df_reduce2['group'] = group
        df_reduce2 = df_reduce2.rename(columns={group:'subgroup'})
        li_group.append(df_reduce2)
    df_all_group = pd.concat(li_group, axis=0, ignore_index=True).set_index(['group','subgroup'])
    df_all_ct = df_pop.append([df_all,df_all_group])

    # Redact
    df_all_redact = redact_round_table(df_all_ct)

    # Create percentage columns 
    for definition in definitions:
        df_all_redact[definition+'_pct'] = round((df_all_redact[definition+suffix].div(df_all_redact[definition+suffix][0]))*100,1)

    # Column order
    li_col_order = []
    for definition in definitions:
        li_col_order.append(definition+suffix)
        li_col_order.append(definition+'_pct')
    df_all_redact = df_all_redact[li_col_order]
    
    # Final redaction step
    df_all_redact = df_all_redact.where(~df_all_redact.isna(), '-')
    return df_all_redact

def display_heatmap(df_clean, definitions):
    # All with measurement
    li_filled = []
    for definition in definitions:
        df_temp = df_clean[['patient_id']].drop_duplicates().set_index('patient_id')
        df_temp[definition+'_filled'] = 1
        df_temp = df_clean[['patient_id', definition+'_filled']].drop_duplicates().dropna().set_index('patient_id')
        li_filled.append(df_temp)

    # Prepare data for heatmap input
    df_temp2 = pd.concat(li_filled, axis=1)
    df_transform = df_temp2.replace(np.nan,0)
    df_dot = df_transform.T.dot(df_transform)
    
    # Create mask to eliminate duplicates in heatmap
    mask = np.triu(np.ones_like(df_dot))
    np.fill_diagonal(mask[::1], 0)

    # Draw the heatmap with the mask
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_dot, annot=True, mask=mask, fmt='g', cmap="YlGnBu", vmin=0)
    plt.show()

def records_over_time(df_clean, definitions, demographic_covariates, clinical_covariates):
    li_df = []
    for definition in definitions:
        df_grouped = df_clean[[definition+'_date',definition]].groupby(definition+'_date').count().reset_index().rename(columns={definition+'_date':'date'}).set_index('date')
        li_df.append(redact_round_table(df_grouped))
    df_all_time = pd.concat(li_df).stack().reset_index().rename(columns={'level_1':'variable',0:'value'})
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.autofmt_xdate()
    sns.lineplot(x = 'date', y = 'value', hue='variable', data = df_all_time, ax=ax).set_title('New records by month')
    ax.legend().set_title('')

    for group in demographic_covariates + clinical_covariates:
        for definition in definitions:
            df_grouped = df_clean[[definition+'_date',definition,group]].groupby(
                                  [definition+'_date',group]).count().reset_index().rename(columns={definition+'_date':'date'}).set_index(['date', group])
            df_time=redact_round_table(df_grouped).reset_index()
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.autofmt_xdate()
            sns.lineplot(x = 'date', y = definition, hue=group, data = df_time, ax=ax).set_title(f'{definition} recorded by {group} and month')
            ax.legend().set_title('')