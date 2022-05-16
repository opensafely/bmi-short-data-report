# -*- coding: utf-8 -*-
import datetime
import itertools
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import seaborn as sns

from ebmdatalab import charts
from functools import reduce
from matplotlib import pyplot as plt

from lib_phenotype_validation import *

# ########################### CONFIGURE OPTIONS HERE ################################

# Import file
input_path = 'output/data/input_processed.feather'

# Definitions
definitions = ['backend_computed_bmi', 'computed_bmi', 'derived_bmi', 'recorded_bmi']

# Code dictionary
code_dict = {
    'ethnicity': {1:'White', 2:'Mixed', 3:'Asian', 4:'Black', 5:'Other', np.nan: 'Unknown', 0: 'Unknown'},
    'imd': {0: 'Unknown', 1: '1 Most deprived', 2: '2', 3: '3', 4: '4', 5: '5 Least deprived'}
}

# Other variables to include
other_vars = ['height_backend', 'weight_backend', 'height_backend_date', 'weight_backend_date']

# Dates
dates = True
date_min = '2015-03-01'
date_max = '2022-03-01'
time_delta = 'M'

# Min/max range
height_min = 0.5
height_max = 2.8

weight_min = 3
weight_max = 500

bmi_min = 4
bmi_max = 200

# Null value – could be multiple values in a list [0,'0',NA]
null = ['0',0,np.nan]

# Covariates
demographic_covariates = ['age_band', 'sex', 'ethnicity', 'region', 'imd']
clinical_covariates = ['dementia', 'diabetes', 'hypertension', 'learning_disability']

# Output path
output_path = 'histograms'

# ####################################################################################

def redact_round_table2(df_in):
    """Redacts counts <= 5 and rounds counts to nearest 5"""
    df_out = df_in.where(df_in > 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
    return df_out

def num_measurements(df_clean, definitions, demographic_covariates, clinical_covariates):
    df_all = pd.DataFrame(df_clean[definitions].count()).T
    df_all['group'],df_all['subgroup'] = ['all','with records']
    df_all = df_all.set_index(['group','subgroup'])

    # By group
    li_group = []
    for group in demographic_covariates + clinical_covariates:
        li_filled_group = []
        for definition in definitions:
            df_temp = df_clean[['patient_id', definition, group]].drop_duplicates().dropna().reset_index(drop=True)
            df_temp2 = df_temp.groupby(['patient_id',group]).count()
            li_filled_group.append(df_temp2)
        df_reduce = reduce(lambda df1, df2: pd.merge(df1, df2,on=['patient_id',group],how='outer'), li_filled_group).reset_index()
        # Remove list from memory
        del li_filled_group 
        df_reduce2 = df_reduce.sort_values(by=group).drop(columns=['patient_id']).groupby(group).sum().reset_index()
        df_reduce2['group'] = group
        df_reduce2 = df_reduce2.rename(columns={group:'subgroup'})
        li_group.append(df_reduce2)

    df_all_group = pd.concat(li_group, axis=0, ignore_index=True).set_index(['group','subgroup'])

    # Remove list from memory
    del li_group 

    # Redact
    df_append = redact_round_table(df_all.append(df_all_group))

    # Final redaction step
    df_append = df_append.where(~df_append.isna(), '-')  

    df_append.to_csv(f'output/{output_path}/tables/num_measurements.csv')
    
def diff_in_dates(df_clean, meas1, meas2, computed_measure):
    df_clean.loc[df_clean[meas1] == 0, meas1+'_date'] = np.nan
    df_clean.loc[df_clean[meas1] == 0, meas1] = np.nan

    df_clean.loc[df_clean[meas2] == 0, meas2+'_date'] = np.nan
    df_clean.loc[df_clean[meas2] == 0, meas2] = np.nan

    df_clean['date_diff'] = (df_clean[meas1+'_date'] - df_clean[meas2+'_date']).dt.days

    df_bmi_date_diff = df_clean[[computed_measure,'date_diff']].dropna().reset_index(drop=True)

    df_bmi_date_diff['gt_year'] = 0
    df_bmi_date_diff.loc[abs(df_bmi_date_diff.date_diff) > 365, 'gt_year'] = 1

    df_agg = df_bmi_date_diff.groupby('gt_year')[computed_measure].agg(
        ['mean','count']
    )
    df_agg['count'] = df_agg['count'].where(
        df_agg['count'] > 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
    df_agg.to_csv(f'output/{output_path}/tables/data_{computed_measure}_diff_{meas1}_{meas2}.csv')
    
    sns.boxplot(x='gt_year', y=computed_measure, data=df_bmi_date_diff, showfliers=False)
    plt.title(f'Distributions of {computed_measure} with date between {meas1} and {meas2} within a year (0) and greater than a year (1)')
    plt.savefig(f'output/{output_path}/figures/boxplot_{computed_measure}_diff_{meas1}_{meas2}.png', bbox_inches="tight")
    plt.close()
    
    df_gt_year = df_bmi_date_diff.loc[df_bmi_date_diff.gt_year == 1]
    df_hist = pd.DataFrame(pd.cut(df_gt_year[computed_measure], 10).value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
    df_hist[computed_measure] = df_hist[computed_measure].where(
        df_hist[computed_measure] > 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
    df_gt_year['bin']=pd.cut(df_gt_year[computed_measure], bins = 10).astype(str)
    df_in2 = redact_round_table2(df_gt_year.groupby('bin').bin.count())
    df_in2.plot(kind='bar')
    plt.title(f'Distribution of {computed_measure} with date between {meas1} and {meas2} greater than a year (1)')
    df_hist.to_csv(f'output/{output_path}/tables/data_{computed_measure}_gt1yr_date_diff.csv')
    plt.savefig(f'output/{output_path}/figures/hist_{computed_measure}_gt1yr_date_diff.png', bbox_inches="tight")
    plt.close()
    
    df_lt_year = df_bmi_date_diff.loc[df_bmi_date_diff.gt_year == 0]
    df_hist = pd.DataFrame(pd.cut(df_lt_year[computed_measure], 10).value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
    df_hist[computed_measure] = df_hist[computed_measure].where(
        df_hist[computed_measure] > 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
    df_lt_year['bin']=pd.cut(df_lt_year[computed_measure], bins = 10).astype(str)
    df_in2 = redact_round_table2(df_lt_year.groupby('bin').bin.count())
    df_in2.plot(kind='bar')
    df_hist.to_csv(f'output/{output_path}/tables/data_{computed_measure}_lt1yr_date_diff.csv')
    plt.title(f'Distribution of {computed_measure} with date between {meas1} and {meas2} within a year')
    plt.savefig(f'output/{output_path}/figures/hist_{computed_measure}_lt1yr_date_diff.png', bbox_inches="tight")
    plt.close()

def hist(df_in, measure, title, path):
    # 10 bins
    try: 
        df_hist = pd.DataFrame(pd.cut(df_in[measure], 10).value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
        df_hist[measure] = df_hist[measure].where(
            df_hist[measure]> 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
        df_in['bin']=pd.cut(df_in[measure], bins = 10).astype(str)
        df_in2 = redact_round_table2(df_in.groupby('bin').bin.count())
        df_in2.plot(kind='bar')
        plt.title(title)
        df_hist.to_csv(f'output/{output_path}/tables/hist_data_{path}.csv')
        plt.savefig(f'output/{output_path}/figures/hist_{path}.png', bbox_inches="tight")
        plt.close()
    except:
        pass

def recent_to_now(df_clean, definitions):
    curr_time = pd.to_datetime("now")
    for definition in definitions:
        df_temp = df_clean[['patient_id', definition+'_date']].sort_values(by=['patient_id', definition+'_date'], ascending=False)
        df_temp2 = df_temp.drop_duplicates(subset='patient_id')
        df_temp2[definition+'_date_diff'] = (curr_time-df_temp2[definition+'_date']).dt.days
        hist(df_temp2, definition+'_date_diff', f'Days between now and most recent {definition}', f'most_recent_{definition}')
    
# ######################### SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_clean = import_clean(input_path, definitions, other_vars, demographic_covariates, 
                            clinical_covariates, null, date_min, date_max, 
                            time_delta, output_path, code_dict, dates)
    # Number of measurements
    num_measurements(df_clean, definitions, demographic_covariates, clinical_covariates)
    # Distribution of measurement to now
    recent_to_now(df_clean, definitions)
    # Distribution of BMI for small vs. large height-weight time difference
    diff_in_dates(df_clean, 'height_backend', 'weight_backend', 'backend_computed_bmi')
    
# ########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()

