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

############################ CONFIGURE OPTIONS HERE ################################

# Import file
input_path = 'output/data/input_processed.feather'

# Definitions
definitions = ['backend_bmi', 'computed_bmi', 'derived_bmi', 'recorded_bmi']

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

#####################################################################################

def redact_round_table2(df_in):
    """Redacts counts <= 5 and rounds counts to nearest 5"""
    df_out = df_in.where(df_in > 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
    return df_out

def hist(df_in, measure, title, path):
    # 30 bins
    try: 
        df_hist = pd.DataFrame(pd.cut(df_in[measure], 30).value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
        df_hist[measure] = df_hist[measure].where(
            df_hist[measure]> 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
        df_in['bin']=pd.cut(df_in[measure], bins = 30).astype(str)
        df_in2 = redact_round_table2(df_in.groupby('bin').bin.count())
        df_in2.plot(kind='bar')
        plt.title(title)
        df_hist.to_csv(f'output/{output_path}/tables/hist_data_{path}.csv')
        plt.savefig(f'output/{output_path}/figures/hist_{path}.png', bbox_inches="tight")
        plt.close()
    except:
        pass

def decile_plot(df_in, measure, title, path):
    try:
        df_dec = pd.DataFrame(pd.qcut(df_in[measure], 10, duplicates='drop').value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
        df_dec[measure] = df_dec[measure].where(
            df_dec[measure]> 5, np.nan).apply(lambda x: 5 * round(x/5) if ~np.isnan(x) else x)
        df_in['bin'] = pd.qcut(df_in[measure], 10, duplicates='drop').astype(str)
        df_in2 = redact_round_table2(df_in.groupby('bin').bin.count())
        df_in2.plot(kind='bar')
        plt.title(title)
        df_dec.to_csv(f'output/{output_path}/tables/decile_plot_{path}.csv')
        plt.savefig(f'output/{output_path}/figures/decile_plot_{path}.png', bbox_inches="tight")
        plt.close()
    except:
        pass
    
def q_n(x, pct):
    return x.quantile(pct)

def subset_q(df_in, measure, pct, less=True):
    # Drop the top 5 highest in measure (outliers)
    df_clean = df_in.loc[~df_in[measure].isin(df_in[measure].nlargest(n=5).tolist())]
    if pct < 1:
        p = q_n(df_clean[measure], pct)
    else:
        p = pct
    if less == True:
        df_p = df_clean.loc[df_clean[measure] < p]
    else: 
        df_p = df_clean.loc[df_clean[measure] > p]
    return df_p

########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_clean = import_clean(input_path, definitions, other_vars, demographic_covariates, 
                            clinical_covariates, null, date_min, date_max, 
                            time_delta, output_path, code_dict, dates)
    
    ### Create histograms
    # All population
    hist(df_clean, 'weight_backend', 'Weight (CTV3 Codes Used in OpenSAFELY-TPP Backend)', 'weight_all')
    hist(df_clean, 'height_backend', 'Height (CTV3 Codes Used in OpenSAFELY-TPP Backend)', 'height_all')
    # Lowest 5%
    df_p5 = subset_q(df_clean, 'backend_bmi', 0.05)
    hist(df_p5, 'weight_backend', 'Weight for Patients with 5% Lowest BMI', 'weight_p5')
    hist(df_p5, 'height_backend', 'Height for Patients with 5% Lowest BMI', 'height_p5')
    # Highest 5%
    df_p95 = subset_q(df_clean, 'backend_bmi', 0.95, less=False)
    hist(df_p95, 'weight_backend', 'Weight for Patients with 5% Highest BMI', 'weight_p95')
    hist(df_p95, 'height_backend', 'Height for Patients with 5% Highest BMI', 'height_p95')
    # Lower than minimum expected range
    df_min = subset_q(df_clean, 'backend_bmi', 4)
    hist(df_min, 'weight_backend', 'Weight for Patients with BMI < 4', 'weight_lt_min')
    hist(df_min, 'height_backend', 'Height for Patients with BMI < 4', 'height_lt_min')
    # Higher than maximum expected range
    df_max = subset_q(df_clean, 'backend_bmi', 200, False)
    hist(df_max, 'weight_backend', 'Weight for Patients with BMI > 200', 'weight_gt_max')
    hist(df_max, 'height_backend', 'Height for Patients with BMI > 200', 'height_gt_max')
    # Histogram of negative values
    df_height_neg = df_clean.loc[df_clean['height_backend'] < 0]
    hist(df_height_neg, 'height_backend', 'Distribution of Negative Heights', 'height_negative')
    df_weight_neg = df_clean.loc[df_clean['weight_backend'] < 0]
    hist(df_weight_neg, 'weight_backend', 'Distribution of Negative Weights', 'weight_negative')
    ### Create decile plots
    # All population
    decile_plot(df_clean, 'weight_backend', 'Weight (CTV3 Codes Used in OpenSAFELY-TPP Backend)', 'weight_all')
    decile_plot(df_clean, 'height_backend', 'Height (CTV3 Codes Used in OpenSAFELY-TPP Backend)', 'height_all')
    # Lowest 5%
    decile_plot(df_p5, 'weight_backend', 'Weight for Patients with 5% Lowest BMI', 'weight_p5')
    decile_plot(df_p5, 'height_backend', 'Height for Patients with 5% Lowest BMI', 'height_p5')
    # Highest 5%
    decile_plot(df_p95, 'weight_backend', 'Weight for Patients with 5% Highest BMI', 'weight_p95')
    decile_plot(df_p95, 'height_backend', 'Height for Patients with 5% Highest BMI', 'height_p95')
    # Lower than minimum expected range
    decile_plot(df_min, 'weight_backend', 'Weight for Patients with BMI < 4', 'weight_lt_min')
    decile_plot(df_min, 'height_backend', 'Height for Patients with BMI < 4', 'height_lt_min')
    # Higher than maximum expected range
    decile_plot(df_max, 'weight_backend', 'Weight for Patients with BMI > 200', 'weight_gt_max')
    decile_plot(df_max, 'height_backend', 'Height for Patients with BMI > 200', 'height_gt_max')
    # Negative values
    decile_plot(df_height_neg, 'height_backend', 'Distribution of Negative Heights', 'height_negative')
    decile_plot(df_weight_neg, 'weight_backend', 'Distribution of Negative Weights', 'weight_negative')
    
########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    