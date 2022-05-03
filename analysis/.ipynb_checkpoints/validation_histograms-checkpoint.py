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

def import_clean(input_path, definitions, other_vars, demographic_covariates, 
                 clinical_covariates, null, date_min, date_max, 
                 time_delta, output_path, code_dict='', dates=False):
    # Import
    df_import = pd.read_feather(input_path)
    # Dates
    if dates==True:
        date_vars = [definition+'_date' for definition in definitions]
        # Create variable that captures difference in measurement dates
        date_diff_vars = []
        # Define start and end dates
        start_date = datetime.datetime.strptime(date_min, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(date_max, '%Y-%m-%d')
        for definition in definitions:
            # Remove OpenSAFELY null dates 
            df_import.loc[df_import[definition+'_date'] == '1900-01-01', definition+'_date'] = np.nan
            # Limit to period of interest             
            df_import[definition+'_date'] = pd.to_datetime(df_import[definition+'_date'])
            df_import.loc[df_import[definition+'_date'] < start_date, definition+'_date'] = np.nan
            df_import.loc[df_import[definition+'_date'] > end_date, definition+'_date'] = np.nan
            # Remove the measurement if outside the date parameters
            df_import.loc[df_import[definition+'_date'].isna(), definition] = np.nan
            df_import 
            # Create difference between measurement dates
            df_import[definition+'_date'] = df_import[definition+'_date'].dt.to_period(time_delta).dt.to_timestamp()
            df_import = df_import.sort_values(by=['patient_id',definition+'_date'])
            df_import['date_diff_' + definition] = round(df_import.groupby('patient_id')[definition+'_date'].diff() / np.timedelta64(1, time_delta))
            date_diff_vars.append('date_diff_' + definition)
    else: 
        date_vars = []
        date_diff_vars = []
    # Codes
    if code_dict!='':
        for key in code_dict:
            df_import[key] = df_import[key].astype(float)
            df_import[key] = df_import[key].replace(code_dict[key])
    
    # Subset to relevant columns
    df_clean = df_import[['patient_id'] + definitions + other_vars + date_vars + date_diff_vars + demographic_covariates + clinical_covariates]
    # Limit to relevant date range
    df_clean = df_clean.sort_values(by='patient_id').reset_index(drop=True)
    # Set null values to nan
    for definition in definitions: 
        df_clean.loc[df_clean[definition].isin(null), definition] = np.nan
     # Create order for categorical variables
    for group in demographic_covariates + clinical_covariates:
        if df_clean[group].dtype.name == 'category':
            li_order = sorted(df_clean[group].dropna().unique().tolist())
            df_clean[group] = df_clean[group].cat.reorder_categories(li_order, ordered=True)
    # Mark patients with value filled/missing for each definition
    li_filled = []
    for definition in definitions:
        df_fill = pd.DataFrame(df_clean.groupby("patient_id")[definition].any().astype('int')).rename(
            columns={definition:definition+'_filled'}
        )
        df_fill[definition+'_missing'] = 1-df_fill[definition+'_filled']
        li_filled.append(df_fill)

    df_filled = pd.concat(li_filled, axis=1)
    # Remove list from memory
    del li_filled  
    df_clean = df_clean.merge(df_filled, on='patient_id')
    
    # Flag all filled/all missing
    li_col_filled = [col for col in df_clean.columns if col.endswith('_filled')]
    li_col_missing = [col for col in df_clean.columns if col.endswith('_missing')]
    df_clean['all_filled'] = (df_clean[li_col_filled].sum(axis=1) == len(definitions)).astype(int)
    df_clean['all_missing'] = (df_clean[li_col_missing].sum(axis=1) == len(definitions)).astype(int)
    
    # Check whether output paths exist or not, create if missing
    path_tables = f'output/{output_path}/tables'
    path_figures = f'output/{output_path}/figures'
    li_filepaths = [path_tables, path_figures]

    for filepath in li_filepaths:
        exists = os.path.exists(filepath)
        if not exists:
            os.makedirs(filepath)

    return df_clean

def hist(df_in, measure, title, path):
    # 10 bins
    try: 
        df_hist = pd.DataFrame(pd.cut(df_in[measure], 10).value_counts().sort_index()).reset_index().rename(columns={'index':'intervals'})
        plt.hist(df_in[measure], 10)
        plt.title(title)
        df_hist.to_csv(f'output/{output_path}/tables/hist_data_{path}.csv')
        plt.savefig(f'output/{output_path}/figures/hist_{path}.png')
    except:
        pass
    
def q_n(x, pct):
    return x.quantile(pct)

def subset_q(df_clean, measure, pct, less=True):
    if pct < 1:
        p = q_n(df_clean[measure], pct)
    else:
        less = False
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
    df_p95 = subset_q(df_clean, 'backend_bmi', 0.95)
    hist(df_p95, 'weight_backend', 'Weight for Patients with 5% Highest BMI', 'weight_p95')
    hist(df_p95, 'height_backend', 'Height for Patients with 5% Highest BMI', 'height_p95')
    # Lowest 10%
    df_p10 = subset_q(df_clean, 'backend_bmi', 0.1)
    hist(df_p10, 'weight_backend', 'Weight for Patients with 10% Lowest BMI', 'weight_p10')
    hist(df_p10, 'height_backend', 'Height for Patients with 10% Lowest BMI', 'height_p10')
    # Highest 10%
    df_p90 = subset_q(df_clean, 'backend_bmi', 0.9)
    hist(df_p90, 'weight_backend', 'Weight for Patients with 10% Highest BMI', 'weight_p90')
    hist(df_p90, 'height_backend', 'Height for Patients with 10% Highest BMI', 'height_p90')
    # Lower than minimum expected range
    df_min = subset_q(df_clean, 'backend_bmi', 4)
    hist(df_min, 'weight_backend', 'Weight for Patients with BMI < 4', 'weight_lt_min')
    hist(df_min, 'height_backend', 'Height for Patients with BMI < 4', 'height_lt_min')
    # Higher than maximum expected range
    df_max = subset_q(df_clean, 'backend_bmi', 200, False)
    hist(df_max, 'weight_backend', 'Weight for Patients with BMI > 200', 'weight_gt_max')
    hist(df_max, 'height_backend', 'Height for Patients with BMI > 200', 'height_gt_max')
        
########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    