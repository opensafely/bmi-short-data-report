from lib_phenotype_validation import *

############################ CONFIGURE OPTIONS HERE ################################

# Import file
input_path = 'output/data/input_processed.feather'

# Definitions
definitions = ['weight','weight_backend']

# Code dictionary
code_dict = {
    'ethnicity': {1:'White', 2:'Mixed', 3:'Asian', 4:'Black', 5:'Other', np.nan: 'Unknown', 0: 'Unknown'},
    'imd': {0: 'Unknown', 1: '1 Most deprived', 2: '2', 3: '3', 4: '4', 5: '5 Least deprived'}
}

# Other variables to include
other_vars = []

# Dates
dates = False
date_min = ''
date_max = ''
time_delta = ''

# Min/max range
min_range = 3
max_range = 500

# Null value – could be multiple values in a list [0,'0',NA]
null = ['0',0,np.nan]

# Covariates
demographic_covariates = ['age_band', 'sex', 'ethnicity', 'region', 'imd']
clinical_covariates = ['dementia', 'diabetes', 'hypertension', 'learning_disability']

# Output path
output_path = 'phenotype_validation_weight'
        
########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_clean = import_clean(input_path, definitions, other_vars, demographic_covariates, 
                        clinical_covariates, null, date_min, date_max, 
                        time_delta, output_path, code_dict, dates)
    # Count patients with records
    patient_counts(df_clean, definitions, demographic_covariates, clinical_covariates, output_path)
    # Count patients without records
    patient_counts(df_clean, definitions, demographic_covariates, clinical_covariates, output_path, missing=True)
    # Report distributions
    report_distribution(df_clean, definitions, len(definitions), output_path, group='')
    for group in demographic_covariates + clinical_covariates:
        report_distribution(df_clean, definitions, len(definitions), output_path, group)
    # Report out-of-range values
    report_out_of_range(df_clean, definitions, min_range, max_range, len(definitions), null, output_path, group='')
    for group in demographic_covariates + clinical_covariates:
        report_out_of_range(df_clean, definitions, min_range, max_range, len(definitions), null, output_path, group)
        
########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    