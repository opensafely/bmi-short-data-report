import sys
from lib_single_definition import *

definition = sys.argv[1]

############################ CONFIGURE OPTIONS HERE ################################

# Import file
input_path = f'output/joined/input_processed_{definition}.feather'

# Time delta
time_delta = 'M'

# Min/max range
min_value = 4
max_value = 200

# Covariates
demographic_covariates = ['age_band', 'sex', 'ethnicity', 'region', 'imd']
clinical_covariates = ['dementia', 'diabetes', 'hypertension', 'learning_disability']

########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_input = import_clean(
        input_path, definition, time_delta,
        demographic_covariates, clinical_covariates
    )
    patient_counts(
        df_input, definition, demographic_covariates, clinical_covariates
    )
    measurement_counts(
        df_input, definition, demographic_covariates, clinical_covariates
    )
    less_than_min(
        df_input, definition, min_value, demographic_covariates, clinical_covariates
    )
    greater_than_max(
        df_input, definition, min_value, demographic_covariates, clinical_covariates
    )
    cdf(df_input, definition)
    distribution(
        df_input, definition, demographic_covariates, clinical_covariates
    )
    records_over_time(
        df_input, definition, demographic_covariates, clinical_covariates
    )
    means_over_time(
        df_input, definition, demographic_covariates, clinical_covariates
    )

########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    
