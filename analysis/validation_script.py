from lib_phenotype_validation import *

############################ CONFIGURE OPTIONS HERE ################################

# Import file
input_path = 'output/data/input_processed.feather'

# Definitions
definitions = ['backend_computed_bmi', 'computed_bmi', 'derived_bmi', 'recorded_bmi']

# Code dictionary
code_dict = {
    "ethnicity": {
        1: "White",
        2: "Mixed",
        3: "Asian",
        4: "Black",
        5: "Other",
        np.nan: "Unknown",
        0: "Unknown",
    },
    "imd": {
        0: "Unknown",
        1: "1 Most deprived",
        2: "2",
        3: "3",
        4: "4",
        5: "5 Least deprived",
    },
}

# Other variables to include
other_vars = [
    "height_backend",
    "weight_backend",
]

# Dates
dates = True
date_min = '2015-03-01'
date_max = '2022-03-01'
time_delta = 'M'

# Min/max range
min_range = 4
max_range = 200

# Null value – could be multiple values in a list [0,'0',NA]
null = [0]

# Covariates
demographic_covariates = ['age_band', 'sex', 'ethnicity', 'region', 'imd']
clinical_covariates = ['dementia', 'diabetes', 'hypertension', 'learning_disability']

# Output path
output_path = 'phenotype_validation_bmi'
        
########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_clean = import_clean(
        input_path, definitions, other_vars, demographic_covariates, 
        clinical_covariates, null, date_min, date_max, 
        time_delta, output_path, code_dict, dates
    )
    # Count patients with records
    patient_counts(
        df_clean, definitions, demographic_covariates, 
        clinical_covariates, output_path
    )
    # Count patients without records
    patient_counts(
        df_clean, definitions, demographic_covariates, 
        clinical_covariates, output_path, missing=True
    )
    # Count number of measurements 
    num_measurements(
        df_clean, definitions, demographic_covariates, 
        clinical_covariates, output_path
    )
    # Report distributions
    report_distribution(df_clean, definitions, output_path, group='')
    for group in demographic_covariates + clinical_covariates:
        report_distribution(df_clean, definitions, output_path, group)
    # Count values out of range
    report_out_of_range(
        df_clean, definitions, min_range, 
        null, output_path, group='', less_than=True
    )
    report_out_of_range(
        df_clean, definitions, max_range, 
        null, output_path, group='', less_than=False
    )
    for group in demographic_covariates + clinical_covariates:
        report_out_of_range(
            df_clean, definitions, min_range, 
            null, output_path, group, less_than=True
        )
        report_out_of_range(
            df_clean, definitions, max_range, 
            null, output_path, group, less_than=False
        )
    # Report new records over time
    records_over_time(
        df_clean, definitions, demographic_covariates, 
        clinical_covariates, output_path,''
    )
    # Report time between measurement and now 
    recent_to_now(df_clean, definitions, output_path)
    # Report means over time
    means_over_time(
        df_clean, definitions, demographic_covariates, 
        clinical_covariates, output_path,''
    )
    # Report number of records and means over time of high computed BMI 
    df_high_computed = df_clean.loc[df_clean['backend_computed_bmi'] > max_range]
    records_over_time(
        df_high_computed, ['backend_computed_bmi'], demographic_covariates, 
        clinical_covariates, output_path,'_greater_than_max'
    )
    means_over_time(
        df_high_computed, ['backend_computed_bmi'], demographic_covariates, 
        clinical_covariates, output_path,'_greater_than_max'
    )
    # Report distribution of height and weight for high computed BMI
    count_table(
        df_high_computed, 'height_backend', 
        output_path, 'height_high_computed_bmi'
    )
    cdf(
        df_high_computed, 'height_backend', 
        output_path, 'height_high_computed_bmi'
    )
    count_table(
        df_high_computed, 'weight_backend', 
        output_path, 'weight_high_computed_bmi'
    )
    cdf(
        df_high_computed, 'weight_backend', 
        output_path, 'weight_high_computed_bmi'
    )
        
########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    
