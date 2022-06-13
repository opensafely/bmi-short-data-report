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
    recent_to_now(df_input, definition)
    if (definition == "backend_computed_bmi") | (definition == "computed_bmi"):
        # Report number of records and means over time of high computed BMI 
        df_high_computed = df_input.loc[df_input[definition] > max_range].rename(
            columns = {definition:"high_"+definition}
        )
        # Check whether output paths exist or not, create if missing
        filepath = f'output/validation/tables/high_{definition}'
        exists = os.path.exists(filepath)
        if not exists:
            os.makedirs(filepath)
        records_over_time(
            df_high_computed, "high_"+definition, demographic_covariates, clinical_covariates
        )
        means_over_time(
            df_high_computed, "high_"+definition, demographic_covariates, clinical_covariates
        )
        # Report distribution of height and weight for high computed BMI
        if definition == "backend_computed_bmi":
            count_table(df_high_computed, "height_backend", f"high_{definition}")
            cdf(df_high_computed, "height_backend", f"high_{definition}")
            count_table(df_high_computed, "weight_backend", f"high_{definition}")
            cdf(df_high_computed, "weight_backend", f"high_{definition}")
        else:
            count_table(df_high_computed, "height", f"high_{definition}")
            cdf(df_high_computed, "height", f"high_{definition}")
            count_table(df_high_computed, "weight", f"high_{definition}")
            cdf(df_high_computed, "weight", f"high_{definition}")

########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
    
