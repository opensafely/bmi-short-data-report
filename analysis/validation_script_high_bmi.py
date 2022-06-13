import sys
from lib_single_definition import *

definition = sys.argv[1]

############################ CONFIGURE OPTIONS HERE ################################

# Import file
input_path = f'output/joined/input_processed_{definition}.feather'

# Time delta
time_delta = 'M'

# Min/max range
max_value = 200

# Covariates
demographic_covariates = ['age_band', 'sex', 'ethnicity', 'region', 'imd']
clinical_covariates = ['dementia', 'diabetes', 'hypertension', 'learning_disability']   

# Height and weight variables
if definition == 'computed_bmi':
    other_variables = ['height', 'weight']
else:
    other_variables = ['height_backend', 'weight_backend']
    
########################## SPECIFY ANALYSES TO RUN HERE ##############################

def main():
    df_input = import_clean(
        input_path, definition, time_delta,
        demographic_covariates, clinical_covariates, 
        other_variables,
    )    
    df_high_computed = df_input.loc[df_input[definition] > max_value].rename(
        columns = {
            definition:"high_"+definition, 
            definition+"_date":"high_"+definition+"_date",
            }
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
    