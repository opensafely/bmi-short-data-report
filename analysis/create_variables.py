from cohortextractor import (
    patients,
)
from codelists import *

from config import end_date

derived_bmi_variables = dict(
    # BMI using OpenSAFELY algorithm - returns latest in period
    derived_bmi1=patients.most_recent_bmi(
        on_or_before=end_date,
        minimum_age_at_measurement=18,
        include_measurement_date=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "date": {"earliest": "2010-02-01", "latest": "2022-04-01"},
            "float": {"distribution": "normal", "mean": 28, "stddev": 8},
            "incidence": 0.8,
        }
    ),
    **{
        f"derived_bmi{n}": patients.most_recent_bmi(
            on_or_before=f"derived_bmi{n-1}_date_measured - 1 day",
            minimum_age_at_measurement=18,
            include_measurement_date=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "date": {"earliest": "2010-02-01", "latest": "2022-04-01"},
                "float": {"distribution": "normal", "mean": 28, "stddev": 8},
                "incidence": 0.80,
            },
        )
        for n in range(2, 11)
    },
)

recorded_bmi_variables = dict(
    # Recorded BMI (coded values)
    recorded_bmi1=patients.with_these_clinical_events(
        bmi_code_snomed,
        on_or_before=end_date,
        find_last_match_in_period=True,
        returning="numeric_value",
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 22.0, "stddev": 4},
        },
    ),
    **{
        f"recorded_bmi{n}": patients.with_these_clinical_events(
            bmi_code_snomed,
            on_or_before=f"recorded_bmi{n-1}_date - 1 day",
            find_last_match_in_period=True,
            returning="numeric_value",
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 22.0, "stddev": 4},
            },
        )
        for n in range(2, 11)
    },  
)

snomed_hw_variables = dict(
    # Weight (SNOMED)
    weight1=patients.with_these_clinical_events(
        weight_codes_snomed,
        on_or_before=end_date,
        find_last_match_in_period=True,
        returning="numeric_value",
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
        },
    ),
    **{
        f"weight{n}": patients.with_these_clinical_events(
            weight_codes_snomed,
            on_or_before=f"weight{n-1}_date - 1 day",
            find_last_match_in_period=True,
            returning="numeric_value",
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
            },
        )
        for n in range(2, 11)
    },
    **{
        f"weight_age{n}": patients.age_as_of(
            f"weight{n}_date",
            return_expectations={
                "rate" : "universal",
                "int" : {"distribution" : "population_ages"}
            }
        )
        for n in range(1,11)
    },
    # Height (SNOMED)
    height1=patients.with_these_clinical_events(
        height_codes_snomed,
        on_or_before=end_date,
        find_last_match_in_period=True,
        returning="numeric_value",
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
        },
    ),
    **{
        f"height{n}": patients.with_these_clinical_events(
            height_codes_snomed,
            on_or_before=f"height{n-1}_date - 1 day",
            find_last_match_in_period=True,
            returning="numeric_value",
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
            },
        )
        for n in range(2, 11)
    },
    **{
        f"height_age{n}": patients.age_as_of(
            f"height{n}_date",
            return_expectations={
                "rate" : "universal",
                "int" : {"distribution" : "population_ages"}
            }
        )
        for n in range(1,11)
    },
)

ctv3_hw_variables = dict(
    # Weight (CTV3 definition in backend)
    weight_backend1=patients.with_these_clinical_events(
        weight_codes_backend,
        on_or_before=end_date,
        find_last_match_in_period=True,
        returning="numeric_value",
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
        },
    ),
    **{
        f"weight_backend{n}": patients.with_these_clinical_events(
            weight_codes_backend,
            on_or_before=f"weight_backend{n-1}_date - 1 day",
            find_last_match_in_period=True,
            returning="numeric_value",
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
            },
        )
        for n in range(2, 11)
    },
    **{
        f"weight_backend_age{n}": patients.age_as_of(
            f"weight_backend{n}_date",
            return_expectations={
                "rate" : "universal",
                "int" : {"distribution" : "population_ages"}
            }
        )
        for n in range(1,11)
    },
    # Height (CTV3 definition in backend)
    height_backend1=patients.with_these_clinical_events(
        height_codes_backend,
        on_or_before=end_date,
        find_last_match_in_period=True,
        returning="numeric_value",
        include_date_of_match=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
        },
    ),
    **{
        f"height_backend{n}": patients.with_these_clinical_events(
            height_codes_backend,
            on_or_before=f"height_backend{n-1}_date - 1 day",
            find_last_match_in_period=True,
            returning="numeric_value",
            include_date_of_match=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
            },
        )
        for n in range(2, 11)
    },
    **{
        f"height_backend_age{n}": patients.age_as_of(
            f"height_backend{n}_date",
            return_expectations={
                "rate" : "universal",
                "int" : {"distribution" : "population_ages"}
            }
        )
        for n in range(1,11)
    },
)

clinical_variables = dict(
    # -------------------
    # Clinical conditions
    # -------------------
    # Chronic cardiac disease
    chronic_cardiac_disease=patients.with_these_clinical_events(
        chronic_cardiac_dis_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Chronic kidney disease
    chronic_kidney_disease=patients.with_these_clinical_events(
        chronic_kidney_dis_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Chronic liver disease
    chronic_liver_disease=patients.with_these_clinical_events(
        chronic_liver_dis_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Chronic respiratory disease
    chronic_respiratory_disease=patients.with_these_clinical_events(
        chronic_respiratory_dis_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Cancer (Haemotological)
    cancer_haem=patients.with_these_clinical_events(
        cancer_haem_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Cancer (Lung)
    cancer_lung=patients.with_these_clinical_events(
        cancer_lung_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Cancer (Other)
    cancer_other=patients.with_these_clinical_events(
        cancer_other_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Dementia
    dementia=patients.with_these_clinical_events(
        dementia_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Diabetes
    diabetes=patients.with_these_clinical_events(
        diabetes_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Housebound
    housebound=patients.with_these_clinical_events(
        housebound_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Hypertension
    hypertension=patients.with_these_clinical_events(
        hypertension_codes,
        between=["index_date - 2 years", "index_date - 1 day"],
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Learning disability
    learning_disability=patients.with_these_clinical_events(
        wider_ld_codes,
        on_or_before="index_date - 1 day",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
)

demographic_variables = dict(
    # Age
    age=patients.age_as_of(
        "index_date",
        return_expectations={
            "rate": "universal",
            "int": {"distribution": "population_ages"},
        },
    ),
    # Age band
    age_band=patients.categorised_as(
        {
            "missing": "DEFAULT",
            "0-19": """ age >= 0 AND age < 20""",
            "20-29": """ age >=  20 AND age < 30""",
            "30-39": """ age >=  30 AND age < 40""",
            "40-49": """ age >=  40 AND age < 50""",
            "50-59": """ age >=  50 AND age < 60""",
            "60-69": """ age >=  60 AND age < 70""",
            "70-79": """ age >=  70 AND age < 80""",
            "80+": """ age >=  80 AND age < 120""",
        },
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "0-19": 0.125,
                    "20-29": 0.125,
                    "30-39": 0.125,
                    "40-49": 0.125,
                    "50-59": 0.125,
                    "60-69": 0.125,
                    "70-79": 0.125,
                    "80+": 0.125,
                }
            },
        },

    ),
    # Sex
    sex=patients.sex(
        return_expectations={
            "rate": "universal",
            "category": {"ratios": {"M": 0.5, "F": 0.5}},
        }
    ),
    # Practice
    practice=patients.registered_practice_as_of(
        "index_date",
        returning="pseudo_id",
        return_expectations={
            "int": {
                "distribution": "normal", "mean": 25, "stddev": 5
            }, "incidence": 0.5}
    ),
    # Ethnicity
    ethnicity=patients.with_these_clinical_events(
        ethnicity_snomed,
        returning="category",
        find_last_match_in_period=True,
        on_or_before="index_date",
        return_expectations={
            "category": {
                "ratios": {
                    "1": 0.5,
                    "2": 0.25,
                    "3": 0.14,
                    "4": 0.07,
                    "5": 0.04,
                }
            },
            "rate": "universal",
        },
    ),
    # Region
    region=patients.registered_practice_as_of(
        "index_date",
        returning="nuts1_region_name",
        return_expectations={"category": {"ratios": {
            "North East": 0.1,
            "North West": 0.1,
            "Yorkshire and the Humber": 0.1,
            "East Midlands": 0.1,
            "West Midlands": 0.1,
            "East of England": 0.1,
            "London": 0.2,
            "South East": 0.2, }}}
    ),
    # IMD
    imd = patients.categorised_as(
        {
            "0": "DEFAULT",
            "1": """index_of_multiple_deprivation >=1 AND index_of_multiple_deprivation < 32844*1/5""",
            "2": """index_of_multiple_deprivation >= 32844*1/5 AND index_of_multiple_deprivation < 32844*2/5""",
            "3": """index_of_multiple_deprivation >= 32844*2/5 AND index_of_multiple_deprivation < 32844*3/5""",
            "4": """index_of_multiple_deprivation >= 32844*3/5 AND index_of_multiple_deprivation < 32844*4/5""",
            "5": """index_of_multiple_deprivation >= 32844*4/5 """,
        },
        index_of_multiple_deprivation = patients.address_as_of(
            "index_date",
            returning = "index_of_multiple_deprivation",
            round_to_nearest = 100,
        ),
        return_expectations = {
            "rate": "universal",
            "category": {
                "ratios": {
                    "0": 0.01,
                    "1": 0.20,
                    "2": 0.20,
                    "3": 0.20,
                    "4": 0.20,
                    "5": 0.19,
                }
            },
        },
    ),
)
