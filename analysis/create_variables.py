from cohortextractor import (
    patients,
)
from codelists import *
from config import start_date, end_date

# Define index date
index_date=start_date

clinical_variables = dict(
    # ----
    # BMI
    # ----
    # Default values in OS - note that these are in inverted order as starting from most recent
    default_bmi_1=patients.most_recent_bmi(
        between=[f"{index_date}", f"{end_date}"],
        minimum_age_at_measurement=18,
        include_measurement_date=True,
        date_format="YYYY-MM-DD",
        return_expectations={
            "date": {"earliest": "2010-02-01", "latest": "2020-01-31"},
            "float": {"distribution": "normal", "mean": 28, "stddev": 8},
            "incidence": 0.80,
        }
    ),
    **{
        f"default_bmi_{n}": patients.most_recent_bmi(
            between=[index_date, f"default_bmi_{n-1}_date_measured - 1 day"],
            minimum_age_at_measurement=18,
            include_measurement_date=True,
            date_format="YYYY-MM-DD",
            return_expectations={
                "date": {"earliest": "2010-02-01", "latest": "2020-01-31"},
                "float": {"distribution": "normal", "mean": 28, "stddev": 8},
                "incidence": 0.80,
            },
        )
        for n in range(2, 30)
    },
    # Recorded
    recorded_bmi_1_date=patients.with_these_clinical_events(
        bmi_code_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="date",
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "date": {"earliest": "index_date"},
        },
    ),
    **{
        f"recorded_bmi_{n}_date": patients.with_these_clinical_events(
            bmi_code_ctv3,
            on_or_after=f"recorded_bmi_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="date",
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "date": {"earliest": "index_date"},
            },
        )
        for n in range(2, 30)
    },
    recorded_bmi_1_value=patients.with_these_clinical_events(
        bmi_code_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="numeric_value",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 22.0, "stddev": 4},
        },
    ),
    **{
        f"recorded_bmi_{n}_value": patients.with_these_clinical_events(
            bmi_code_ctv3,
            on_or_after=f"recorded_bmi_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="numeric_value",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 22.0, "stddev": 4},
            },
        )
        for n in range(2, 30)
    },
    # Weight
    weight_1_date=patients.with_these_clinical_events(
        weight_codes_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="date",
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "date": {"earliest": "index_date"},
        },
    ),
    **{
        f"weight_{n}_date": patients.with_these_clinical_events(
            weight_codes_ctv3,
            on_or_after=f"weight_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="date",
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "date": {"earliest": "index_date"},
            },
        )
        for n in range(2, 30)
    },
    weight_1_value=patients.with_these_clinical_events(
        weight_codes_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="numeric_value",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
        },
    ),
    **{
        f"weight_{n}_value": patients.with_these_clinical_events(
            weight_codes_ctv3,
            on_or_after=f"weight_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="numeric_value",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 70.0, "stddev": 10.0},
            },
        )
        for n in range(2, 30)
    },
    # Height
    height_1_date=patients.with_these_clinical_events(
        height_codes_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="date",
        date_format="YYYY-MM-DD",
        return_expectations={
            "incidence": 0.7,
            "date": {"earliest": "index_date"},
        },
    ),
    **{
        f"height_{n}_date": patients.with_these_clinical_events(
            height_codes_ctv3,
            on_or_after=f"height_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="date",
            date_format="YYYY-MM-DD",
            return_expectations={
                "incidence": 0.7,
                "date": {"earliest": "index_date"},
            },
        )
        for n in range(2, 30)
    },
    height_1_value=patients.with_these_clinical_events(
        height_codes_ctv3,
        on_or_after=f"{index_date}",
        find_first_match_in_period=True,
        returning="numeric_value",
        return_expectations={
            "incidence": 0.7,
            "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
        },
    ),
    **{
        f"height_{n}_value": patients.with_these_clinical_events(
            weight_codes_ctv3,
            on_or_after=f"height_{n-1}_date + 1 day",
            find_first_match_in_period=True,
            returning="numeric_value",
            return_expectations={
                "incidence": 0.7,
                "float": {"distribution": "normal", "mean": 1.65, "stddev": 0.06},
            },
        )
        for n in range(2, 30)
    },
    # -------------------
    # Clinical conditions
    # -------------------
    # Asthma
    # Chronic cardiac disease
    # Chronic kidney disease
    # Chronic liver disease
    # Chronic respiratory disease
    # Dementia
    # Diabetes
    # Housebound
    # Hypertension
    # Learning disability
    learning_disability=patients.with_these_clinical_events(
        wider_ld_codes,
        on_or_before="index_date",
        returning="binary_flag",
        return_expectations={"incidence": 0.01, },
    ),
    # Obesity
    # Psychosis, schizophrenia, or bipolar
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
    # Ethnicity
    ethnicity=patients.categorised_as(
        {
            "0": "DEFAULT",
            "1": "eth='1' OR (NOT eth AND ethnicity_sus='1')",
            "2": "eth='2' OR (NOT eth AND ethnicity_sus='2')",
            "3": "eth='3' OR (NOT eth AND ethnicity_sus='3')",
            "4": "eth='4' OR (NOT eth AND ethnicity_sus='4')",
            "5": "eth='5' OR (NOT eth AND ethnicity_sus='5')",
        },
        return_expectations={
            "category": {
                "ratios": {
                    "1": 0.2,
                    "2": 0.2,
                    "3": 0.2,
                    "4": 0.2,
                    "5": 0.2
                }
            },
            "incidence": 0.4,
        },

        eth=patients.with_these_clinical_events(
            ethnicity_codes,
            returning="category",
            find_last_match_in_period=True,
            include_date_of_match=False,
            return_expectations={
                "category": {
                    "ratios": {
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.2,
                        "4": 0.2,
                        "5": 0.2
                    }
                },
                "incidence": 0.75,
            },
        ),

        # fill missing ethnicity from SUS
        ethnicity_sus=patients.with_ethnicity_from_sus(
            returning="group_6",
            use_most_frequent_code=True,
            return_expectations={
                "category": {
                    "ratios": {
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.2,
                        "4": 0.2,
                        "5": 0.2
                    }
                },
                "incidence": 0.4,
            },
        ),
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
    imd=patients.address_as_of(
        "index_date",
        returning="index_of_multiple_deprivation",
        round_to_nearest=100,
        return_expectations={
            "rate": "universal",
            "category": {
                "ratios": {
                    "100": 0.2,
                    "200": 0.2,
                    "300": 0.2,
                    "400": 0.2,
                    "500": 0.2
                }
            },
        },
    ),
)

# # SNOMED codes for ethnicity
# # Main
# eth2001=patients.with_these_clinical_events(
#     codelists.eth2001,
#     returning="category",
#     find_last_match_in_period=True,
#     on_or_before="index_date",
#     return_expectations={
#         "category": {
#             "ratios": {
#                 "1": 0.5,
#                 "2": 0.25,
#                 "3": 0.125,
#                 "4": 0.0625,
#                 "5": 0.03125,
#                 "6": 0.015625,
#                 "7": 0.0078125,
#                 "8": 0.0078125,
#             }
#         },
#         "rate": "universal",
#     },
# ),
# # Any other ethnicity code
# non_eth2001_dat=patients.with_these_clinical_events(
#     codelists.non_eth2001,
#     returning="date",
#     find_last_match_in_period=True,
#     on_or_before="index_date",
#     date_format="YYYY-MM-DD",
# ),
# # Ethnicity not given - patient refused
# eth_notgiptref_dat=patients.with_these_clinical_events(
#     codelists.eth_notgiptref,
#     returning="date",
#     find_last_match_in_period=True,
#     on_or_before="index_date",
#     date_format="YYYY-MM-DD",
# ),
# # Ethnicity not stated
# eth_notstated_dat=patients.with_these_clinical_events(
#     codelists.eth_notstated,
#     returning="date",
#     find_last_match_in_period=True,
#     on_or_before="index_date",
#     date_format="YYYY-MM-DD",
# ),
# # Ethnicity no record
# eth_norecord_dat=patients.with_these_clinical_events(
#     codelists.eth_norecord,
#     returning="date",
#     find_last_match_in_period=True,
#     on_or_before="index_date",
#     date_format="YYYY-MM-DD",
# ),