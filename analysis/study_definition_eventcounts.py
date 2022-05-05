from cohortextractor import StudyDefinition, patients
from create_variables import demographic_variables
from config import start_date, end_date
from codelists import bmi_code_snomed, weight_codes_snomed, height_codes_snomed

study = StudyDefinition(
    index_date=start_date,
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.65,
    },
    population=patients.satisfying(
        """
        (sex = "M" OR sex = "F") AND
        (age >= 18 AND age <= 110)
        """,
    ),

    # Deregistration date (to censor these patients in longitudinal analyses)
    dereg_date=patients.date_deregistered_from_all_supported_practices(
        between=[start_date, end_date],
        date_format="YYYY-MM-DD",
        return_expectations={"date": {"earliest": "index_date"}},
    ),

    # Death date (to censor these patients in longitudinal analyses)
    died_date_ons=patients.died_from_any_cause(
        between=[start_date, end_date],
        returning="date_of_death",
        date_format="YYYY-MM-DD",
        return_expectations={"date": {"earliest": "index_date"}},
    ),

    **demographic_variables,
    recorded_bmi_count=patients.with_these_clinical_events(
        bmi_code_snomed,
        on_or_before=end_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    weight_count=patients.with_these_clinical_events(
        weight_codes_snomed,
        on_or_before=end_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    height_count=patients.with_these_clinical_events(
        height_codes_snomed,
        on_or_before=end_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
)
