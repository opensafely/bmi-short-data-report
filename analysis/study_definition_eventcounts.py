from cohortextractor import StudyDefinition, patients
from create_variables import demographic_variables, clinical_variables
from config import *
from codelists import (
    bmi_code_snomed,
    weight_codes_snomed,
    height_codes_snomed,
    weight_codes_backend,
    height_codes_backend,
)

study = StudyDefinition(
    index_date=index_date,
    default_expectations={
        "date": {"earliest": '2015-01-01', "latest": index_date},
        "rate": "uniform",
        "incidence": 0.65,
    },
    **population_spec,
    # Deregistration date (to censor these patients in longitudinal analyses)
    dereg_date=patients.date_deregistered_from_all_supported_practices(
        on_or_before=index_date,
        date_format="YYYY-MM-DD",
        return_expectations={"date": {"earliest": "index_date"}},
    ),
    # Death date (to censor these patients in longitudinal analyses)
    died_date_ons=patients.died_from_any_cause(
        on_or_before=index_date,
        returning="date_of_death",
        date_format="YYYY-MM-DD",
        return_expectations={"date": {"earliest": "index_date"}},
    ),
    **demographic_variables,
    recorded_bmi_count=patients.with_these_clinical_events(
        bmi_code_snomed,
        on_or_before=index_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    weight_snomed_count=patients.with_these_clinical_events(
        weight_codes_snomed,
        on_or_before=index_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    weight_backend_count=patients.with_these_clinical_events(
        weight_codes_backend,
        on_or_before=index_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    height_snomed_count=patients.with_these_clinical_events(
        height_codes_snomed,
        on_or_before=index_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    height_backend_count=patients.with_these_clinical_events(
        height_codes_backend,
        on_or_before=index_date,
        returning="number_of_matches_in_period",
        return_expectations={
            "incidence": 0.7,
            "int": {"distribution": "normal", "mean": 5, "stddev": 4},
        },
    ),
    **{
        k: v
        for k, v in clinical_variables.items()
        if not k.startswith("height")
        and (not k.startswith("weight"))
        and ("bmi" not in k)
    }
)
