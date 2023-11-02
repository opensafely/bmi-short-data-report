from cohortextractor import StudyDefinition, patients, codelist, codelist_from_csv, combine_codelists  # NOQA
from create_variables import demographic_variables, clinical_variables
from codelists import *

from config import start_date, end_date

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
        # Looking at registered patients yearly
        registered=patients.registered_with_one_practice_between(
            start_date, end_date,
            return_expectations={"incidence": 0.9},
        )
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
    **clinical_variables,

)
