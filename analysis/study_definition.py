from cohortextractor import StudyDefinition, patients, codelist, codelist_from_csv, combine_codelists  # NOQA
from create_variables import demographic_variables, clinical_variables
from config import start_date, end_date
from codelists import *

study = StudyDefinition(
    index_date=start_date,
    default_expectations={
        "date": {"earliest": start_date, "latest": end_date},
        "rate": "uniform",
        "incidence": 0.65,
    },
    population=patients.satisfying(
        """
        registered AND
        (NOT died) AND
        (sex = "M" OR sex = "M") AND
        (age >= 18 AND age <= 110) AND
        (region != "")
        """,
        registered=patients.registered_as_of(
            "index_date",
            return_expectations={"incidence": 0.9},
        ),
        died=patients.died_from_any_cause(
            on_or_before=end_date,
            returning="binary_flag",
            return_expectations={"incidence": 0.1}
        )
    ),

    **demographic_variables,
    **clinical_variables,

)
