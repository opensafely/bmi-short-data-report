from cohortextractor import StudyDefinition, patients, codelist, codelist_from_csv, combine_codelists  # NOQA
from create_variables import *
from codelists import *

from config import *

study = StudyDefinition(
    index_date=index_date,
    default_expectations={
        "date": {"earliest": '2015-01-01', "latest": index_date},
        "rate": "uniform",
        "incidence": 0.65,
    },
    population=patients.satisfying(
        """
        (sex = "M" OR sex = "F") AND
        (age >= 18 AND age <= 110)
        """,
        age=patients.age_as_of(
            "index_date",
            return_expectations={
                "rate": "universal",
                "int": {"distribution": "population_ages"},
            },
        ),
        sex=patients.sex(
            return_expectations={
                "rate": "universal",
                "category": {"ratios": {"M": 0.5, "F": 0.5}},
            }
        ),
    ),

    **derived_bmi_variables,

)
