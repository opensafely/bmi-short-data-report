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
    **clinical_variables,

)
