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
    **derived_bmi_variables,

)
