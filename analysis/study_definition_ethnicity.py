from cohortextractor import (
    StudyDefinition,
    patients,
)
from config import end_date

from codelists import ethnicity_snomed

study = StudyDefinition(
    default_expectations={
        "date": {"earliest": "1900-01-01", "latest": "today"},
        "rate": "uniform",
    },
    index_date=end_date,
    population=patients.all(),
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
                    "3": 0.125,
                    "4": 0.0625,
                    "5": 0.03125,
                    "6": 0.015625,
                    "7": 0.0078125,
                    "8": 0.0078125,
                }
            },
            "rate": "universal",
        },
    ),
)