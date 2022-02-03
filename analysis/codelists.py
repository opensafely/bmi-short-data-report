from cohortextractor import (
    codelist_from_csv,
)

# ---------------------------
# BMI and height/weight codes
# ---------------------------

bmi_code_snomed = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-bmi.csv",
    system="snomed",
    column='code'
)
height_codes_snomed = codelist_from_csv(
    "codelists/opensafely-height-snomed.csv",
    system="snomed",
    column='code'
)
weight_codes_snomed = codelist_from_csv(
    "codelists/opensafely-weight-snomed.csv",
    system="snomed",
    column='code'
)

# ----------------
# Ethnicity codes
# ----------------
ethnicity_snomed = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-eth2001.csv",
    system="snomed",
    column="code",
    category_column="grouping_16_id",
)

# --------------------
# Clinical conditions
# --------------------
# Learning disability
wider_ld_codes = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-learndis.csv",
    system="snomed",
    column="code"
)
