from cohortextractor import (
    codelist,
    codelist_from_csv,
)

# ---------------------------
# BMI and height/weight codes
# ---------------------------

bmi_code_snomed = codelist(
    ["301331008"],    # Finding of body mass index (finding)
    system="snomed",
)  
weight_codes_snomed = codelist(
    [
        "27113001",   # Body weight (observable entity)
        "162763007",  # On examination - weight(finding)
    ],
    system="snomed",
)
height_codes_snomed = codelist(
    [
        "271603002",  # Height / growth measure (observable entity)
        "162755006",  # On examination - height (finding)
    ],
    system="snomed",
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
