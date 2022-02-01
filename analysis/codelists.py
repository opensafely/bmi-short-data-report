from cohortextractor import (
    codelist,
    codelist_from_csv,
)

# ---------------------------
# BMI and height/weight codes
# ---------------------------

# CTV3 
bmi_code_ctv3 = codelist(
    ["22K.."],
    system="ctv3",
)
weight_codes_ctv3 = codelist(
    [
        "X76C7",  # Concept containing "body weight" terms:
        "22A..",  # O/E weight
    ],
    system="ctv3",
)
height_codes_ctv3 = codelist(
    [
        "XM01E",  # Concept containing height/length/stature/growth terms:
        "229..",  # O/E height
    ],
    system="ctv3",
)
# SNOMED 
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
# CTV3 
ethnicity_codes = codelist_from_csv(
        "codelists/opensafely-ethnicity.csv",
        system="ctv3",
        column="Code",
        category_column="Grouping_6",
    )
# SNOMED
# Main
eth2001 = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-eth2001.csv",
    system="snomed",
    column="code",
    category_column="grouping_16_id",
)
# Any other ethnicity code
non_eth2001 = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-non_eth2001.csv",
    system="snomed",
    column="code",
)
# Ethnicity not given - patient refused
eth_notgiptref = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-eth_notgiptref.csv",
    system="snomed",
    column="code",
)
# Ethnicity not stated
eth_notstated = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-eth_notstated.csv",
    system="snomed",
    column="code",
)
# Ethnicity no record
eth_norecord = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-eth_norecord.csv",
    system="snomed",
    column="code",
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
