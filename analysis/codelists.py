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
    category_column="grouping_6_id",
)

# --------------------
# Clinical conditions
# --------------------
# Chronic cardiac disease
chronic_cardiac_dis_codes = codelist_from_csv(
    "codelists/opensafely-chronic-cardiac-disease-snomed.csv",
    system="snomed",
    column="id"
)
# Chronic kidney disease
chronic_kidney_dis_codes = codelist_from_csv(
    "codelists/opensafely-chronic-kidney-disease-snomed.csv",
    system="snomed",
    column="id"
)
# Chronic liver disease
chronic_liver_dis_codes = codelist_from_csv(
    "codelists/opensafely-chronic-liver-disease-snomed.csv",
    system="snomed",
    column="id"
)
# Chronic respiratory disease
chronic_respiratory_dis_codes = codelist_from_csv(
    "codelists/opensafely-chronic-respiratory-disease-snomed.csv",
    system="snomed",
    column="id"
)
# Cancer (Haemotological)
cancer_haem_codes = codelist_from_csv(
    "codelists/opensafely-haematological-cancer-snomed.csv",
    system="snomed",
    column="id"
)
# Cancer (Lung)
cancer_lung_codes = codelist_from_csv(
    "codelists/opensafely-lung-cancer-snomed.csv",
    system="snomed",
    column="id"
)
# Cancer (Other)
cancer_other_codes = codelist_from_csv(
    "codelists/opensafely-cancer-excluding-lung-and-haematological-snomed.csv",
    system="snomed",
    column="id"
)
# Dementia
dementia_codes = codelist_from_csv(
    "codelists/opensafely-dementia-snomed.csv",
    system="snomed",
    column="id"
)
# Diabetes
diabetes_codes = codelist_from_csv(
    "codelists/opensafely-diabetes-snomed.csv",
    system="snomed",
    column="id"
)
# Housebound
housebound_codes = codelist_from_csv(
    "codelists/opensafely-housebound.csv",
    system="snomed",
    column="code"
)
# Hypertension
hypertension_codes = codelist_from_csv(
    "codelists/opensafely-hypertension-snomed.csv",
    system="snomed",
    column="id"
)
# Learning disability
wider_ld_codes = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-learndis.csv",
    system="snomed",
    column="code"
)
# Severe obesity
sev_obesity_codes = codelist_from_csv(
    "codelists/primis-covid19-vacc-uptake-sev_obesity.csv",
    system="snomed",
    column="code",
)
