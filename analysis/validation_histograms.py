import numpy as np
import pandas as pd
from sys import stderr
from matplotlib import pyplot as plt
from lib_phenotype_validation import import_clean

########################## CONFIGURE OPTIONS HERE #############################

# Import file
input_path = "output/data/input_processed.feather"

# Definitions
definitions = ["backend_bmi", "computed_bmi", "derived_bmi", "recorded_bmi"]

# Code dictionary
code_dict = {
    "ethnicity": {
        1: "White",
        2: "Mixed",
        3: "Asian",
        4: "Black",
        5: "Other",
        np.nan: "Unknown",
        0: "Unknown",
    },
    "imd": {
        0: "Unknown",
        1: "1 Most deprived",
        2: "2",
        3: "3",
        4: "4",
        5: "5 Least deprived",
    },
}

# Other variables to include
other_vars = [
    "height_backend",
    "weight_backend",
    "height_backend_date",
    "weight_backend_date",
]

# Dates
dates = True
date_min = "2015-03-01"
date_max = "2022-03-01"
time_delta = "M"

# Min/max range
height_min = 0.5
height_max = 2.8

weight_min = 3
weight_max = 500

bmi_min = 4
bmi_max = 200

# Null value – could be multiple values in a list [0,'0',NA]
null = ["0", 0, np.nan]

# Covariates
demographic_covariates = ["age_band", "sex", "ethnicity", "region", "imd"]
clinical_covariates = [
    "dementia",
    "diabetes",
    "hypertension",
    "learning_disability",
]

# Output path
output_path = "histograms"

#####################################################################################


def redact_round_table2(df_in):
    """Redacts counts <= 5 and rounds counts to nearest 5"""
    df_out = df_in.where(df_in > 5, np.nan).apply(
        lambda x: 5 * round(x / 5) if ~np.isnan(x) else x
    )
    return df_out


def hist(df_in, measure, title, path, n_bins=30):
    try:
        df_measure = df_in[measure]
        nan_count = df_measure.isna().sum()
        counts, bins = np.histogram(df_measure[df_measure.notna()], bins=n_bins)
        formatted_bins = [f"{b} - {bins[i+1]}" for i, b in enumerate(bins[:-1])]
        df_hist = pd.DataFrame({"bins": formatted_bins, "counts": counts})
        df_hist["counts"] = redact_round_table2(df_hist["counts"])
        plt.hist(bins[:-1], bins, weights=counts)
        plt.title(title)
        plt.savefig(
            f"output/{output_path}/figures/hist_{path}.png", bbox_inches="tight",
        )
        plt.close()
        df_hist = pd.concat(
            [df_hist, pd.DataFrame({"bins": "NaN", "counts": nan_count}, index=[0])]
        ).reset_index()
        df_hist.to_csv(f"output/{output_path}/tables/hist_data_{path}.csv")
    except Exception as e:
        print(
            f"Error plotting histogram for measure:{measure}, path:{path}. {e}",
            file=stderr,
        )
        raise


def decile_plot(df_in, measure, title, path):
    try:
        df_dec = (
            pd.DataFrame(
                pd.qcut(df_in[measure], 10, duplicates="drop")
                .value_counts()
                .sort_index()
            )
            .reset_index()
            .rename(columns={"index": "intervals"})
        )
        df_dec[measure] = (
            df_dec[measure]
            .where(df_dec[measure] > 5, np.nan)
            .apply(lambda x: 5 * round(x / 5) if ~np.isnan(x) else x)
        )
        df_in["bin"] = pd.qcut(df_in[measure], 10, duplicates="drop").astype(str)
        df_in2 = redact_round_table2(df_in.groupby("bin").bin.count())
        df_in2.plot(kind="bar")
        plt.title(title)
        df_dec.to_csv(f"output/{output_path}/tables/decile_plot_{path}.csv")
        plt.savefig(
            f"output/{output_path}/figures/decile_plot_{path}.png", bbox_inches="tight",
        )
        plt.close()
    except:
        pass


def q_n(x, pct):
    """
    Returns the value at the given quantile
    """
    return x.quantile(pct)


def subset_q(df_in, measure, threshold, less=True):
    """
    Subsets the data based on numeric threshold
    """
    if less == True:
        df_subset = df_in.loc[df_in[measure] < threshold]
    else:
        df_subset = df_in.loc[df_in[measure] > threshold]
    return df_subset


def count_table(df_in, measure, path):
    """
    Counts and outputs the number of non-NA rows
    """
    ct_table = pd.DataFrame(df_in[[measure]].count(), columns=["counts"])
    ct_table.to_csv(f"output/{output_path}/tables/ct_{path}.csv")


def cdf(df_in, measure, path):
    """
    Computes and plots the cumulative distribution function (CDF)
    """
    # Frequency
    df_stats = df_in[[measure]]
    df_freq = (
        df_stats.groupby(measure)[measure]
        .agg("count")
        .pipe(pd.DataFrame)
        .rename(columns={measure: "frequency"})
    )
    # PDF
    df_freq["pdf"] = df_freq["frequency"] / sum(df_freq["frequency"])
    # CDF
    df_freq["cdf"] = df_freq["pdf"].cumsum()
    df_freq = df_freq.reset_index()
    df_freq.plot(x=measure, y="cdf", grid=True)
    plt.title(f"CDF of {measure}")
    plt.savefig(f"output/{output_path}/figures/cdf_{path}.png", bbox_inches="tight")
    plt.close()


###################### SPECIFY ANALYSES TO RUN HERE ########################


def main():
    df_clean = import_clean(
        input_path,
        definitions,
        other_vars,
        demographic_covariates,
        clinical_covariates,
        null,
        date_min,
        date_max,
        time_delta,
        output_path,
        code_dict,
        dates,
    )
    # Get rid of 0s in weight/height
    for v in ["height_backend", "weight_backend"]:
        # Set null values to nan
        df_clean.loc[df_clean[v].isin(null), v] = np.nan
    # Count negative values
    df_height_neg = df_clean.loc[df_clean["height_backend"] < 0]
    count_table(df_height_neg, "height_backend", "neg_height")
    df_weight_neg = df_clean.loc[df_clean["weight_backend"] < 0]
    count_table(df_weight_neg, "weight_backend", "neg_weight")
    # Count high, unreasonable values
    df_height_gt = subset_q(df_clean, "height_backend", 250, less=False)
    count_table(df_height_gt, "height_backend", "high_height")
    df_weight_gt = subset_q(df_clean, "weight_backend", 500, less=False)
    count_table(df_weight_gt, "weight_backend", "high_weight")
    # Create datasets for reasonable ranges
    # Height (0-3; meters)
    df_height_m = df_clean.loc[
        (df_clean["height_backend"] > 0) & (df_clean["height_backend"] < 3)
    ]
    # Height (10-250; cm)
    df_height_cm = df_clean.loc[
        (df_clean["height_backend"] > 10) & (df_clean["height_backend"] < 250)
    ]
    # Weight (0-500; should cover most kg and lbs)
    df_weight_bound = df_clean.loc[
        (df_clean["weight_backend"] > 0) & (df_clean["weight_backend"] < 500)
    ]
    ### Create histograms
    # Reasonable height (considering cm/in measurements)
    hist(
        df_height_m,
        "height_backend",
        "Distribution of Height Between 0 and 3 (meters)",
        "height_meter_range",
    )
    hist(
        df_height_cm,
        "height_backend",
        "Distribution of Height Between 10 and 250 (cm)",
        "height_cm_range",
    )
    # Reasonable weight (considering stone/lbs)
    hist(
        df_weight_bound,
        "weight_backend",
        "Distribution of Weight Between 0 and 500",
        "weight_bound",
    )
    ### Create CDFs
    # Reasonable height (considering cm/in measurements)
    cdf(df_height_m, "height_backend", "height_meter_range")
    cdf(df_height_cm, "height_backend", "height_cm_range")
    # Reasonable weight (considering stone/lbs)
    cdf(df_weight_bound, "weight_backend", "weight_bound")


########################## DO NOT EDIT – RUNS SCRIPT ##############################

if __name__ == "__main__":
    main()
