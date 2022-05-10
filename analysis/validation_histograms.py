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
        counts, bins = np.histogram(
            df_measure[df_measure.notna()],
            bins=n_bins)
        formatted_bins = [
            f"{b} - {bins[i+1]}" for i, b in enumerate(bins[:-1])
        ]
        df_hist = pd.DataFrame({"bins": formatted_bins, "counts": counts})
        df_hist['counts'] = redact_round_table2(df_hist['counts'])
        plt.hist(bins[:-1], bins, weights=counts)
        plt.title(title)
        plt.savefig(
            f"output/{output_path}/figures/hist_{path}.png",
            bbox_inches="tight",
        )
        plt.close()
        df_hist = pd.concat(
            [df_hist,
             pd.DataFrame({"bins": "NaN", "counts": nan_count}, index=[0])]).reset_index()
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
        df_in["bin"] = pd.qcut(df_in[measure], 10, duplicates="drop").astype(
            str
        )
        df_in2 = redact_round_table2(df_in.groupby("bin").bin.count())
        df_in2.plot(kind="bar")
        plt.title(title)
        df_dec.to_csv(f"output/{output_path}/tables/decile_plot_{path}.csv")
        plt.savefig(
            f"output/{output_path}/figures/decile_plot_{path}.png",
            bbox_inches="tight",
        )
        plt.close()
    except:
        pass


def q_n(x, pct):
    return x.quantile(pct)


def subset_q(df_in, measure, pct, less=True):
    # Drop the top 5 highest in measure (outliers)
    df_clean = df_in.loc[
        ~df_in[measure].isin(df_in[measure].nlargest(n=5).tolist())
    ]
    if pct < 1:
        p = q_n(df_clean[measure], pct)
    else:
        p = pct
    if less:
        df_p = df_clean.loc[df_clean[measure] < p]
    else:
        df_p = df_clean.loc[df_clean[measure] > p]
    return df_p


def kde_plot(df_in, measure, title, path):
    try:
        df_kde = pd.DataFrame(df_in[measure])
        df_kde.plot.kde()
        plt.title(title)
        plt.savefig(
            f"output/{output_path}/figures/kde_{path}.png", bbox_inches="tight"
        )
    except:
        pass


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
    ## Create histograms
    # All population
    hist(
        df_clean,
        "weight_backend",
        "Weight (CTV3 Codes Used in OpenSAFELY-TPP Backend)",
        "weight_all",
    )
    hist(
        df_clean,
        "height_backend",
        "Height (CTV3 Codes Used in OpenSAFELY-TPP Backend)",
        "height_all",
    )
    # Histogram of negative values
    df_height_neg = df_clean.loc[df_clean["height_backend"] < 0]
    hist(
        df_height_neg,
        "height_backend",
        "Distribution of Negative Heights",
        "height_negative",
    )
    df_weight_neg = df_clean.loc[df_clean["weight_backend"] < 0]
    hist(
        df_weight_neg,
        "weight_backend",
        "Distribution of Negative Weights",
        "weight_negative",
    )
    # Reasonable height (considering cm/in measurements)
    df_height_bound = df_clean.loc[
        (df_clean["height_backend"] > 0) & (df_clean["height_backend"] < 250)
    ]
    hist(
        df_height_bound,
        "height_backend",
        "Distribution of Height Between 0 and 250",
        "height_bound",
    )
    # Reasonable weight (considering stone/lbs)
    df_weight_bound = df_clean.loc[
        (df_clean["weight_backend"] > 0) & (df_clean["weight_backend"] < 500)
    ]
    hist(
        df_weight_bound,
        "weight_backend",
        "Distribution of Weight Between 0 and 500",
        "weight_bound",
    )
    # Above reasonable range
    df_height_gt = subset_q(df_clean, "height_backend", 250, less=False)
    hist(
        df_height_gt,
        "height_backend",
        "Distribution of Height Above 250",
        "height_gt_bound",
    )
    df_weight_gt = subset_q(df_clean, "weight_backend", 500, less=False)
    hist(
        df_weight_gt,
        "weight_backend",
        "Distribution of Weight Above 500",
        "weight_gt_bound",
    )
    ## Create decile plots
    # All population
    decile_plot(
        df_clean,
        "weight_backend",
        "Weight (CTV3 Codes Used in OpenSAFELY-TPP Backend)",
        "weight_all",
    )
    decile_plot(
        df_clean,
        "height_backend",
        "Height (CTV3 Codes Used in OpenSAFELY-TPP Backend)",
        "height_all",
    )
    # Negative values
    decile_plot(
        df_height_neg,
        "height_backend",
        "Distribution of Negative Heights",
        "height_negative",
    )
    decile_plot(
        df_weight_neg,
        "weight_backend",
        "Distribution of Negative Weights",
        "weight_negative",
    )
    # Reasonable height (considering cm/in measurements)
    decile_plot(
        df_height_bound,
        "height_backend",
        "Distribution of Height Between 0 and 250",
        "height_bound",
    )
    # Reasonable weight (considering stone/lbs)
    decile_plot(
        df_weight_bound,
        "weight_backend",
        "Distribution of Weight Between 0 and 500",
        "weight_bound",
    )
    # Above reasonable range
    decile_plot(
        df_height_gt,
        "height_backend",
        "Distribution of Height Above 250",
        "height_gt_bound",
    )
    decile_plot(
        df_weight_gt,
        "weight_backend",
        "Distribution of Weight Above 500",
        "weight_gt_bound",
    )
    ## Create KDE plots
    # All population
    kde_plot(
        df_clean,
        "height_backend",
        "Height (CTV3  Codes Used in OpenSAFELY-TPP Backend)",
        "height_all",
    )
    kde_plot(
        df_clean,
        "weight_backend",
        "Weight (CTV3  Codes Used in OpenSAFELY-TPP Backend)",
        "weight_all",
    )
    # Negative Values
    kde_plot(
        df_height_neg,
        "height_backend",
        "Distribution of Negative Heights",
        "height_negative",
    )
    kde_plot(
        df_weight_neg,
        "weight_backend",
        "Distribution of Negative Weights",
        "weight_negative",
    )
    # Reasonable height/weight
    kde_plot(
        df_height_bound,
        "height_backend",
        "Distribution of Height Between 0 and 250",
        "height_bound",
    )
    kde_plot(
        df_weight_bound,
        "weight_backend",
        "Distribution of Weight Between 0 and 500",
        "weight_bound",
    )
    # Above reasonable range
    kde_plot(
        df_height_gt,
        "height_backend",
        "Distribution of Height Above 250",
        "height_gt_bound",
    )
    kde_plot(
        df_weight_gt,
        "weight_backend",
        "Distribution of Weight Above 500",
        "weight_gt_bound",
    )


######################## DO NOT EDIT – RUNS SCRIPT ###########################

if __name__ == "__main__":
    main()
