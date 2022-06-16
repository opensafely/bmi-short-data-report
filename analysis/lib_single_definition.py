import os
import numpy as np
import pandas as pd


def redact(df_in, var):
    df_in.loc[df_in[var] > 5, var] = 5 * round(df_in[var]/5)
    df_in.loc[(df_in[var].isna()) | (df_in[var] < 6), var] = '-'


def import_clean(input_path, definition, time_delta,
                 demographic_covariates, clinical_covariates,
                 other_variables = []):
    df_input = pd.read_feather(
        input_path,
        columns = ['patient_id'] + [definition] + [definition+'_date'] + 
                   demographic_covariates + clinical_covariates + other_variables
    )
    # Drop null values
    df_input = df_input.loc[df_input[definition] != 0].reset_index(drop=True)
    # Create order for categorical variables
    for group in demographic_covariates + clinical_covariates:
        if df_input[group].dtype.name == 'category':
            li_order = sorted(df_input[group].dropna().unique().tolist())
            df_input[group] = pd.Categorical(df_input[group], categories=li_order)
    # Dates to time delta
    df_input[f'{definition}_date'] = df_input[f'{definition}_date'].dt.to_period(
        time_delta).dt.to_timestamp()
    # Check whether output paths exist or not, create if missing
    filepath = f'output/validation/tables/{definition}'
    exists = os.path.exists(filepath)
    if not exists:
        os.makedirs(filepath)
    return df_input


def patient_counts(df_input, definition, 
                   demographic_covariates, clinical_covariates):
    # Overall
    df_counts = pd.DataFrame(
        df_input.agg({"patient_id": pd.Series.nunique}
        ), columns=[definition]).T
    df_counts = df_counts.rename(
        columns={'patient_id':'num_patients'})
    df_out = df_counts.where(
        df_counts > 5, np.nan).apply(lambda x: 5 * round(x / 5)).fillna('-')
    df_out.to_csv(f'output/validation/tables/{definition}/{definition}_patient_counts.csv')

    # By group
    for group in demographic_covariates + clinical_covariates:
        df_group = df_input.groupby(group).agg({"patient_id": pd.Series.nunique})
        df_group = df_group.rename(columns={'patient_id':'num_patients'})
        df_out_group = df_group.where(
            df_group > 5, np.nan
        ).apply(lambda x: 5 * round(x / 5)).fillna('-')
        df_out_group.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_patient_counts.csv')


def measurement_counts(df_input, definition, 
                       demographic_covariates, clinical_covariates):
    # Overall
    df_counts = pd.DataFrame(
        df_input.agg({definition: "count"}
        ), columns=[definition]).T
    df_counts = df_counts.rename(
        columns={definition:'num_measurements'})
    df_out = df_counts.where(
        df_counts > 5, np.nan).apply(lambda x: 5 * round(x / 5)).fillna('-')
    df_out.to_csv(f'output/validation/tables/{definition}/{definition}_measurement_counts.csv')

    # By group
    for group in demographic_covariates + clinical_covariates:
        df_group = df_input.groupby(group).agg({definition: "count"})
        df_group = df_group.rename(columns={definition:'num_measurements'})
        df_out_group = df_group.where(
            df_group > 5, np.nan
        ).apply(lambda x: 5 * round(x / 5)).fillna('-')
        df_out_group.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_measurement_counts.csv')


def distribution(df_input, definition, 
                 demographic_covariates, clinical_covariates):
    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    # Overall
    df_distribution = df_input.agg({definition: ['median',q25,q75]}).T
    df_distribution['iqr'] = df_distribution['q75'] - df_distribution['q25']
    df_distribution['lower_extreme'] = df_distribution['q25']-(1.5*df_distribution['iqr'])
    df_distribution['upper_extreme'] = df_distribution['q75']+(1.5*df_distribution['iqr'])
    df_distribution = df_distribution[['lower_extreme','q25','median','q75','upper_extreme']]
    df_distribution.to_csv(f'output/validation/tables/{definition}/{definition}_distribution.csv')

    # By group
    for group in demographic_covariates + clinical_covariates:
        df_group_dist=df_input.groupby(group).agg({definition: ['median',q25,q75]}).droplevel(0,1)
        df_group_dist['iqr'] = df_group_dist['q75'] - df_group_dist['q25']
        df_group_dist['lower_extreme'] = df_group_dist['q25']-(1.5*df_group_dist['iqr'])
        df_group_dist['upper_extreme'] = df_group_dist['q75']+(1.5*df_group_dist['iqr'])
        df_group_dist = df_group_dist[['lower_extreme','q25','median','q75','upper_extreme']]
        df_group_dist.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_distribution.csv')


def cdf(df_input, definition, out_folder):
    # Compute frequency
    df_stats = df_input[[definition]]
    df_freq = (
        df_stats.groupby(definition)[definition]
        .agg("count")
        .pipe(pd.DataFrame)
        .rename(columns={definition: "frequency"})
    )
    # Compute PDF
    df_freq["pdf"] = df_freq["frequency"] / sum(df_freq["frequency"])
    # Compute CDF
    df_freq["cdf"] = df_freq["pdf"].cumsum()
    df_freq = df_freq.reset_index()
    df_freq.to_csv(f'output/validation/tables/{out_folder}/{definition}_cdf_data.csv')


def less_than_min(df_input, definition, min_value, 
                  demographic_covariates, clinical_covariates):
    # Overall
    df_lt_min = df_input.loc[df_input[definition] < min_value]
    df_out = pd.DataFrame(
        df_lt_min[definition].agg(['count','mean'])
    )
    if df_out.loc['count'][definition] > 5:
        df_out.loc['count'][definition] = 5 * round(df_out.loc['count'][definition]/5)
    else:
        df_out[definition] = '-'
    df_out.to_csv(f'output/validation/tables/{definition}/{definition}_less_than_min.csv')

    # By group
    for group in demographic_covariates + clinical_covariates:
        df_group = df_lt_min.groupby(group)[definition].agg(
            [('count', 'count'),('mean', 'mean')]
        )
        df_group.loc[df_group['count'] > 5, 'count'] = 5 * round(df_group['count']/5)
        df_group.loc[(df_group['count'].isna()) | (df_group['count'] < 6), ['count', 'mean']] = ['-','-']
        df_group.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_less_than_min.csv')


def greater_than_max(df_input, definition, max_value, 
                     demographic_covariates, clinical_covariates):
    # Overall
    df_gt_max = df_input.loc[df_input[definition] > max_value]
    df_out = pd.DataFrame(
        df_gt_max[definition].agg(['count','mean'])
    )
    if df_out.loc['count'][definition] > 5:
        df_out.loc['count'][definition] = 5 * round(df_out.loc['count'][definition]/5)
    else:
        df_out[definition] = '-'
    df_out.to_csv(f'output/validation/tables/{definition}/{definition}_greater_than_max.csv')

    # By group
    for group in demographic_covariates + clinical_covariates:
        df_group = df_gt_max.groupby(group)[definition].agg(
            [('count', 'count'),('mean', 'mean')]
        )
        df_group.loc[df_group['count'] > 5, 'count'] = 5 * round(df_group['count']/5)
        df_group.loc[(df_group['count'].isna()) | (df_group['count'] < 6), ['count', 'mean']] = ['-','-']
        df_out.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_greater_than_max.csv')


def records_over_time(df_input, definition, 
                      demographic_covariates, clinical_covariates):
    # Overall
    df_records = df_input[[f'{definition}_date',definition]].groupby(
        f'{definition}_date'
    ).count().reset_index().rename(columns={f'{definition}_date':'date'})
    redact(df_records, definition)
    df_records.to_csv(f'output/validation/tables/{definition}/{definition}_records_over_time.csv')

    for group in demographic_covariates + clinical_covariates:
        df_group = df_input[
            [f'{definition}_date', definition, group]
        ].groupby(
            [f'{definition}_date', group]
        ).count().reset_index().rename(columns={f'{definition}_date':'date'})
        redact(df_group, definition)
        df_group.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_records_over_time.csv')

        
def means_over_time(df_input, definition, 
                    demographic_covariates, clinical_covariates):
    # Overall
    df_means = df_input.groupby(f'{definition}_date')[definition].agg(
        ['count','mean']
    ).reset_index().rename(columns={f'{definition}_date':'date'})
    # Redact and round values
    redact(df_means, 'count')
    df_means.loc[df_means['count'] == '-', 'mean'] = '-'
    df_means.to_csv(f'output/validation/tables/{definition}/{definition}_means_over_time.csv')

    for group in demographic_covariates + clinical_covariates:
        df_group = df_input.groupby([f'{definition}_date',group])[definition].agg(
            ['count','mean']
        ).reset_index().rename(columns={f'{definition}_date':'date'})
        # Redact and round values
        redact(df_group, 'count')
        df_group.loc[df_group['count'] == '-', 'mean'] = '-'
        df_group.to_csv(f'output/validation/tables/{definition}/{definition}_{group}_means_over_time.csv')


def recent_to_now(df_input, definition):
    curr_time = pd.to_datetime("now")
    df_temp = df_input[['patient_id', definition+'_date']].sort_values(by=['patient_id', definition+'_date'], ascending=False)
    df_temp2 = df_temp.drop_duplicates(subset='patient_id')
    # Compute difference between dates (in days)
    df_temp2[definition+'_date_diff'] = (curr_time-df_temp2[definition+'_date']).dt.days
    cdf(df_temp2, definition+'_date_diff', definition)


def count_table(df_input, definition, out_folder):
    ct_table = pd.DataFrame(df_input[[definition]].count(), columns=["counts"])
    ct_table.to_csv(f"output/validation/tables/{out_folder}/ct_{definition}.csv")