import pandas as pd

for bmi in ['backend_computed_bmi','computed_bmi','derived_bmi','recorded_bmi']:
    df_bmi = pd.read_feather(f'output/joined/input_processed_{bmi}.feather')

    # Drop measurements with invalid dates 
    df_bmi = df_bmi.loc[df_bmi[f'{bmi}_date'] > '1900-01-01'].reset_index(drop=True)

    # Create age
    df_bmi['date_of_birth'] = pd.to_datetime(df_bmi['date_of_birth'])
    df_bmi[f'{bmi}_date'] = pd.to_datetime(df_bmi[f'{bmi}_date'])
    df_bmi['age'] = df_bmi[f'{bmi}_date'].dt.year - df_bmi['date_of_birth'].dt.year - (
        (df_bmi[f'{bmi}_date'].dt.month) < 
        (df_bmi['date_of_birth'].dt.month)
    )

    # Create age_band
    df_bmi['age_band'] = 'missing'
    df_bmi.loc[(df_bmi['age'] >=0) & (df_bmi['age'] < 20), 'age_band'] = '0-19'
    df_bmi.loc[(df_bmi['age'] >=20) & (df_bmi['age'] < 30), 'age_band'] = '20-29'
    df_bmi.loc[(df_bmi['age'] >=30) & (df_bmi['age'] < 40), 'age_band'] = '30-39'
    df_bmi.loc[(df_bmi['age'] >=40) & (df_bmi['age'] < 50), 'age_band'] = '40-49'
    df_bmi.loc[(df_bmi['age'] >=50) & (df_bmi['age'] < 60), 'age_band'] = '50-59'
    df_bmi.loc[(df_bmi['age'] >=60) & (df_bmi['age'] < 70), 'age_band'] = '60-69'
    df_bmi.loc[(df_bmi['age'] >=70) & (df_bmi['age'] < 80), 'age_band'] = '70-79'
    df_bmi.loc[(df_bmi['age'] >=80) & (df_bmi['age'] < 120), 'age_band'] = '80+'

    df_bmi.to_feather(f'output/joined/input_processed_{bmi}.feather')