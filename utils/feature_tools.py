from functools import reduce
import re

from boruta import BorutaPy
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


def correlation_threshold(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Filter Correlation Matrix by Correlation Threshold

    :param corr_matrix: a dataframe that is a correlation matrix
    :param threshold: the correlation threshold (a float)
    :return: dataframe of unique pairs that meet the correlation threshold
    """
    threshold = threshold

    filtered_pairs = corr_matrix.stack()
    filtered_pairs = filtered_pairs[filtered_pairs > threshold]

    # Remove self-correlations and duplicate pairs
    filtered_pairs = filtered_pairs.reset_index()
    filtered_pairs.columns = ['Column 1', 'Column 2', 'Correlation']

    filtered_pairs = filtered_pairs[filtered_pairs['Column 1'] != filtered_pairs['Column 2']]

    filtered_pairs['min_col'] = filtered_pairs[['Column 1', 'Column 2']].min(axis=1)
    filtered_pairs['max_col'] = filtered_pairs[['Column 1', 'Column 2']].max(axis=1)
    unique_pairs = filtered_pairs[['min_col', 'max_col', 'Correlation']].drop_duplicates()

    unique_pairs.columns = ['Column 1', 'Column 2', 'Correlation']

    return unique_pairs


def objects_to_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any object types to categoricals.
    :param df: DataFrame to convert
    :return: DataFrame with object columns converted to categorical dtype
    """
    object_columns = list(df.select_dtypes(include="object"))
    df[object_columns] = df[object_columns].apply(lambda x: x.astype('category'))
    return df


def one_hot_categoricals(df: pd.DataFrame, drop: str = 'first') -> pd.DataFrame:
    """
    One hot encode any  categorical columns "category" type
    :param drop: whether to drop the first column in the one-hot encoded column
    :param df: DataFrame to one hot encode
    :return: DataFrame with one hot encoded categorical columns
    """
    encoder = OneHotEncoder(sparse_output=False, drop=drop)
    encoded = encoder.fit_transform(df.select_dtypes(include='category'))
    df_one_hot = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    return pd.concat([df.drop(columns=df.select_dtypes(include='category').columns), df_one_hot], axis=1)


def boruta_feature_selection(df: pd.DataFrame, target: pd.Series, drop=None, class_weight: str = 'balanced',
                             max_depth: int = 5, max_iter: int = 40):
    """
    Wrapper for BorutaPy feature selection
    :param df:
    :param target:
    :param drop: None or 'first' to drop first column for linear modeling when performing one hot encoding
    :param class_weight:
    :param max_depth:
    :return: New Dataframe containing features selected from df by BorutaPy and array of column rankings from original df
    """
    df_boruta = objects_to_categoricals(df.copy())
    df_boruta = one_hot_categoricals(df_boruta, drop=drop)
    X = df_boruta.values
    y = target
    rf = RandomForestClassifier(n_jobs=-1, class_weight=class_weight, max_depth=max_depth)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, max_iter=max_iter)
    feat_selector.fit(X, y)
    return df_boruta.loc[:, feat_selector.support_], feat_selector.ranking_


def drop_least_target_correlated_features(corr_mat: pd.DataFrame, target_series: pd.Series, df2: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Compare the correlation between two variables and the target to decide which variables are less informative and
    drop those columns

    Parameters:
    - corr_mat (pd.DataFrame): A dataframe correlation matrix with columns "Column 1", "Column 2",
                               and "Correlation" with threshold applied to it.
    - target_series (pd.Series): A series where the values used in "Column 1" and "Column 2" of df1 are indices for the
                                 correlation with the target.
    - df2 (pd.DataFrame): A dataframe where the column headers correspond to values from "Column 1" and "Column 2".

    Returns:
    - pd.DataFrame: The modified df2 with the appropriate columns dropped.
    """
    corr_mat = corr_mat.sort_values(by=['Correlation'], ascending=False)
    ser1_df = target_series.rename_axis('col').reset_index(name='ser1_value')

    df1_with_values = corr_mat \
        .merge(ser1_df, left_on="Column 1", right_on="col") \
        .rename(columns={"ser1_value": "val1"}) \
        .drop(columns="col") \
        .merge(ser1_df, left_on="Column 2", right_on="col") \
        .rename(columns={"ser1_value": "val2"}) \
        .drop(columns="col")

    df1_with_values['drop_col'] = df1_with_values.apply(
        lambda row: row['Column 1'] if row['val1'] < row['val2'] else row['Column 2'],
        axis=1
    )

    columns_to_drop = df1_with_values['drop_col'].unique()

    return df2.drop(columns=columns_to_drop, errors='ignore')


def summary_stat_df(df: pd.DataFrame, by_col: str, agg_ct_name: str) -> pd.DataFrame:
    """
    Create a new DataFrame with counts and aggregations created from grouping by a particular column and then generating
    summary statistics of numerical variables and frequency encoding object-type variables.
    note: NaN values are converted to 0

    :param df: DataFrame to summarize containing numerical and object type variables
    :param by_col: column name to group by
    :param agg_ct_name: name to give to the row of counts generated by grouping by by by_col
    :return: new DataFrame containing counts and summary stats.
    """
    stats_df = pd.DataFrame()
    count_dfs = []

    numerical_cols = df.select_dtypes(include='number').columns.difference([by_col])
    object_cols = df.select_dtypes(include='object').columns

    stats_df[agg_ct_name] = df.groupby(by_col).size()

    numerical_stats = df.groupby(by_col)[numerical_cols].agg(
        ['min', 'max', 'mean', 'median', 'std']
    )
    numerical_stats.columns = ['_'.join(col).strip() for col in numerical_stats.columns.values]
    numerical_stats = numerical_stats.reset_index()

    for col in object_cols:
        count_df = df.groupby(by_col)[col].value_counts().unstack(fill_value=0)
        count_df = count_df.add_prefix(f'{col}_').add_suffix('_cts')
        count_df = count_df.reset_index()
        count_dfs.append(count_df)

    if count_dfs:
        object_counts = reduce(lambda left, right: pd.merge(left, right, on=by_col, how='outer'), count_dfs)
        stats_df = pd.merge(stats_df, numerical_stats, on=by_col, how='outer')
        stats_df = pd.merge(stats_df, object_counts, on=by_col, how='outer')
    else:
        stats_df = pd.merge(stats_df, numerical_stats, on=by_col, how='outer')

    return stats_df.fillna(0)


def clean_column_names(df):
    df.columns = [re.sub(r'[^\w]', '_', col) for col in df.columns]
    return df


def to_downcasted_numeric(df, cols):
    df_converted = df.copy()
    df_converted[cols] = df_converted[cols].apply(pd.to_numeric, errors='coerce')
    df_converted[cols] = df_converted[cols].apply(lambda col: pd.to_numeric(col, downcast='integer')
                                     if pd.api.types.is_integer_dtype(col)
                                     else pd.to_numeric(col, downcast='float'))
    return df_converted


def get_columns_by_cardinality(df: pd.DataFrame, datatype='object', threshold: int = 3):
    categorical_cols = df.select_dtypes(include=datatype).columns

    more_than_threshold = []
    less_or_equal_threshold = []

    for col in categorical_cols:
        num_categories = df[col].nunique()
        if num_categories > threshold:
            more_than_threshold.append(col)
        else:
            less_or_equal_threshold.append(col)

    return more_than_threshold, less_or_equal_threshold


def shapiro_test(df, feats, sample_size=5000):
    df_sample = df.iloc[:sample_size].dropna()

    results = {}
    for feat in feats:
        statistic, p_value = stats.shapiro(df_sample[feat])
        results[feat] = (statistic, p_value)
        print(
            f"Shapiro-Wilk test for '{feat}':\n"
            f"  Statistic = {statistic:.3f}, p-value = {p_value:.3f}\n"
        )

    return results


def interval_feats():
    return [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_CREDIT_div_AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT_div_AMT_INCOME_TOTAL",
        "AMT_ANNUITY_div_AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE",
        "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE",
        "COMMONAREA_MODE",
        "ELEVATORS_MODE",
        "ENTRANCES_MODE",
        "FLOORSMAX_MODE",
        "FLOORSMIN_MODE",
        "LANDAREA_MODE",
        "LIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI",
        "ELEVATORS_MEDI",
        "ENTRANCES_MEDI",
        "FLOORSMAX_MEDI",
        "FLOORSMIN_MEDI",
        "LANDAREA_MEDI",
        "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI",
        "NONLIVINGAPARTMENTS_MEDI",
        "NONLIVINGAREA_MEDI",
        "TOTALAREA_MODE",
        "AMT_ANNUITY_max_x",
        "AMT_CREDIT_MAX_OVERDUE_max",
        "AMT_CREDIT_MAX_OVERDUE_std",
        "AMT_CREDIT_SUM_median",
        "AMT_CREDIT_SUM_DEBT_max",
        "AMT_CREDIT_SUM_DEBT_std",
        "AMT_CREDIT_SUM_LIMIT_max",
        "AMT_CREDIT_SUM_LIMIT_mean",
        "AMT_CREDIT_SUM_OVERDUE_max",
        "CREDIT_DAY_OVERDUE_std",
        "DAYS_CREDIT_min",
        "DAYS_CREDIT_max",
        "DAYS_CREDIT_median",
        "DAYS_CREDIT_std",
        "DAYS_CREDIT_ENDDATE_min",
        "DAYS_CREDIT_ENDDATE_max",
        "DAYS_CREDIT_ENDDATE_mean",
        "DAYS_CREDIT_UPDATE_max",
        "DAYS_CREDIT_UPDATE_mean",
        "DAYS_ENDDATE_FACT_max",
        "MONTHS_BALANCE_mean_mean",
        "STATUS_1_cts_max",
        "STATUS_1_cts_mean",
        "STATUS_1_cts_std",
        "STATUS_2_cts_mean",
        "AMT_CREDIT_LIMIT_ACTUAL_median",
        "AMT_CREDIT_LIMIT_ACTUAL_std",
        "AMT_DRAWINGS_ATM_CURRENT_max",
        "AMT_DRAWINGS_ATM_CURRENT_mean",
        "AMT_DRAWINGS_CURRENT_std",
        "AMT_DRAWINGS_POS_CURRENT_mean",
        "AMT_DRAWINGS_POS_CURRENT_std",
        "AMT_PAYMENT_TOTAL_CURRENT_max",
        "AMT_RECEIVABLE_PRINCIPAL_max",
        "AMT_RECIVABLE_min",
        "AMT_RECIVABLE_median",
        "CNT_DRAWINGS_ATM_CURRENT_mean",
        "CNT_DRAWINGS_ATM_CURRENT_std",
        "CNT_DRAWINGS_CURRENT_max",
        "CNT_DRAWINGS_CURRENT_mean",
        "CNT_INSTALMENT_MATURE_CUM_min",
        "CNT_INSTALMENT_MATURE_CUM_std",
        "MONTHS_BALANCE_max_x",
        "MONTHS_BALANCE_mean",
        "CNT_INSTALMENT_min",
        "CNT_INSTALMENT_FUTURE_median",
        "CNT_INSTALMENT_FUTURE_std",
        "MONTHS_BALANCE_min",
        "MONTHS_BALANCE_max_y",
        "MONTHS_BALANCE_std",
        "SK_DPD_std",
        "SK_DPD_DEF_max",
        "AMT_ANNUITY_max_y",
        "AMT_ANNUITY_mean",
        "AMT_CREDIT_median",
        "AMT_CREDIT_std",
        "AMT_DOWN_PAYMENT_max",
        "CNT_PAYMENT_max",
        "CNT_PAYMENT_mean",
        "CNT_PAYMENT_std",
        "DAYS_DECISION_min",
        "DAYS_DECISION_mean",
        "DAYS_DECISION_std",
        "DAYS_FIRST_DRAWING_std",
        "DAYS_FIRST_DUE_min",
        "DAYS_FIRST_DUE_std",
        "DAYS_LAST_DUE_1ST_VERSION_median",
        "DAYS_TERMINATION_mean",
        "HOUR_APPR_PROCESS_START_min",
        "RATE_DOWN_PAYMENT_max",
        "RATE_DOWN_PAYMENT_std",
        "SELLERPLACE_AREA_max",
        "AMT_PAYMENT_min",
        "AMT_PAYMENT_max",
        "AMT_PAYMENT_std",
        "DAYS_ENTRY_PAYMENT_min",
        "DAYS_ENTRY_PAYMENT_std",
        "DAYS_INSTALMENT_max",
        "NUM_INSTALMENT_NUMBER_std",
        "NUM_INSTALMENT_VERSION_min",
        "NUM_INSTALMENT_VERSION_max",
        "NUM_INSTALMENT_VERSION_mean",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "DAYS_REGISTRATION",
    ]
