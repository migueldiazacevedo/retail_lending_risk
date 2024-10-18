import gc
import warnings
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def can_downcast_float(series: pd.Series, target_dtype: np.dtype) -> bool:
    """Check if a float series can be safely downcasted to the target dtype."""
    try:
        # Skip downcasting for large values to avoid overflow
        if series.abs().max() > np.finfo(target_dtype).max:
            return False
        return np.allclose(series, series.astype(target_dtype), rtol=1e-05, atol=1e-08)
    except (FloatingPointError, ValueError, OverflowError):
        return False


def can_cast_to_int(series: pd.Series, target_dtype: np.dtype) -> bool:
    """Check if a series can be safely downcasted to the target integer dtype."""
    try:
        # Skip downcasting for large values to avoid overflow
        if series.min() < np.iinfo(target_dtype).min or series.max() > np.iinfo(target_dtype).max:
            return False
        return True
    except (FloatingPointError, ValueError, OverflowError):
        return False


def check_memory_usage(df: pd.DataFrame, message: str):
    """Prints the memory usage of the DataFrame."""
    mem_usage = df.memory_usage().sum() / 1024 ** 2
    print(f'Memory usage {message}: {mem_usage:.2f} MB')
    return mem_usage


def reduce_mem_usage(df: pd.DataFrame, int_cast: bool = True, obj_to_category: bool = False,
                     subset: List[str] = None) -> pd.DataFrame:
    """
    Reduce the memory usage of a Pandas DataFrame by converting column types to lower memory equivalents.
    """
    start_mem = check_memory_usage(df, 'before optimization')

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols, desc="Optimizing columns"):
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            max_val, min_val = df[col].max(), df[col].min()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if int_cast and pd.api.types.is_integer_dtype(df[col]):
                    if can_cast_to_int(df[col], np.int8):
                        df[col] = df[col].astype(np.int8)
                    elif can_cast_to_int(df[col], np.uint8):
                        df[col] = df[col].astype(np.uint8)
                    elif can_cast_to_int(df[col], np.int16):
                        df[col] = df[col].astype(np.int16)
                    elif can_cast_to_int(df[col], np.uint16):
                        df[col] = df[col].astype(np.uint16)
                    elif can_cast_to_int(df[col], np.int32):
                        df[col] = df[col].astype(np.int32)
                    elif can_cast_to_int(df[col], np.uint32):
                        df[col] = df[col].astype(np.uint32)
                elif pd.api.types.is_float_dtype(df[col]):
                    # Only downcast if values are not too large for float16 or float32
                    if max_val < np.finfo(np.float16).max and min_val > np.finfo(np.float16).min:
                        df[col] = df[col].astype(np.float16)
                    elif max_val < np.finfo(np.float32).max and min_val > np.finfo(np.float32).min:
                        df[col] = df[col].astype(np.float32)

        elif obj_to_category and pd.api.types.is_object_dtype(col_type):
            df[col] = df[col].astype('category')

    gc.collect()
    end_mem = check_memory_usage(df, 'after optimization')
    print(f'Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%')
    return df


def groupby_repeat_pivot(df, by, instance_name='instance'):
    """
    Group dataframe by a repeated instance (e.g. an ID that appears more than once), count the number of instances
    :param df: Pandas DataFrame to group, pivot, and consider the nan values
    :param by: column to group by
    :param instance_name: name of the instance that you grouped by for new column in grouped_pivoted_df hat counts the
    instances
    :return: df, grouped_pivoted_df
    """
    df[f'{instance_name}_number'] = df.groupby(by).cumcount() + 1
    grouped_pivoted_df = df.pivot(index=by, columns=f'{instance_name}_number')
    grouped_pivoted_df.columns = [f'{col[0]}_{col[1]}' for col in grouped_pivoted_df.columns]
    grouped_pivoted_df = grouped_pivoted_df.reset_index()
    grouped_pivoted_df[f'n_{instance_name}s'] = df.groupby(by).size().values
    return df, grouped_pivoted_df


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
