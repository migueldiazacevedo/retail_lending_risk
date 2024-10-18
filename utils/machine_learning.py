import lightgbm as lgb
import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures, MathFeatures, RelativeFeatures
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogCpTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config

from utils.feature_tools import clean_column_names, to_downcasted_numeric


def model_assessment(y_true, y_pred, do_auc=True, do_acc=True, do_f1=True, do_precision=True, do_recall=True):
    results = {}

    if do_auc:
        results['auc'] = roc_auc_score(y_true, y_pred)
        print(f"AUC-ROC: {results['auc']}")

    if do_acc:
        results['accuracy'] = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {results['accuracy']}")

    if do_f1:
        results['f1'] = f1_score(y_true, y_pred)
        print(f"F1-Score: {results['f1']}")

    if do_precision:
        results['precision'] = precision_score(y_true, y_pred)
        print(f"Precision: {results['precision']}")

    if do_recall:
        results['recall'] = recall_score(y_true, y_pred)
        print(f"Recall: {results['recall']}")

    return results


def concatenate_model_scores(dict_list, col_names):
    """
    Concatenate multiple dictionaries into a single DataFrame.

    Args:
        dict_list (list of dict): A list of dictionaries to be converted to DataFrames.
        col_names (list of str): A list of column names corresponding to each dictionary.

    Returns:
        pd.DataFrame: A DataFrame with concatenated scores.
    """
    model_scores = pd.DataFrame.from_dict(dict_list[0], orient="index", columns=[col_names[0]])

    for i in range(1, len(dict_list)):
        df = pd.DataFrame.from_dict(dict_list[i], orient="index", columns=[col_names[i]])
        model_scores = pd.concat([model_scores, df], axis=1)

    return model_scores


def remove_verbose_col_name(col_name):
    parts = col_name.split("__")
    return parts[-1]


def project_feature_transformer():
    """
    instantiates feature transformer that is used for this project
    """
    application_flag_columns = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                                'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                                'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                                'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                                'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    cb_inqs = ['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    feature_creation = ColumnTransformer([
        ("CyclicalEncodeHrs", CyclicalFeatures(drop_original=True), ["HOUR_APPR_PROCESS_START"]),
        ("nAppFlagAllZeros",
         MathFeatures(variables=application_flag_columns, func="sum", new_variables_names=["nFLAGS_ALL_ZEROS"]),
         application_flag_columns),
        ("ANY_EXT_SOURCE_MISSING", AggregateMissingIndicator(columns=ext_sources, output_name="ANY_EXT_SOURCE_MISSING"),
         ext_sources),
        ("ANY_CREDIT_BUREAU_INQ_MISSING",
         AggregateMissingIndicator(columns=cb_inqs, output_name="ANY_CREDIT_BUREAU_INQ_MISSING"), cb_inqs),
        ("EXT_SOURCE_MEAN", MathFeatures(variables=ext_sources, func="mean", missing_values='ignore',
                                         new_variables_names=["EXT_SOURCE_MEAN"]), ext_sources),
        ("EXT_SOURCE_STD", MathFeatures(variables=ext_sources, func="std", missing_values='ignore',
                                        new_variables_names=["EXT_SOURCE_STD"]), ext_sources),
        ("ImputeDaysEmployed",
         ImputeBasedOnOtherColumn(column_to_impute='DAYS_EMPLOYED', column_for_condition='NAME_INCOME_TYPE',
                                  what_to_look_for=['Pensioner', 'Unemployed']), ["DAYS_EMPLOYED", "NAME_INCOME_TYPE"]),
        ("CREDIT_ANNUITY_RATIO",
         RelativeFeatures(variables=["AMT_CREDIT", "AMT_ANNUITY"], reference=["AMT_ANNUITY"], func=["div"],
                          missing_values='ignore'), ["AMT_CREDIT", "AMT_ANNUITY"]),
        ("CREDIT_BY_INCOME",
         RelativeFeatures(variables=["AMT_CREDIT", "AMT_INCOME_TOTAL"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                          missing_values='ignore'), ["AMT_CREDIT", "AMT_INCOME_TOTAL"]),
        ("ANNUITY_BY_INCOME",
         RelativeFeatures(variables=["AMT_ANNUITY", "AMT_INCOME_TOTAL"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                          missing_values='ignore'), ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]),
    ],
        remainder="passthrough")

    return feature_creation


def log_signed_transform(X):
    """Apply log transformation with absolute values while retaining sign."""
    return np.sign(X) * np.log(np.abs(X) + 1)


def rename_log_columns(df):
    """
    Rename columns containing 'log' to end with '_log'.
    """
    log_columns = df.columns[df.columns.str.contains("log")]
    df.rename(columns={col: f"{col}_log" for col in log_columns}, inplace=True)
    return df


# %%
class AggregateMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, columns, output_name="ANY_MISSING"):
        """
        Initialize the transformer with the columns to check for missing values.

        Parameters:
        columns: list of str
            The list of column names to check for missing values.
        """
        self.columns = columns
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        missing_indicator = X[self.columns].isnull().any(axis=1).astype(int)
        result = pd.concat(
            [X[self.columns], missing_indicator.rename(self.output_name)], axis=1
        )

        return result

    def get_feature_names_out(self, input_features=None):
        original_columns = self.columns if input_features is None else input_features
        return list(original_columns) + [self.output_name]


#%%
class ImputeBasedOnOtherColumn(BaseEstimator, TransformerMixin):
    def __init__(
        self, column_to_impute, column_for_condition, what_to_look_for, impute_value=0
    ):
        self.column_to_impute = column_to_impute
        self.column_for_condition = column_for_condition
        self.what_to_look_for = what_to_look_for
        self.impute_value = impute_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy[self.column_to_impute] = np.where(
            X_copy[self.column_for_condition].isin(self.what_to_look_for),
            self.impute_value,
            X_copy[self.column_to_impute],
        )

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return [self.column_to_impute, self.column_for_condition]


# %%
def instantiate_feature_creation_column_transformer():
    application_flag_columns = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
    ]

    ext_sources_columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    cb_inquiry_columns = [
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    feature_creation = ColumnTransformer(
        [
            (
                "CyclicalEncodeHrs",
                CyclicalFeatures(drop_original=True),
                ["HOUR_APPR_PROCESS_START"],
            ),
            (
                "nAppFlagAllZeros",
                MathFeatures(
                    variables=application_flag_columns,
                    func="sum",
                    new_variables_names=["nFLAGS_ALL_ZEROS"],
                ),
                application_flag_columns,
            ),
            (
                "ANY_EXT_SOURCE_MISSING",
                AggregateMissingIndicator(
                    columns=ext_sources_columns, output_name="ANY_EXT_SOURCE_MISSING"
                ),
                ext_sources_columns,
            ),
            (
                "ANY_CREDIT_BUREAU_INQ_MISSING",
                AggregateMissingIndicator(
                    columns=cb_inquiry_columns, output_name="ANY_CREDIT_BUREAU_INQ_MISSING"
                ),
                cb_inquiry_columns,
            ),
            (
                "EXT_SOURCE_MEAN",
                MathFeatures(
                    variables=ext_sources_columns,
                    func="mean",
                    missing_values="ignore",
                    new_variables_names=["EXT_SOURCE_MEAN"],
                ),
                ext_sources_columns,
            ),
            (
                "EXT_SOURCE_STD",
                MathFeatures(
                    variables=ext_sources_columns,
                    func="std",
                    missing_values="ignore",
                    new_variables_names=["EXT_SOURCE_STD"],
                ),
                ext_sources_columns,
            ),
            (
                "ImputeDaysEmployed",
                ImputeBasedOnOtherColumn(
                    column_to_impute="DAYS_EMPLOYED",
                    column_for_condition="NAME_INCOME_TYPE",
                    what_to_look_for=["Pensioner", "Unemployed"],
                ),
                ["DAYS_EMPLOYED", "NAME_INCOME_TYPE"],
            ),
            (
                "CREDIT_ANNUITY_RATIO",
                RelativeFeatures(
                    variables=["AMT_CREDIT", "AMT_ANNUITY"],
                    reference=["AMT_ANNUITY"],
                    func=["div"],
                    missing_values="ignore",
                ),
                ["AMT_CREDIT", "AMT_ANNUITY"],
            ),
            (
                "CREDIT_BY_INCOME",
                RelativeFeatures(
                    variables=["AMT_CREDIT", "AMT_INCOME_TOTAL"],
                    reference=["AMT_INCOME_TOTAL"],
                    func=["div"],
                    missing_values="ignore",
                ),
                ["AMT_CREDIT", "AMT_INCOME_TOTAL"],
            ),
            (
                "ANNUITY_BY_INCOME",
                RelativeFeatures(
                    variables=["AMT_ANNUITY", "AMT_INCOME_TOTAL"],
                    reference=["AMT_INCOME_TOTAL"],
                    func=["div"],
                    missing_values="ignore",
                ),
                ["AMT_ANNUITY", "AMT_INCOME_TOTAL"],
            ),
        ],
        remainder="passthrough",
    )

    return feature_creation, application_flag_columns, ext_sources_columns, cb_inquiry_columns


# %%
def instantiate_column_categories_and_column_imputer():
    numeric_columns = [
        "HOUR_APPR_PROCESS_START_sin",
        "HOUR_APPR_PROCESS_START_cos",
        "nFLAGS_ALL_ZEROS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "DAYS_EMPLOYED",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_CREDIT_div_AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT_div_AMT_INCOME_TOTAL",
        "AMT_ANNUITY_div_AMT_INCOME_TOTAL",
        "CNT_CHILDREN",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
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
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "n_cb_applications",
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
        "CREDIT_ACTIVE_Active_cts",
        "CREDIT_TYPE_Consumer credit_cts",
        "CREDIT_TYPE_Credit card_cts",
        "CREDIT_TYPE_Microloan_cts",
        "CREDIT_TYPE_Mortgage_cts",
        "n_old_credit_balances",
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
        "n_old_cash_balances",
        "CNT_INSTALMENT_min",
        "CNT_INSTALMENT_FUTURE_median",
        "CNT_INSTALMENT_FUTURE_std",
        "MONTHS_BALANCE_min",
        "MONTHS_BALANCE_max_y",
        "MONTHS_BALANCE_std",
        "SK_DPD_std",
        "SK_DPD_DEF_max",
        "NAME_CONTRACT_STATUS_Active_cts",
        "NAME_CONTRACT_STATUS_Completed_cts",
        "n_prev_apps",
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
        "NAME_CONTRACT_STATUS_Approved_cts",
        "CODE_REJECT_REASON_HC_cts",
        "CODE_REJECT_REASON_LIMIT_cts",
        "CODE_REJECT_REASON_SCOFR_cts",
        "NAME_CLIENT_TYPE_New_cts",
        "NAME_CLIENT_TYPE_Refreshed_cts",
        "NAME_GOODS_CATEGORY_Furniture_cts",
        "NAME_GOODS_CATEGORY_Mobile_cts",
        "NAME_PORTFOLIO_Cards_cts",
        "NAME_PRODUCT_TYPE_walk-in_cts",
        "CHANNEL_TYPE_AP+ (Cash loan)_cts",
        "NAME_SELLER_INDUSTRY_XNA_cts",
        "NAME_YIELD_GROUP_high_cts",
        "NAME_YIELD_GROUP_low_action_cts",
        "NAME_YIELD_GROUP_low_normal_cts",
        "PRODUCT_COMBINATION_Cash Street: high_cts",
        "PRODUCT_COMBINATION_Cash Street: middle_cts",
        "PRODUCT_COMBINATION_Cash X-Sell: high_cts",
        "PRODUCT_COMBINATION_Cash X-Sell: low_cts",
        "PRODUCT_COMBINATION_POS industry with interest_cts",
        "PRODUCT_COMBINATION_POS industry without interest_cts",
        "PRODUCT_COMBINATION_POS mobile with interest_cts",
        "n_prev_payments",
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
    ]

    boolean_columns = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "ANY_EXT_SOURCE_MISSING",
        "ANY_CREDIT_BUREAU_INQ_MISSING",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "LIVE_CITY_NOT_WORK_CITY",
        "EMERGENCYSTATE_MODE",
    ]

    categorical_columns = [
        "NAME_INCOME_TYPE",
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_TYPE_SUITE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
    ]

    document_flag_columns = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "nFLAGS_ALL_ZEROS",
    ]

    housing_object_columns = [
        "FLAG_OWN_REALTY",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ]

    housing_numeric_columns = [
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
    ]

    contact_flag_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    columns_to_drop = [
        "AMT_ANNUITY_div_AMT_ANNUITY",
        "AMT_INCOME_TOTAL_div_AMT_INCOME_TOTAL",
    ]

    columns_to_drop_boruta_and_corr = [
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "CNT_CHILDREN",
        "NAME_TYPE_SUITE",
        "REGION_RATING_CLIENT_W_CITY",
        "WEEKDAY_APPR_PROCESS_START",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "LIVE_CITY_NOT_WORK_CITY",
        "ORGANIZATION_TYPE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ]

    credit_bureau_columns = [
        "n_cb_applications",
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
        "STATUS_1_cts_x_std",
        "CREDIT_ACTIVE_Active_cts",
        "CREDIT_TYPE_Consumer credit_cts",
        "CREDIT_TYPE_Credit card_cts",
        "CREDIT_TYPE_Microloan_cts",
        "CREDIT_TYPE_Mortgage_cts",
        "n_old_credit_balances",
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
    ]

    prev_home_credit_columns = [
        "n_old_cash_balances",
        "CNT_INSTALMENT_min",
        "CNT_INSTALMENT_FUTURE_median",
        "CNT_INSTALMENT_FUTURE_std",
        "MONTHS_BALANCE_min",
        "MONTHS_BALANCE_max_y",
        "MONTHS_BALANCE_std",
        "SK_DPD_std",
        "SK_DPD_DEF_max",
        "NAME_CONTRACT_STATUS_Active_cts",
        "NAME_CONTRACT_STATUS_Completed_cts",
        "n_prev_apps",
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
        "NAME_CONTRACT_STATUS_Approved_cts",
        "CODE_REJECT_REASON_HC_cts",
        "CODE_REJECT_REASON_LIMIT_cts",
        "CODE_REJECT_REASON_SCOFR_cts",
        "NAME_CLIENT_TYPE_New_cts",
        "NAME_CLIENT_TYPE_Refreshed_cts",
        "NAME_GOODS_CATEGORY_Furniture_cts",
        "NAME_GOODS_CATEGORY_Mobile_cts",
        "NAME_PORTFOLIO_Cards_cts",
        "NAME_PRODUCT_TYPE_walk-in_cts",
        "CHANNEL_TYPE_AP+ (Cash loan)_cts",
        "NAME_SELLER_INDUSTRY_XNA_cts",
        "NAME_YIELD_GROUP_high_cts",
        "NAME_YIELD_GROUP_low_action_cts",
        "NAME_YIELD_GROUP_low_normal_cts",
        "PRODUCT_COMBINATION_Cash Street: high_cts",
        "PRODUCT_COMBINATION_Cash Street: middle_cts",
        "PRODUCT_COMBINATION_Cash X-Sell: high_cts",
        "PRODUCT_COMBINATION_Cash X-Sell: low_cts",
        "PRODUCT_COMBINATION_POS industry with interest_cts",
        "PRODUCT_COMBINATION_POS industry without interest_cts",
        "PRODUCT_COMBINATION_POS mobile with interest_cts",
        "n_prev_payments",
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
    ]

    column_imputer = ColumnTransformer(
        [
            (
                "objects_with_unknown",
                SimpleImputer(strategy="constant", fill_value="unknown"),
                boolean_columns + categorical_columns,
            ),
            (
                "numeric_with_0",
                SimpleImputer(strategy="constant", fill_value=0),
                numeric_columns,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return (column_imputer, numeric_columns, boolean_columns, categorical_columns, document_flag_columns,
            housing_object_columns, housing_numeric_columns, contact_flag_columns, columns_to_drop,
            columns_to_drop_boruta_and_corr, credit_bureau_columns, prev_home_credit_columns)


# %%
def lgbm_baseline_model(X_train, y_train, X_val, y_val):
    X_train_lgbm_baseline = clean_column_names(X_train.copy())
    y_train_lgbm_baseline = y_train.copy()
    X_val_lgbm_baseline = clean_column_names(X_val.copy())
    y_val_lgbm_baseline = y_val.copy()

    X_train_categorical = X_train_lgbm_baseline.select_dtypes(include=["object"])
    X_train_lgbm_baseline[X_train_categorical.columns] = X_train_lgbm_baseline[
        X_train_categorical.columns
    ].astype("category")
    X_val_categorical = X_val_lgbm_baseline.select_dtypes(include=["object"])
    X_val_lgbm_baseline[X_val_categorical.columns] = X_val_lgbm_baseline[
        X_val_categorical.columns
    ].astype("category")

    num_neg = (y_train_lgbm_baseline == 0).sum()
    num_pos = (y_train_lgbm_baseline == 1).sum()
    scale_pos_weight = num_neg / num_pos

    lgbm_baseline_train = lgb.Dataset(
        X_train_lgbm_baseline,
        label=y_train_lgbm_baseline,
        categorical_feature=list(X_train_categorical.columns),
    )
    lgbm_baseline_val = lgb.Dataset(
        X_val_lgbm_baseline,
        label=y_val_lgbm_baseline,
        categorical_feature=list(X_val_categorical.columns),
        reference=lgbm_baseline_train,
    )

    lgbm_baseline_param = {
        "objective": "binary",
        "num_leaves": 31,
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
    }
    num_round = 100
    bst_baseline_lgbm_trained = lgb.train(
        lgbm_baseline_param, lgbm_baseline_train, num_round, valid_sets=[lgbm_baseline_val]
    )
    return (bst_baseline_lgbm_trained, X_train_lgbm_baseline, y_train_lgbm_baseline, X_val_lgbm_baseline,
            y_val_lgbm_baseline)


# %%
def lgbm_preprocessing(
    raw_X_df,
    y_df,
        numeric_columns=None,
        boolean_columns=None,
        categorical_columns=None,
        columns_to_drop=None,
):
    feature_creation, application_flag_columns, ext_sources_columns, cb_inquiry_columns = (
        instantiate_feature_creation_column_transformer())
    (column_imputer, num_cols, bool_cols, cat_cols, document_flag_columns, housing_object_columns,
     housing_numeric_columns, contact_flag_columns, cols_to_drop, columns_to_drop_boruta_and_corr,
     credit_bureau_columns, prev_home_credit_columns) = instantiate_column_categories_and_column_imputer()
    if columns_to_drop is None:
        columns_to_drop = cols_to_drop
    if categorical_columns is None:
        categorical_columns = cat_cols
    if boolean_columns is None:
        boolean_columns = bool_cols
    if numeric_columns is None:
        numeric_columns = num_cols

    X = raw_X_df.copy()
    X_full = feature_creation.fit_transform(X)
    columns = feature_creation.get_feature_names_out()
    X_df = pd.DataFrame(X_full, columns=columns)
    X_df.columns = [remove_verbose_col_name(col) for col in X_df.columns]
    X_df = X_df.loc[:, ~X_df.columns.duplicated()].copy()
    feat_drop = DropFeatures(features_to_drop=columns_to_drop)
    X_df = feat_drop.fit_transform(X_df)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df = column_imputer.fit_transform(X_df)
    columns = column_imputer.get_feature_names_out()
    X_df = pd.DataFrame(X_df, columns=columns)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df[boolean_columns] = X_df[boolean_columns].astype("bool")
    X_df[categorical_columns] = X_df[categorical_columns].astype("category")
    X_lgbm = clean_column_names(X_df.copy())
    y_lgbm = y_df.copy()
    num_neg = (y_lgbm == 0).sum()
    num_pos = (y_lgbm == 1).sum()
    scale_pos_weight = num_neg / num_pos
    return X_lgbm, y_lgbm, scale_pos_weight


# %%
def log_reg_preprocessing(
    raw_X_df,
    y_df,
    numeric_columns = None,
    boolean_columns = None,
    categorical_columns = None,
    columns_to_drop = None,

):
    feature_creation, application_flag_columns, ext_sources_columns, cb_inquiry_columns = (
    instantiate_feature_creation_column_transformer())
    (column_imputer, num_cols, bool_cols, cat_cols, document_flag_columns, housing_object_columns,
    housing_numeric_columns, contact_flag_columns, cols_to_drop, columns_to_drop_boruta_and_corr,
    credit_bureau_columns, prev_home_credit_columns) = instantiate_column_categories_and_column_imputer()

    if columns_to_drop is None:
        columns_to_drop = cols_to_drop
    if categorical_columns is None:
        categorical_columns = cat_cols
    if boolean_columns is None:
        boolean_columns = bool_cols
    if numeric_columns is None:
        numeric_columns = num_cols

    X = raw_X_df.copy()
    X_full = feature_creation.fit_transform(X)
    columns = feature_creation.get_feature_names_out()
    X_df = pd.DataFrame(X_full, columns=columns)
    X_df.columns = [remove_verbose_col_name(col) for col in X_df.columns]
    X_df = X_df.loc[:, ~X_df.columns.duplicated()].copy()
    feat_drop = DropFeatures(features_to_drop=columns_to_drop)
    X_df = feat_drop.fit_transform(X_df)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df = column_imputer.fit_transform(X_df)
    columns = column_imputer.get_feature_names_out()
    X_df = pd.DataFrame(X_df, columns=columns)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df[boolean_columns] = X_df[boolean_columns].astype("bool")
    X_df[boolean_columns] = X_df[boolean_columns].astype("int")
    X_df[categorical_columns] = X_df[categorical_columns].astype("category")
    X_log_reg = X_df
    y_log_reg = y_df.copy()
    return X_log_reg, y_log_reg


#%%
# note this is the same for rf and lr
def rf_preprocessing(
    raw_X_df,
    y_df,
        numeric_columns=None,
        boolean_columns=None,
        categorical_columns=None,
        columns_to_drop=None,

):
    feature_creation, application_flag_columns, ext_sources_columns, cb_inquiry_columns = (
        instantiate_feature_creation_column_transformer())
    (column_imputer, num_cols, bool_cols, cat_cols, document_flag_columns, housing_object_columns,
     housing_numeric_columns, contact_flag_columns, cols_to_drop, columns_to_drop_boruta_and_corr,
     credit_bureau_columns, prev_home_credit_columns) = instantiate_column_categories_and_column_imputer()

    if columns_to_drop is None:
        columns_to_drop = cols_to_drop
    if categorical_columns is None:
        categorical_columns = cat_cols
    if boolean_columns is None:
        boolean_columns = bool_cols
    if numeric_columns is None:
        numeric_columns = num_cols

    X = raw_X_df.copy()
    X_full = feature_creation.fit_transform(X)
    columns = feature_creation.get_feature_names_out()
    X_df = pd.DataFrame(X_full, columns=columns)
    X_df.columns = [remove_verbose_col_name(col) for col in X_df.columns]
    X_df = X_df.loc[:, ~X_df.columns.duplicated()].copy()
    feat_drop = DropFeatures(features_to_drop=columns_to_drop)
    X_df = feat_drop.fit_transform(X_df)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df = column_imputer.fit_transform(X_df)
    columns = column_imputer.get_feature_names_out()
    X_df = pd.DataFrame(X_df, columns=columns)
    X_df = to_downcasted_numeric(X_df, numeric_columns)
    X_df[boolean_columns] = X_df[boolean_columns].astype("bool")
    X_df[boolean_columns] = X_df[boolean_columns].astype("int")
    X_df[categorical_columns] = X_df[categorical_columns].astype("category")
    X_rf = X_df
    y_rf = y_df.copy()

    return X_rf, y_rf


#%%
def general_preprocessing(df):

    set_config(transform_output="pandas")

    feats_to_log = ["AMT_CREDIT", "AMT_ANNUITY", "n_cb_applications", "n_old_cash_balances", "n_old_credit_balances",
                    "n_prev_apps", "n_prev_payments", "AMT_PAYMENT_min", "CNT_INSTALMENT_FUTURE_std", "AMT_GOODS_PRICE"]

    feats_to_abs_log = ['DAYS_CREDIT_median', 'DAYS_CREDIT_max', 'DAYS_CREDIT_UPDATE_mean', 'DAYS_EMPLOYED',
                        'DAYS_INSTALMENT_max']

    impute_log_transform_features = ColumnTransformer([
        ("impute_with_0", SimpleImputer(strategy="constant", fill_value=0),
         feats_to_log + feats_to_abs_log + ["DAYS_CREDIT_ENDDATE_mean"])
    ],
        remainder='passthrough', verbose_feature_names_out=False)

    X = impute_log_transform_features.fit_transform(df)
    columns = impute_log_transform_features.get_feature_names_out()
    X_df = pd.DataFrame(X, columns=columns)
    X_df.columns = [remove_verbose_col_name(col) for col in X_df.columns]

    LogTransformFeatures = ColumnTransformer([
        ("original", "passthrough", feats_to_log),
        ("log_transformed", LogCpTransformer(C=1), feats_to_log),
    ], remainder="drop")

    AbsLogTransformFeatures = ColumnTransformer([
        ("original", "passthrough", feats_to_abs_log),
        ("log_transformed", LogCpTransformer(C='auto'), feats_to_abs_log),
    ], remainder="drop")

    signedAbsLogTransformer = ColumnTransformer([
        ("original", "passthrough", ["DAYS_CREDIT_ENDDATE_mean"]),
        ("log_transformed", FunctionTransformer(log_signed_transform), ["DAYS_CREDIT_ENDDATE_mean"]),
    ], remainder="drop")

    application_flag_columns = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                                'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                                'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                                'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                                'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    cb_inqs = ['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    feature_creation = ColumnTransformer([
        ("CyclicalEncodeHrs", CyclicalFeatures(), ["HOUR_APPR_PROCESS_START"]),
        ("nAppFlagAllZeros",
         MathFeatures(variables=application_flag_columns, func="sum", new_variables_names=["nFLAGS_ALL_ZEROS"]),
         application_flag_columns),
        ("ANY_EXT_SOURCE_MISSING", AggregateMissingIndicator(columns=ext_sources, output_name="ANY_EXT_SOURCE_MISSING"),
         ext_sources),
        ("ANY_CREDIT_BUREAU_INQ_MISSING",
         AggregateMissingIndicator(columns=cb_inqs, output_name="ANY_CREDIT_BUREAU_INQ_MISSING"), cb_inqs),
        ("EXT_SOURCE_MEAN", MathFeatures(variables=ext_sources, func="mean", missing_values='ignore',
                                         new_variables_names=["EXT_SOURCE_MEAN"]), ext_sources),
        ("EXT_SOURCE_STD", MathFeatures(variables=ext_sources, func="std", missing_values='ignore',
                                        new_variables_names=["EXT_SOURCE_STD"]), ext_sources),
        ("ImputeDaysEmployed",
         ImputeBasedOnOtherColumn(column_to_impute='DAYS_EMPLOYED', column_for_condition='NAME_INCOME_TYPE',
                                  what_to_look_for=['Pensioner', 'Unemployed']), ["DAYS_EMPLOYED", "NAME_INCOME_TYPE"]),
        ("LogTransformFeatures", LogTransformFeatures, feats_to_log),
        ("AbsLogTransformFeatures", AbsLogTransformFeatures, feats_to_abs_log),
        ("signedAbsLogTransform", signedAbsLogTransformer, ["DAYS_CREDIT_ENDDATE_mean"])
    ],
        remainder="passthrough")

    feature_creation_pipeline = Pipeline(steps=[
        ('impute_log_transform_features', impute_log_transform_features),
        ('feature_creation', feature_creation)
    ])

    df_processed = feature_creation_pipeline.fit_transform(X_df)
    df_processed = rename_log_columns(df_processed)
    df_processed.columns = [remove_verbose_col_name(col) for col in df_processed.columns]

    ratio_features = ColumnTransformer([
        ("CREDIT_ANNUITY_RATIO",
         RelativeFeatures(variables=["AMT_CREDIT"], reference=["AMT_ANNUITY"], func=["div"], missing_values='ignore',
                          fill_value=np.nan), ["AMT_CREDIT", "AMT_ANNUITY"]),
        ("CREDIT_BY_INCOME", RelativeFeatures(variables=["AMT_CREDIT"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                                              missing_values='ignore', fill_value=np.nan),
         ["AMT_CREDIT", "AMT_INCOME_TOTAL"]),
        ("ANNUITY_BY_INCOME", RelativeFeatures(variables=["AMT_ANNUITY"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                                               missing_values='ignore', fill_value=np.nan),
         ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]),
        ("logCREDIT_ANNUITY_RATIO",
         RelativeFeatures(variables=["AMT_CREDIT_log"], reference=["AMT_ANNUITY_log"], func=["div"],
                          missing_values='ignore', fill_value=np.nan), ["AMT_CREDIT_log", "AMT_ANNUITY_log"]),
        ("logCREDIT_BY_INCOME",
         RelativeFeatures(variables=["AMT_CREDIT_log"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                          missing_values='ignore', fill_value=np.nan), ["AMT_CREDIT_log", "AMT_INCOME_TOTAL"]),
        ("logANNUITY_BY_INCOME",
         RelativeFeatures(variables=["AMT_ANNUITY_log"], reference=["AMT_INCOME_TOTAL"], func=["div"],
                          missing_values='ignore', fill_value=np.nan), ["AMT_ANNUITY_log", "AMT_INCOME_TOTAL"]),
    ], remainder="passthrough")

    df_processed = ratio_features.fit_transform(df_processed)
    df_processed.columns = [remove_verbose_col_name(col) for col in df_processed.columns]
    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

    boolean_columns = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
                       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                       'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'ANY_EXT_SOURCE_MISSING',
                       'ANY_CREDIT_BUREAU_INQ_MISSING', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL',
                       'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
                       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION',
                       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY',
                       'EMERGENCYSTATE_MODE']

    categorical_columns = ['NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_TYPE_SUITE',
                           'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
                           'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                           'WALLSMATERIAL_MODE']

    df_processed[boolean_columns] = df_processed[boolean_columns].astype('bool')
    df_processed[categorical_columns] = df_processed[categorical_columns].astype('category')

    df_processed = clean_column_names(df_processed)

    return df_processed
