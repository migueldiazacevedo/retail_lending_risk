import pickle
from typing import Dict, Optional

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field


class LoanData(BaseModel):
    AMT_CREDIT: Optional[float]
    AMT_ANNUITY: Optional[float]
    AMT_CREDIT_div_AMT_ANNUITY: Optional[float]
    AMT_INCOME_TOTAL: Optional[float]
    AMT_ANNUITY_div_AMT_INCOME_TOTAL: Optional[float]
    AMT_ANNUITY_log: Optional[float]
    AMT_ANNUITY_log_div_AMT_INCOME_TOTAL: Optional[float]
    HOUR_APPR_PROCESS_START_sin: Optional[float]
    FLAG_DOCUMENT_2: bool
    FLAG_DOCUMENT_3: bool
    FLAG_DOCUMENT_4: bool
    FLAG_DOCUMENT_5: bool
    FLAG_DOCUMENT_6: bool
    FLAG_DOCUMENT_7: bool
    FLAG_DOCUMENT_8: bool
    FLAG_DOCUMENT_9: bool
    FLAG_DOCUMENT_10: bool
    FLAG_DOCUMENT_11: bool
    FLAG_DOCUMENT_12: bool
    FLAG_DOCUMENT_13: bool
    FLAG_DOCUMENT_14: bool
    FLAG_DOCUMENT_15: bool
    FLAG_DOCUMENT_16: bool
    FLAG_DOCUMENT_17: bool
    FLAG_DOCUMENT_18: bool
    FLAG_DOCUMENT_19: bool
    FLAG_DOCUMENT_21: bool
    nFLAGS_ALL_ZEROS: bool
    EXT_SOURCE_1: Optional[float]
    EXT_SOURCE_2: Optional[float]
    EXT_SOURCE_3: Optional[float]
    ANY_EXT_SOURCE_MISSING: bool
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float]
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float]
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float]
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float]
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float]
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float]
    EXT_SOURCE_MEAN: Optional[float]
    EXT_SOURCE_STD: Optional[float]
    DAYS_EMPLOYED: Optional[float]
    AMT_PAYMENT_min: Optional[float]
    CNT_INSTALMENT_FUTURE_std: Optional[float]
    n_old_credit_balances_log: Optional[float]
    n_prev_apps_log: Optional[float]
    n_prev_payments_log: Optional[float]
    AMT_PAYMENT_min_log: Optional[float]
    DAYS_CREDIT_max: Optional[float]
    DAYS_CREDIT_UPDATE_mean: Optional[float]
    DAYS_INSTALMENT_max: Optional[float]
    DAYS_CREDIT_ENDDATE_mean_log: Optional[float]
    NAME_CONTRACT_TYPE: Optional[str]
    CODE_GENDER: Optional[str]
    FLAG_OWN_CAR: bool
    FLAG_OWN_REALTY: bool
    CNT_CHILDREN: Optional[int]
    NAME_TYPE_SUITE: Optional[str]
    NAME_EDUCATION_TYPE: Optional[str]
    NAME_FAMILY_STATUS: Optional[str]
    NAME_HOUSING_TYPE: Optional[str]
    REGION_POPULATION_RELATIVE: Optional[float]
    DAYS_BIRTH: Optional[float]
    DAYS_REGISTRATION: Optional[float]
    DAYS_ID_PUBLISH: Optional[float]
    OWN_CAR_AGE: Optional[float]
    FLAG_MOBIL: bool
    FLAG_WORK_PHONE: bool
    FLAG_CONT_MOBILE: bool
    FLAG_PHONE: bool
    FLAG_EMAIL: bool
    OCCUPATION_TYPE: Optional[str]
    REGION_RATING_CLIENT: Optional[int]
    WEEKDAY_APPR_PROCESS_START: Optional[str]
    REG_REGION_NOT_LIVE_REGION: bool
    LIVE_REGION_NOT_WORK_REGION: bool
    REG_CITY_NOT_LIVE_CITY: bool
    REG_CITY_NOT_WORK_CITY: bool
    ORGANIZATION_TYPE: Optional[str]
    YEARS_BUILD_AVG: Optional[float]
    ENTRANCES_AVG: Optional[float]
    FLOORSMAX_AVG: Optional[float]
    LANDAREA_AVG: Optional[float]
    NONLIVINGAPARTMENTS_AVG: Optional[float]
    NONLIVINGAREA_AVG: Optional[float]
    BASEMENTAREA_MODE: Optional[float]
    YEARS_BEGINEXPLUATATION_MODE: Optional[float]
    LIVINGAPARTMENTS_MODE: Optional[float]
    COMMONAREA_MEDI: Optional[float]
    FONDKAPREMONT_MODE: Optional[str]
    HOUSETYPE_MODE: Optional[str]
    WALLSMATERIAL_MODE: Optional[str]
    EMERGENCYSTATE_MODE: Optional[str]
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float]
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float]
    DAYS_LAST_PHONE_CHANGE: Optional[float]
    AMT_ANNUITY_max_x: Optional[float]
    AMT_CREDIT_MAX_OVERDUE_max: Optional[float]
    AMT_CREDIT_MAX_OVERDUE_std: Optional[float]
    AMT_CREDIT_SUM_median: Optional[float]
    AMT_CREDIT_SUM_DEBT_max: Optional[float]
    AMT_CREDIT_SUM_DEBT_std: Optional[float]
    AMT_CREDIT_SUM_LIMIT_max: Optional[float]
    AMT_CREDIT_SUM_LIMIT_mean: Optional[float]
    AMT_CREDIT_SUM_OVERDUE_max: Optional[float]
    CREDIT_DAY_OVERDUE_std: Optional[float]
    DAYS_CREDIT_min: Optional[float]
    DAYS_CREDIT_std: Optional[float]
    DAYS_CREDIT_ENDDATE_min: Optional[float]
    DAYS_CREDIT_ENDDATE_max: Optional[float]
    DAYS_CREDIT_UPDATE_max: Optional[float]
    DAYS_ENDDATE_FACT_max: Optional[float]
    MONTHS_BALANCE_mean_mean: Optional[float]
    STATUS_1_cts_max: Optional[float]
    STATUS_1_cts_mean: Optional[float]
    STATUS_1_cts_std: Optional[float]
    CREDIT_ACTIVE_Active_cts: Optional[float]
    CREDIT_TYPE_Consumer_credit_cts: Optional[float]
    CREDIT_TYPE_Credit_card_cts: Optional[float]
    CREDIT_TYPE_Microloan_cts: Optional[float]
    CREDIT_TYPE_Mortgage_cts: Optional[float]
    AMT_CREDIT_LIMIT_ACTUAL_median: Optional[float]
    AMT_CREDIT_LIMIT_ACTUAL_std: Optional[float]
    AMT_DRAWINGS_ATM_CURRENT_max: Optional[float]
    AMT_DRAWINGS_ATM_CURRENT_mean: Optional[float]
    AMT_DRAWINGS_CURRENT_std: Optional[float]
    AMT_DRAWINGS_POS_CURRENT_mean: Optional[float]
    AMT_DRAWINGS_POS_CURRENT_std: Optional[float]
    AMT_PAYMENT_TOTAL_CURRENT_max: Optional[float]
    AMT_RECEIVABLE_PRINCIPAL_max: Optional[float]
    AMT_RECIVABLE_min: Optional[float]
    AMT_RECIVABLE_median: Optional[float]
    CNT_DRAWINGS_ATM_CURRENT_mean: Optional[float]
    CNT_DRAWINGS_ATM_CURRENT_std: Optional[float]
    CNT_DRAWINGS_CURRENT_max: Optional[float]
    CNT_DRAWINGS_CURRENT_mean: Optional[float]
    CNT_INSTALMENT_MATURE_CUM_min: Optional[float]
    CNT_INSTALMENT_MATURE_CUM_std: Optional[float]
    MONTHS_BALANCE_max_x: Optional[float]
    CNT_INSTALMENT_min: Optional[float]
    CNT_INSTALMENT_FUTURE_median: Optional[float]
    MONTHS_BALANCE_std: Optional[float]
    SK_DPD_std: Optional[float]
    SK_DPD_DEF_max: Optional[float]
    NAME_CONTRACT_STATUS_Active_cts: Optional[float]
    NAME_CONTRACT_STATUS_Completed_cts: Optional[float]
    AMT_ANNUITY_max_y: Optional[float]
    AMT_ANNUITY_mean: Optional[float]
    AMT_CREDIT_median: Optional[float]
    AMT_CREDIT_std: Optional[float]
    AMT_DOWN_PAYMENT_max: Optional[float]
    CNT_PAYMENT_max: Optional[float]
    CNT_PAYMENT_mean: Optional[float]
    CNT_PAYMENT_std: Optional[float]
    DAYS_DECISION_min: Optional[float]
    DAYS_DECISION_mean: Optional[float]
    DAYS_DECISION_std: Optional[float]
    DAYS_FIRST_DRAWING_std: Optional[float]
    DAYS_FIRST_DUE_min: Optional[float]
    DAYS_FIRST_DUE_std: Optional[float]
    DAYS_LAST_DUE_1ST_VERSION_median: Optional[float]
    DAYS_TERMINATION_mean: Optional[float]
    HOUR_APPR_PROCESS_START_min: Optional[float]
    RATE_DOWN_PAYMENT_max: Optional[float]
    RATE_DOWN_PAYMENT_std: Optional[float]
    SELLERPLACE_AREA_max: Optional[float]
    NAME_CONTRACT_STATUS_Approved_cts: Optional[float]
    CODE_REJECT_REASON_HC_cts: Optional[float]
    CODE_REJECT_REASON_LIMIT_cts: Optional[float]
    CODE_REJECT_REASON_SCOFR_cts: Optional[float]
    NAME_CLIENT_TYPE_New_cts: Optional[float]
    NAME_CLIENT_TYPE_Refreshed_cts: Optional[float]
    NAME_GOODS_CATEGORY_Furniture_cts: Optional[float]
    NAME_GOODS_CATEGORY_Mobile_cts: Optional[float]
    NAME_PORTFOLIO_Cards_cts: Optional[float]
    NAME_PRODUCT_TYPE_walk_in_cts: Optional[float]
    CHANNEL_TYPE_AP___Cash_loan__cts: Optional[float]
    NAME_SELLER_INDUSTRY_Auto_cts: Optional[float]
    NAME_SELLER_INDUSTRY_Consumer_electronics_cts: Optional[float]
    NAME_SELLER_INDUSTRY_Furniture_cts: Optional[float]
    CNT_PAYMENT_max_y: Optional[float]
    NAME_CONTRACT_STATUS_Refused_cts: Optional[float]
    NAME_SELLER_INDUSTRY_XNA_cts: Optional[float] = Field(None, description="Counts of NAME_SELLER_INDUSTRY_XNA")
    NAME_YIELD_GROUP_high_cts: Optional[float] = Field(None, description="Counts of NAME_YIELD_GROUP_high")
    NAME_YIELD_GROUP_low_action_cts: Optional[float] = Field(None, description="Counts of NAME_YIELD_GROUP_low_action")
    NAME_YIELD_GROUP_low_normal_cts: Optional[float] = Field(None, description="Counts of NAME_YIELD_GROUP_low_normal")
    PRODUCT_COMBINATION_Cash_Street__high_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_Cash_Street__high")
    PRODUCT_COMBINATION_Cash_Street__middle_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_Cash_Street__middle")
    PRODUCT_COMBINATION_Cash_X_Sell__high_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_Cash_X_Sell__high")
    PRODUCT_COMBINATION_Cash_X_Sell__low_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_Cash_X_Sell__low")
    PRODUCT_COMBINATION_POS_industry_with_interest_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_POS_industry_with_interest")
    PRODUCT_COMBINATION_POS_industry_without_interest_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_POS_industry_without_interest")
    PRODUCT_COMBINATION_POS_mobile_with_interest_cts: Optional[float] = Field(None, description="Counts of PRODUCT_COMBINATION_POS_mobile_with_interest")
    AMT_PAYMENT_max: Optional[float] = Field(None, description="Maximum of AMT_PAYMENT")
    AMT_PAYMENT_std: Optional[float] = Field(None, description="Standard deviation of AMT_PAYMENT")
    NUM_INSTALMENT_NUMBER_std: Optional[float] = Field(None, description="Standard deviation of NUM_INSTALMENT_NUMBER")
    NUM_INSTALMENT_VERSION_min: Optional[float] = Field(None, description="Minimum of NUM_INSTALMENT_VERSION")
    NUM_INSTALMENT_VERSION_max: Optional[float] = Field(None, description="Maximum of NUM_INSTALMENT_VERSION")
    NUM_INSTALMENT_VERSION_mean: Optional[float] = Field(None, description="Mean of NUM_INSTALMENT_VERSION")


model_file_path = "/model/model.pkl"
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)


X = pd.read_parquet("/data/X_data.parquet")
categorical_columns = list(X.select_dtypes(["category", "bool"]).columns)
X[categorical_columns] = X[categorical_columns].astype("object")


app = FastAPI()


class PredictionOut(BaseModel):
    default_proba: float


def preprocess_input(data: Dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df[categorical_columns] = df[categorical_columns].astype("object")
    return df


@app.post("/predict", response_model=PredictionOut)
def predict(input_data: LoanData):
    input_df = preprocess_input(input_data.dict())
    preds = model.predict_proba(input_df)[0, 1]
    result = {"default_proba": preds}
    return result


@app.get("/")
def home():
    return {"message": "Loan default prediction API is running"}
