import gc
import re
import time
from contextlib import contextmanager

from imblearn.combine import SMOTEENN
# from Pevious_appl import Join_aggregated_lastprevappl
# from referans import display_importances, submission_file_name
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from Functions import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# To display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df.columns = ["".join(c if c.isalnum(   ) else "_" for c in str(x)) for x in df.columns]
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns




def application_train_test(num_rows=None, nan_as_category=False):
    train = pd.read_csv(r"E:\PROJECTS\Home_Credit_new\Data\application_train.csv", nrows=num_rows)
    test = pd.read_csv(r"E:\PROJECTS\Home_Credit_new\Data\application_test.csv", nrows=num_rows)

    train = train.append(test).reset_index()

    def one_hot_encoder(train, test ,categorical_cols, nan_as_category=True):
        original_columns_train = list(train.columns)
        original_columns_test = list(test.columns)
        train = pd.get_dummies(train, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
        test = pd.get_dummies(test, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
        new_columns = [c for c in train.columns if c not in original_columns_train]
        return train, test, new_columns

    outliers_update(train, test)
    days_features(train, test)
    generate_binary_from_numerical(train, test, "DAYS_LAST_PHONE_CHANGE")
    # train, test = new_features_from_EXT_features(train, test)
    new_features_domain_knowledge(train, test)
    class_synchronization(train, test)
    collect_features(train, test, "FLAG_DOCUMENT")
    drop_group_of_features(train, test, "REG_")
    collect_features_and_binary(train, test, "CREDIT_BUREAU")
    row_cross_replace(train, test, "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "Pensioner", "Pensioner")
    White_collar, Blue_collar, Laborers = occupation_rare()
    row_rename(train, test, "OCCUPATION_TYPE", "White_collar", White_collar)
    row_rename(train, test, "OCCUPATION_TYPE", "Blue_collar", Blue_collar)
    row_rename(train, test, "OCCUPATION_TYPE", "Laborers", Laborers)
    personal_asset(train, test, "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "Personal_Assets")
    row_cross_replace(train, test, "NAME_EDUCATION_TYPE", "NAME_EDUCATION_TYPE", "Academic degree", "Higher education")
    education_years(train, test)
    row_rename(train, test, "NAME_INCOME_TYPE", "Working", ["Student", "Unemployed", "Businessman"])
    row_rename(train, test, "NAME_HOUSING_TYPE", "Other",
               ["Co-op apartment", "Municipal apartment", "Office apartment", "Rented apartment", "With parents"])
    row_rename(train, test, "NAME_FAMILY_STATUS", "Married", ["Civil marriage"])
    row_rename(train, test, "NAME_FAMILY_STATUS", "Single", ["Separated", "Single / not married", "Widow"])
    log_transformation(train, test)
    drop_features(train, test)
    categorical_feats, numerical_feats = columns_dtypes(train)
    train,test, new_ohe = one_hot_encoder(train, test, categorical_feats,True)
    implement_robust_function(train, test)
    gc.collect()
    return train

def Create_New_Ext_Source_Features(df):
    df['EXT_SOURCE_3TOGETHER'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_MULTIPLY_EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_MULTIPLY_EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_MULTIPLY_EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_DIVIDE_EXT_SOURCE_2'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_DIVIDE_EXT_SOURCE_3'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_DIVIDE_EXT_SOURCE_3'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']

    return df

def bureau_and_balance(num_rows=None, nan_as_category=True):
    bb = pd.read_csv(r"E:\PROJECTS\Home_Credit_new\Data\bureau_balance.csv", nrows=num_rows)
    bb["MONTHS_BALANCE"] = bb["MONTHS_BALANCE"].abs()
    df, n_col_bb_cat = one_hot_encoder(bb, False)
    # TEKİLLEŞTİRME:Groupby ile
    # Problem: Her bir kategorik değişken  için tek tek istenen işlemin girilmesi? Bunun yerine, bir dictionar oluşturup, karşısına istediğimizi doldurtalım

    bb_aggregatiosn = {"MONTHS_BALANCE": ["min", "max", "count"]}  # Numerik için elle oluşturduk

    # Yeni ortaya çıkan kategorikler için dictionary yapısını for loop ile yazıyoruz.

    for col in n_col_bb_cat:
        bb_aggregatiosn[col] = ["mean", "sum"]
    # Gruplama yapalım
    bb_agg = df.groupby("SK_ID_BUREAU").agg(bb_aggregatiosn)
    # INDEX problemini çözelin( hiyerarşik Index)
    bb_agg.columns = pd.Index(["BB_" + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])  # her bir level ve altındakini yakalasın ve _ ile birleştisn. string birleştirme işlemi yaptık

    ## BUREAU kısmına geçelim

    bureau = pd.read_csv(r"E:\PROJECTS\Home_Credit_new\Data\bureau.csv", nrows=num_rows)

    bureau["DAYS_ENDDATE_FACT"] = bureau["DAYS_ENDDATE_FACT"].abs()
    bureau["DAYS_CREDIT"] = bureau["DAYS_CREDIT"].abs()

    # A- Kategorik Değişkenlere OHE uygula
    ##-- Amaç: kategorik değişkenelre istatistik uyuglayarak sınıfları arasında anlamlı bilgilere ulaşmak

    df, caty_col_bureau = one_hot_encoder(bureau, False)

    ###### B- TABLOLARI BİRLEŞTİRME#########

    bureau_bb = df.join(bb_agg, how="left",on="SK_ID_BUREAU")  # Bureau Balance'ı Bureau tablosuna, Bureau tablosunda olanlar üzerinden bağladık.

    # C-- SK_ID_BUREAU'dan kurtul
    ##--- Bir üst tabloda karşılığı olmadığı için kurtulmak faydalı

    bureau_bb.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    # D-- Agg döngülerini yazalım
    num_col_org = [col for col in bureau.columns if bureau[col].dtypes != "O" and col not in "SK_ID_CURR" and col not in "SK_ID_BUREAU"]  # orjinal halinin numerik değişkenlerini yakalaylım. Balance ile birleştirmeden önceki hali gerekli

    # Agg içinn istediklerimizi oluşturalım:
    bureau_bb_agg = {}
    for col in num_col_org:    bureau_bb_agg[col] = ["min", "max", "sum"]

    for col in caty_col_bureau:    bureau_bb_agg[col] = ["mean"]

    for col in n_col_bb_cat: bureau_bb_agg["BB_" + col + "_MEAN"] = ["mean"]  # balance tablosu  ohe sonrası hali için önce mean ekle bir tur

    for col in n_col_bb_cat: bureau_bb_agg["BB_" + col + "_SUM"] = ["sum"]  # balance tablosu ohe sonrası hali için sum ekle bir tur, ama statu o not found hatasını ortadan kaldırmak

     # E: Groupby alalım:
    # --SK_ID_CURR'lar tekilleşmiş oldu.
    bureau_bb_agg = bureau_bb.groupby("SK_ID_CURR").agg(bureau_bb_agg)
    # F: Hiyeraşik index problemi


    bureau_bb_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_bb_agg.columns.tolist()])

    bureau_bb_agg['Bureau_balance_existornot'] = 1

    return bureau_bb_agg




def one_hot_encoderr(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def Last_prev_prepare(num_rows=None):
    df = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\Data\previous_application.csv', nrows=num_rows)

    Date_cols = list(df.columns[df.columns.str.contains('DAYS', regex=True)])
    # # To convert date values to absolute form
    df.loc[:, Date_cols] = np.absolute(df[Date_cols])
    # # To select last previous application
    temp_df = df.groupby('SK_ID_CURR').agg({'DAYS_DECISION': 'min'})
    # Drop duplicates
    df = df.drop_duplicates(subset=['SK_ID_CURR', 'DAYS_DECISION'])

    temp_df = temp_df.reset_index()

    df = pd.merge(df, temp_df, how='inner', left_on=['SK_ID_CURR', 'DAYS_DECISION'],
                  right_on=['SK_ID_CURR', 'DAYS_DECISION'])

    return df

def Drop_columns(df):
    drop_list = ['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START',
             'HOUR_APPR_PROCESS_START', 'RATE_INTEREST_PRIVILEGED', 'AMT_GOODS_PRICE', 'DAYS_LAST_DUE', 'SELLERPLACE_AREA',
             'RATE_DOWN_PAYMENT', 'NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY', 'NAME_SELLER_INDUSTRY', 'PRODUCT_COMBINATION']

    df.drop(drop_list, axis=1, inplace=True)

    return df

def Drop_columns_before_modelling(df):
    drop_list = [#'INSTAL_PAYMENT_DIFF_MAX', 'POS_NAME_CONTRACT_STATUS_Completed_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MAX',
                 # 'EXT_SOURCE_1_MULTIPLY_EXT_SOURCE_2', 'APPROVED_AMT_ANNUITY_MEAN', 'BURO_DAYS_CREDIT_ENDDATE_MIN',
                 # 'INSTAL_PAYMENT_DIFF_MEAN', 'APPROVED_AMT_ANNUITY_SUM', 'APPROVED_DAYS_DECISION_MEAN',
                 # 'BURO_DAYS_ENDDATE_FACT_MAX',
                 # 'AMT_INCOME_TOTAL', 'DAYS_DECISION', 'INSTAL_AMT_INSTALMENT_MEAN', 'APPROVED_APP_CREDIT_PERC_MEAN',
                 # 'BURO_BB_STATUS_0_MEAN_MEAN', 'REFUSED_DAYS_DECISION_MEAN', 'LANDAREA_MEDI', 'OWN_CAR_AGE',
                 # 'PREV_NAME_YIELD_GROUP_high_MEAN', 'POS_SK_DPD_DEF_MEAN', 'EXT_SOURCE_3TOGETHER',
                 # 'BURO_DAYS_CREDIT_MAX',
                 # 'DAYS_TERMINATION', 'PREV_CODE_REJECT_REASON_HC_MEAN', 'DAYS_FIRST_DUE', 'POS_MONTHS_BALANCE_MAX',
                 # 'BURO_DAYS_CREDIT_SUM', 'INSTAL_PAYMENT_DIFF_SUM', 'PREV_DAYS_DECISION_MEAN',
                 # 'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',
                 # 'CODE_GENDER_M', 'INSTAL_PAYMENT_PERC_SUM', 'BURO_AMT_CREDIT_SUM_LIMIT_MAX', 'INSTAL_PAYMENT_PERC_VAR',
                 # 'REFUSED_APP_CREDIT_PERC_MEAN', 'INSTAL_DPD_MAX', 'BURO_CREDIT_ACTIVE_Closed_MEAN', 'INSTAL_DPD_SUM',
                 # 'BURO_AMT_CREDIT_SUM_DEBT_MIN', 'BURO_AMT_CREDIT_MAX_OVERDUE_SUM', 'PREV_AMT_CREDIT_MEAN',
                 # 'YEARS_BUILD_MEDI',
                 # 'BURO_DAYS_CREDIT_UPDATE_MIN', 'INSTAL_PAYMENT_DIFF_VAR', 'APPROVED_AMT_APPLICATION_SUM',
                 # 'APPROVED_AMT_CREDIT_MEAN',
                 # 'PREV_NAME_YIELD_GROUP_low_normal_MEAN', 'COMMONAREA_MEDI', 'CC_AMT_PAYMENT_CURRENT_MEAN',
                 # 'REFUSED_AMT_ANNUITY_MEAN',
                 # 'BURO_BB_STATUS_C_MEAN_MEAN', 'BURO_DAYS_CREDIT_UPDATE_SUM', 'NONLIVINGAREA_MEDI',
                 # 'APPROVED_AMT_CREDIT_SUM',
                 # 'NAME_EDUCATION_TYPE', 'CC_CNT_DRAWINGS_CURRENT_VAR', 'REGION_RATING_CLIENT',
                 # 'APPROVED_AMT_APPLICATION_MEAN',
                 # 'REFUSED_AMT_APPLICATION_MEAN', 'BURO_CREDIT_ACTIVE_Active_MEAN', 'POS_SK_DPD_MEAN',
                 # 'PREV_CNT_PAYMENT_SUM',
                 # 'AMT_CREDIT_y', 'PREV_CHANNEL_TYPE_Credit and cash offices_MEAN', 'INSTAL_COUNT',
                 # 'PREV_AMT_APPLICATION_MEAN',
                 # 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN', 'AMT_APPLICATION', 'BURO_BB_STATUS_0_SUM_SUM',
                 # 'BURO_CREDIT_TYPE_Creditcard_MEAN',
                 # 'BURO_AMT_ANNUITY_MAX', 'PREV_NAME_YIELD_GROUP_middle_MEAN', 'CC_CNT_DRAWINGS_ATM_CURRENT_VAR',
                 # 'POS_COUNT',
                 # 'AMT_DOWN_PAYMENT', 'PREV_AMT_ANNUITY_SUM', 'BURO_CREDIT_TYPE_Mortgage_MEAN', 'REFUSED_AMT_CREDIT_SUM',
                 # 'BURO_BB_STATUS_X_MEAN_MEAN', 'REFUSED_AMT_CREDIT_MEAN', 'REFUSED_AMT_APPLICATION_SUM',
                 # 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN', 'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',
                 # 'BURO_BB_STATUS_1_MEAN_MEAN',
                 # 'PREV_NAME_CLIENT_TYPE_Refreshed_MEAN', 'REFUSED_CNT_PAYMENT_MEAN', 'BURO_AMT_CREDIT_MAX_OVERDUE_MIN',
                 # 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN', 'CNT_PAYMENT', 'REFUSED_AMT_ANNUITY_SUM',
                 # 'CC_CNT_DRAWINGS_CURRENT_MEAN',
                 # 'PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN', 'BURO_AMT_ANNUITY_MIN',
                 # 'BURO_DAYS_ENDDATE_FACT_SUM',
                 # 'REFUSED_APP_CREDIT_PERC_VAR', 'PREV_NAME_YIELD_GROUP_XNA_MEAN', 'CREDIT_BUREAU',
                 # 'BURO_AMT_CREDIT_SUM_LIMIT_SUM',
                 # 'PREV_NAME_PRODUCT_TYPE_x-sell_MEAN', 'PREV_AMT_APPLICATION_SUM', 'PREV_NAME_CLIENT_TYPE_New_MEAN',
                 # 'PREV_CHANNEL_TYPE_Country-wide_MEAN', 'BURO_CREDIT_TYPE_Consumercredit_MEAN',
                 # 'CC_AMT_PAYMENT_CURRENT_SUM',
                 # 'BURO_CREDIT_TYPE_Microloan_MEAN', 'CC_AMT_PAYMENT_CURRENT_VAR', 'NAME_FAMILY_STATUS_Single',
                 # 'PREV_AMT_CREDIT_SUM',
                 # 'POS_SK_DPD_DEF_MAX', 'PREV_NAME_YIELD_GROUP_low_action_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_SUM',
                 # 'OBS_30_CNT_SOCIAL_CIRCLE', 'PREV_NAME_PORTFOLIO_Cash_MEAN', 'RATIO_DEF_60_to_DEF_30',
                 # 'PREV_NAME_PAYMENT_TYPE_XNA_MEAN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN', 'BURO_BB_STATUS_C_SUM_SUM',
                 # 'REFUSED_CNT_PAYMENT_SUM', 'PREV_CHANNEL_TYPE_Stone_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_VAR',
                 # 'PREV_NAME_PRODUCT_TYPE_XNA_MEAN', 'BURO_CREDIT_TYPE_Carloan_MEAN', 'CC_AMT_PAYMENT_CURRENT_MIN',
                 # 'CC_AMT_DRAWINGS_ATM_CURRENT_MEAN', 'PREV_NAME_PORTFOLIO_POS_MEAN', 'PREV_CODE_REJECT_REASON_XAP_MEAN',
                 # 'CC_AMT_DRAWINGS_ATM_CURRENT_SUM', 'FLAG_OWN_CAR', 'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
                 # 'PREV_NAME_CLIENT_TYPE_Repeater_MEAN', 'OBS_60_CNT_SOCIAL_CIRCLE',
                 # 'POS_NAME_CONTRACT_STATUS_Signed_MEAN',
                 # 'PREV_NAME_CONTRACT_STATUS_Approved_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_VAR',
                 # 'PREV_CHANNEL_TYPE_Contact center_MEAN',
                 # 'REFUSED_AMT_DOWN_PAYMENT_MEAN', 'CC_AMT_PAYMENT_CURRENT_MAX', 'CC_AMT_DRAWINGS_ATM_CURRENT_MAX',
                 # 'CC_CNT_DRAWINGS_CURRENT_MAX', 'BURO_BB_STATUS_X_SUM_SUM', 'PREV_DOWN_PAYMENT_STATUS_MEAN',
                 # 'NONLIVINGAPARTMENTS_MEDI',
                 # 'CC_AMT_INST_MIN_REGULARITY_VAR', 'PREV_NAME_PORTFOLIO_Cards_MEAN',
                 # 'BURO_AMT_ANNUITY_SUM',
                 # 'APPROVED_DOWN_PAYMENT_STATUS_MEAN', 'OCCUPATION_TYPE_White_collar',
                 # 'CC_AMT_PAYMENT_TOTAL_CURRENT_SUM',
                 # 'CC_AMT_DRAWINGS_POS_CURRENT_VAR', 'NAME_INCOME_TYPE_State servant',
                 # 'FLAG_DOCUMENT',
                 # 'PREV_NAME_PORTFOLIO_XNA_MEAN',
                 # 'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN', 'CC_AMT_DRAWINGS_POS_CURRENT_SUM',
                 # 'CC_AMT_DRAWINGS_CURRENT_MEAN',
                 # 'BURO_AMT_CREDIT_SUM_OVERDUE_MAX', 'CC_CNT_DRAWINGS_POS_CURRENT_MEAN', 'CC_AMT_BALANCE_MEAN',
                 # bundan oncesi 217 column
                 'RATIO_OBS_60_to_OBS_30',
                 'DEF_30_CNT_SOCIAL_CIRCLE', 'NAME_INCOME_TYPE_Working', 'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN',
                 'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN', 'DEF_60_CNT_SOCIAL_CIRCLE', 'CC_AMT_BALANCE_VAR',
                 'PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN', 'CC_SK_DPD_DEF_MEAN', 'CC_CNT_DRAWINGS_CURRENT_SUM',
                 'POS_SK_DPD_MAX',
                 'CC_AMT_DRAWINGS_CURRENT_VAR', 'CC_AMT_DRAWINGS_POS_CURRENT_MAX', 'CC_CNT_DRAWINGS_POS_CURRENT_VAR',
                 'CC_AMT_RECIVABLE_MEAN', 'OCCUPATION_TYPE_Drivers', 'CC_AMT_DRAWINGS_CURRENT_SUM',
                 'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX',
                 'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR', 'CC_AMT_RECIVABLE_VAR', 'CC_CNT_INSTALMENT_MATURE_CUM_VAR',
                 'CHANNEL_TYPE_Channel of corporate sales', 'CC_AMT_RECIVABLE_MIN', 'OCCUPATION_TYPE_nan',
                 'CC_CNT_DRAWINGS_ATM_CURRENT_SUM', 'CC_CNT_DRAWINGS_POS_CURRENT_MAX', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
                 'CNT_CHILDREN',
                 'POS_NAME_CONTRACT_STATUS_Returnedtothestore_MEAN', 'CC_AMT_DRAWINGS_POS_CURRENT_MEAN',
                 'PREV_CHANNEL_TYPE_Regional / Local_MEAN', 'NAME_YIELD_GROUP_low_action', 'CC_AMT_BALANCE_MAX',
                 'NAME_YIELD_GROUP_high', 'BURO_BB_STATUS_1_SUM_SUM', 'CC_AMT_INST_MIN_REGULARITY_SUM',
                 'CC_CNT_INSTALMENT_MATURE_CUM_MEAN', 'BURO_CREDIT_ACTIVE_Sold_MEAN', 'CC_MONTHS_BALANCE_SUM',
                 'CC_CNT_DRAWINGS_POS_CURRENT_SUM', 'CC_AMT_DRAWINGS_CURRENT_MAX', 'INSTAL_PAYMENT_PERC_MAX',
                 'CC_AMT_RECIVABLE_MAX',
                 'CC_MONTHS_BALANCE_MEAN', 'CC_CNT_INSTALMENT_MATURE_CUM_SUM', 'CC_AMT_RECEIVABLE_PRINCIPAL_VAR',
                 'CC_MONTHS_BALANCE_VAR', 'NAME_CONTRACT_TYPE_Revolving loans_x', 'BURO_AMT_CREDIT_SUM_OVERDUE_SUM',
                 'CC_AMT_BALANCE_MIN', 'CC_AMT_TOTAL_RECEIVABLE_MEAN', 'CC_AMT_RECEIVABLE_PRINCIPAL_MAX',
                 'BURO_BB_STATUS_2_MEAN_MEAN',
                 'OCCUPATION_TYPE_Laborers', 'CC_AMT_DRAWINGS_CURRENT_MIN', 'BURO_AMT_CREDIT_SUM_LIMIT_MIN',
                 'CC_AMT_INST_MIN_REGULARITY_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX', 'CC_AMT_RECEIVABLE_PRINCIPAL_SUM',
                 'CC_AMT_INST_MIN_REGULARITY_MAX', 'CC_AMT_TOTAL_RECEIVABLE_MIN', 'CC_AMT_RECIVABLE_SUM',
                 'CC_AMT_TOTAL_RECEIVABLE_VAR',
                 'CC_AMT_BALANCE_SUM', 'NAME_HOUSING_TYPE_Other', 'NAME_CONTRACT_STATUS_Refused',
                 'CC_CNT_DRAWINGS_ATM_CURRENT_MAX',
                 'DAYS_FIRST_DRAWING', 'APPROVED_DOWN_PAYMENT_STATUS_SUM', 'CC_AMT_TOTAL_RECEIVABLE_MAX',
                 'CC_NAME_CONTRACT_STATUS_Active_SUM', 'CC_MONTHS_BALANCE_MIN', 'CODE_REJECT_REASON_SCOFR',
                 'CC_AMT_RECEIVABLE_PRINCIPAL_MIN', 'CC_CNT_DRAWINGS_POS_CURRENT_MIN', 'BURO_BB_STATUS_2_SUM_SUM',
                 'BURO_CREDIT_TYPE_Anothertypeofloan_MEAN', 'CC_NAME_CONTRACT_STATUS_Active_MEAN',
                 'NAME_YIELD_GROUP_low_normal',
                 'CC_CNT_INSTALMENT_MATURE_CUM_MAX', 'REFUSED_DOWN_PAYMENT_STATUS_MEAN', 'BURO_CREDIT_DAY_OVERDUE_MAX',
                 'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_NAME_CONTRACT_STATUS_Active_VAR', 'CC_SK_DPD_SUM',
                 'DAYS_LAST_PHONE_CHANGE_Binary', 'CC_COUNT', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX',
                 'CC_AMT_TOTAL_RECEIVABLE_SUM',
                 'CC_AMT_DRAWINGS_POS_CURRENT_MIN', 'NFLAG_INSURED_ON_APPROVAL', 'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM',
                 'CC_SK_DPD_VAR',
                 'FLAG_OWN_REALTY', 'PREV_CODE_REJECT_REASON_CLIENT_MEAN', 'PREV_DOWN_PAYMENT_STATUS_SUM',
                 'CHANNEL_TYPE_AP+ (Cash loan)', 'CC_SK_DPD_MEAN', 'BURO_BB_STATUS_4_SUM_SUM', 'CC_SK_DPD_DEF_VAR',
                 'CC_NAME_CONTRACT_STATUS_Completed_MEAN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN',
                 'CC_NAME_CONTRACT_STATUS_Signed_VAR',
                 'BURO_BB_STATUS_4_MEAN_MEAN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_VAR',
                 'CC_NAME_CONTRACT_STATUS_Completed_VAR',
                 'Personal_Assets', 'CC_SK_DPD_MAX', 'BURO_BB_STATUS_3_MEAN_MEAN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN',
                 'CC_MONTHS_BALANCE_MAX', 'NAME_PAYMENT_TYPE_Cash through the bank', 'CHANNEL_TYPE_Regional / Local',
                 'BURO_CREDIT_DAY_OVERDUE_SUM', 'CODE_REJECT_REASON_XAP', 'CODE_REJECT_REASON_HC',
                 'NAME_PAYMENT_TYPE_XNA',
                 'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN', 'NAME_CONTRACT_STATUS_Canceled',
                 'REFUSED_DOWN_PAYMENT_STATUS_SUM', 'POS_NAME_CONTRACT_STATUS_Approved_MEAN',
                 'BURO_BB_STATUS_5_MEAN_MEAN',
                 'CC_NAME_CONTRACT_STATUS_Completed_SUM', 'BURO_CNT_CREDIT_PROLONG_SUM', 'CC_CNT_DRAWINGS_CURRENT_MIN',
                 'CC_AMT_INST_MIN_REGULARITY_MIN', 'CREDIT_BUREAU_Binary', 'CC_SK_DPD_DEF_SUM',
                 'CC_CNT_INSTALMENT_MATURE_CUM_MIN',
                 'NAME_CONTRACT_TYPE_Consumer loans', 'BURO_CREDIT_TYPE_Loanforbusinessdevelopment_MEAN',
                 'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN', 'CC_NAME_CONTRACT_STATUS_Signed_MEAN', 'NAME_YIELD_GROUP_middle',
                 'PREV_RATE_INTEREST_PRIMARY_MEAN', 'CHANNEL_TYPE_Country-wide', 'BURO_CNT_CREDIT_PROLONG_MAX',
                 'CC_NAME_CONTRACT_STATUS_Active_MIN', 'NAME_CONTRACT_TYPE_Cash loans', 'NAME_PRODUCT_TYPE_walk-in',
                 'CHANNEL_TYPE_Credit and cash offices', 'CODE_REJECT_REASON_SCO', 'NAME_INCOME_TYPE_Pensioner',
                 'NAME_CONTRACT_TYPE_Revolving loans_y', 'CHANNEL_TYPE_Contact center',
                 'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
                 'NAME_PRODUCT_TYPE_XNA', 'APPROVED_RATE_INTEREST_PRIMARY_MEAN', 'NAME_PRODUCT_TYPE_x-sell',
                 'CHANNEL_TYPE_Stone',
                 'RATE_INTEREST_PRIMARY', 'NAME_PORTFOLIO_POS', 'NAME_CLIENT_TYPE_New', 'NAME_CLIENT_TYPE_Repeater',
                 'NAME_PORTFOLIO_XNA', 'NAME_PAYMENT_TYPE_Non-cash from your account',
                 'CC_NAME_CONTRACT_STATUS_Completed_MAX',
                 'NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Unused offer', 'OCCUPATION_TYPE_Pensioner',
                 'NAME_CLIENT_TYPE_Refreshed', 'NAME_YIELD_GROUP_XNA', 'CC_SK_DPD_DEF_MAX', 'NAME_PORTFOLIO_Cards',
                 'BURO_BB_STATUS_5_SUM_SUM', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX', 'BURO_BB_STATUS_3_SUM_SUM',
                 'CODE_REJECT_REASON_CLIENT', 'CC_NAME_CONTRACT_STATUS_Signed_MAX',
                 'BURO_CREDIT_CURRENCY_currency1_MEAN',
                 'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM', 'NAME_PORTFOLIO_Cash', 'CC_NAME_CONTRACT_STATUS_Demand_MAX',
                 'CC_NAME_CONTRACT_STATUS_Demand_SUM', 'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
                 'BURO_CREDIT_TYPE_Loanforworkingcapitalreplenishment_MEAN',
                 'BURO_CREDIT_TYPE_Cashloannonearmarked_MEAN',
                 'PREV_NAME_CLIENT_TYPE_nan_MEAN',
                 'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
                 'PREV_NAME_CONTRACT_STATUS_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_Completed_MIN',
                 'PREV_NAME_CONTRACT_TYPE_nan_MEAN',
                 'CC_NAME_CONTRACT_STATUS_Demand_MIN', 'CC_NAME_CONTRACT_STATUS_Approved_VAR',
                 'CC_NAME_CONTRACT_STATUS_Demand_MEAN',
                 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
                 'CC_NAME_CONTRACT_STATUS_Approved_SUM',
                 'CC_NAME_CONTRACT_STATUS_Approved_MIN', 'BURO_CNT_CREDIT_PROLONG_MIN',
                 'BURO_CREDIT_TYPE_Interbankcredit_MEAN',
                 'Prev_appl_existornot', 'BURO_CREDIT_CURRENCY_currency2_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
                 'CC_NAME_CONTRACT_STATUS_Demand_VAR', 'BURO_CREDIT_TYPE_Loanforpurchaseofsharesmarginlending_MEAN',
                 'PREV_NAME_YIELD_GROUP_nan_MEAN', 'CC_CNT_DRAWINGS_ATM_CURRENT_MIN',
                 'BURO_CREDIT_TYPE_Loanforthepurchaseofequipment_MEAN', 'CC_NAME_CONTRACT_STATUS_Active_MAX',
                 'BURO_CREDIT_CURRENCY_currency3_MEAN', 'BURO_CREDIT_CURRENCY_currency4_MEAN',
                 'CC_NAME_CONTRACT_STATUS_Approved_MAX',
                 'PREV_NAME_PRODUCT_TYPE_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
                 'PREV_NAME_PORTFOLIO_nan_MEAN',
                 'REFUSED_RATE_INTEREST_PRIMARY_MEAN', 'NAME_CONTRACT_TYPE_nan_y',
                 'CC_NAME_CONTRACT_STATUS_Refused_MAX',
                 'NAME_PRODUCT_TYPE_nan', 'POS_NAME_CONTRACT_STATUS_Amortizeddebt_MEAN', 'CHANNEL_TYPE_Car dealer',
                 'BURO_AMT_CREDIT_SUM_OVERDUE_MIN', 'CHANNEL_TYPE_nan', 'CODE_GENDER_nan',
                 'BURO_CREDIT_TYPE_Realestateloan_MEAN',
                 'CODE_REJECT_REASON_LIMIT', 'BURO_CREDIT_DAY_OVERDUE_MIN', 'NAME_YIELD_GROUP_nan',
                 'CODE_REJECT_REASON_SYSTEM',
                 'CODE_REJECT_REASON_VERIF', 'NAME_PORTFOLIO_nan', 'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
                 'NAME_PORTFOLIO_Cars',
                 'NAME_PAYMENT_TYPE_nan', 'NAME_CONTRACT_TYPE_XNA',
                 'NAME_PAYMENT_TYPE_Cashless from the account of the employer',
                 'NAME_CONTRACT_TYPE_nan_x', 'NAME_INCOME_TYPE_nan', 'CODE_REJECT_REASON_XNA', 'CODE_REJECT_REASON_nan',
                 'NAME_HOUSING_TYPE_nan', 'NAME_FAMILY_STATUS_nan', 'Bureau_balance_existornot',
                 'NAME_CONTRACT_STATUS_nan',
                 'POS_NAME_CONTRACT_STATUS_Demand_MEAN', 'CC_NAME_CONTRACT_STATUS_Refused_MEAN',
                 'BURO_CREDIT_TYPE_Unknowntypeofloan_MEAN', 'CC_NAME_CONTRACT_STATUS_Refused_MIN',
                 'PREV_CODE_REJECT_REASON_nan_MEAN',
                 'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'CC_NAME_CONTRACT_STATUS_Refused_VAR', 'NAME_CLIENT_TYPE_XNA',
                 'CC_NAME_CONTRACT_STATUS_Sentproposal_MAX', 'CC_NAME_CONTRACT_STATUS_Sentproposal_MEAN',
                 'PREV_CHANNEL_TYPE_nan_MEAN',
                 'CC_NAME_CONTRACT_STATUS_Sentproposal_MIN', 'CC_NAME_CONTRACT_STATUS_Sentproposal_SUM',
                 'CC_NAME_CONTRACT_STATUS_Sentproposal_VAR', 'CC_NAME_CONTRACT_STATUS_Signed_MIN', 'CC_SK_DPD_MIN',
                 'CC_NAME_CONTRACT_STATUS_Signed_SUM', 'NAME_CLIENT_TYPE_nan', 'CC_NAME_CONTRACT_STATUS_nan_MAX',
                 'CC_NAME_CONTRACT_STATUS_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_nan_MIN',
                 'CC_NAME_CONTRACT_STATUS_nan_SUM',
                 'CC_NAME_CONTRACT_STATUS_nan_VAR', 'CC_SK_DPD_DEF_MIN', 'BURO_CREDIT_TYPE_Mobileoperatorloan_MEAN',
                 'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'POS_NAME_CONTRACT_STATUS_XNA_MEAN',
                 'BURO_CREDIT_ACTIVE_Baddebt_MEAN']

    df.drop(drop_list, axis=1, inplace=True)

    return df



def Join_aggregated_lastprevappl(df1, df2):
    df1['SK_ID_CURR'] = df1.index
    Joined_df = df2.join(df1, on='SK_ID_CURR', how='left', lsuffix="_LAST_APPL").reset_index()
    Joined_df.drop(['index', 'SK_ID_CURR_LAST_APPL'], axis=1, inplace=True)

    return Joined_df


def previous_applications(num_rows=None):
    prev = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\Data\previous_application.csv', nrows=num_rows)
    prev = Drop_columns(prev)
    # Convert NAME_CONTRACT_STATUS
    prev['NAME_CONTRACT_STATUS'] = prev['NAME_CONTRACT_STATUS'].replace('Canceled', 'Refused')
    prev['NAME_CONTRACT_STATUS'] = prev['NAME_CONTRACT_STATUS'].replace('Unused offer', 'Approved')
    # Operation after Rare Analyzer
    prev.loc[(prev['NAME_CONTRACT_TYPE'] == 'XNA'), 'NAME_CONTRACT_TYPE'] = 'Cash loans'
    prev.loc[(prev['CODE_REJECT_REASON'].apply(
        lambda x: x in ['XNA', 'LIMIT', 'SCO', 'SCOFR', 'SYSTEM', 'VERIF'])), 'CODE_REJECT_REASON'] = 'HC'
    prev.loc[(prev['NAME_PORTFOLIO'] == 'Cars'), 'NAME_PORTFOLIO'] = 'Cards'
    prev.loc[(prev['CHANNEL_TYPE'].apply(
        lambda x: x in ['Car dealer', 'Channel of corporate sales'])), 'CHANNEL_TYPE'] = 'Country-wide'

    # Prepare Last Previous Applications
    df_last_prev_appl = Last_prev_prepare(num_rows)
    # Drop columns for Min data
    df_last_prev_appl = Drop_columns(df_last_prev_appl)
    prev, cat_cols = one_hot_encoderr(prev, nan_as_category=True)
    df_last_prev_appl, cat_cols2 = one_hot_encoderr(df_last_prev_appl, nan_as_category=True)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    # prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # ADD FEATURES
    # Down payment status
    prev['DOWN_PAYMENT_STATUS'] = np.where(prev['AMT_DOWN_PAYMENT'] > 0, 1, 0)
    # Value asked / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['sum', 'mean'],
        'AMT_APPLICATION': ['mean', 'sum'],
        'AMT_CREDIT': ['mean', 'sum'],
        'APP_CREDIT_PERC': ['mean', 'var'],
        'DOWN_PAYMENT_STATUS': ['sum', 'mean'],
        'AMT_DOWN_PAYMENT': ['mean'],
        'DAYS_DECISION': ['mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'RATE_INTEREST_PRIMARY': 'mean'}

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev

    Joined_df = Join_aggregated_lastprevappl(prev_agg, df_last_prev_appl)

    Joined_df['Prev_appl_existornot'] = 1

    return Joined_df

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\Data\POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\Data\installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\Data\credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def UnderSampling(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    X_80, X_20 = train_test_split(train_df, test_size=0.2, random_state=123)  # Splitting 20 percent of data for prediction

    # Shuffling data
    shuffled_df = X_80.sample(frac=1, random_state=4)

    # Create a seperate dataframe for customers subscribed term project
    yes_df = shuffled_df.loc[shuffled_df['TARGET'] == 1]

    # Pick random samples (specified count) from majority class
    no_df = shuffled_df.loc[shuffled_df['TARGET'] == 0].sample(n=int(df.TARGET.value_counts()[1] * 0.75),
                                                               random_state=123)

    # Concat datasets of two classes
    balanced_df = pd.concat([yes_df, no_df])

    print((balanced_df.TARGET.value_counts() / balanced_df.TARGET.count()))

    plt.bar(['Approved', 'Not Approved'], balanced_df.TARGET.value_counts().values, facecolor='brown',
            edgecolor='brown', linewidth=0.5,
            ls='dashed')
    sns.set(font_scale=1)
    plt.title('Target Variable', fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Amount')
    plt.show()

    df = balanced_df.append(test_df)

    return df


def OverSampling(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    X = train_df.drop('TARGET', axis=1)
    y = train_df.TARGET
    os = RandomOverSampler(sampling_strategy='minority')

    X_new, y_new = os.fit_resample(X, y)

    train_df_new = X_new.join(y_new.to_frame())

    df = train_df_new.append(test_df)

    return df

def OverSampling_SMOTE(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df_X = train_df.drop('TARGET', axis=1)
    train_df_y = train_df.TARGET

    # SMOTE
    print('Creating Smote Data...')
    smote = SMOTE(k_neighbors=5, n_jobs=-1)
    smote_enn = make_pipeline(SimpleImputer(), SMOTEENN(smote=smote))
    X_res, y_res = smote_enn.fit_resample(train_df_X, train_df_y)

    X_res_df = pd.DataFrame(X_res, columns=train_df_X.columns)

    train_df_new = X_res_df.join(y_res.to_frame())

    df = train_df_new.append(test_df)

    # Save data to csv file
    df.to_csv('data/df_prepared_to_model.csv')

    # Save data to pickle file
    df.to_pickle("data/df_prepared_to_model.pkl")

    return df
def OverSampling_SMOTE_v2(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df_X = train_df.drop('TARGET', axis=1)
    train_df_y = train_df.TARGET

    train_df_X.replace(np.nan, -1, inplace=True)

    print('Creating Smote Data...')
    # sm = ADASYN()
    sm = SMOTE()
    X, y = sm.fit_resample(train_df_X, train_df_y)
    train_df_new = X.join(y.to_frame())
    df = train_df_new.append(test_df)

    return df

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    df = Drop_columns_before_modelling(df)
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    # print(oof_preds.shape)
    sub_preds = np.zeros(test_df.shape[0])
    # print(sub_preds.shape)
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        # print(train_x.shape)
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=34,0
            subsample=0.8715623,
            max_depth=6,
            reg_alpha=0.05,
            reg_lambda=0.07,
            min_split_gain=0.1,
            min_child_weight=40,
            silent=-1,
            verbose=-1, )000000000

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    df_test = feature_importance_df.groupby('feature').agg({'importance':'sum'}).sort_values(by='importance', ascending=False)
    lst = df_test.iloc[60:, :].index.to_list()
    print(lst)
    # print(len(lst))

    feature_importance_df.to_pickle('outputs/features/feature_importance_df.pkl')

    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances01.png')

def main(debug=False):
    num_rows = 10000 if debug else None
    # GET DATA FROM PICKLE FILE
    # df = pd.read_pickle(r'E:\PROJECTS\Home_Credit_new\data\df_prepared_to_model.pkl')

    # # GET DATA FROM CSV
    # df = pd.read_csv(r'E:\PROJECTS\Home_Credit_new\data\df_prepared_to_model.csv', nrows=num_rows)

    df = application_train_test(num_rows)
    df = Create_New_Ext_Source_Features(df)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        df = df.reset_index(drop=True)
        df.loc[(df['Bureau_balance_existornot'].isnull()), 'Bureau_balance_existornot'] = 0
        update_indices1 = df[df.Bureau_balance_existornot == 0].index
        df.iloc[update_indices1, 69:-1] = -1
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.merge(prev, how='left', on='SK_ID_CURR')
        df.loc[(df['Prev_appl_existornot'].isnull()), 'Prev_appl_existornot'] = 0
        update_indices = df[df.Prev_appl_existornot == 0].index
        df.iloc[update_indices, 144:-1] = -1
        del prev
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    # with timer("Run LightGBM OverSampling SMOTE"):
    #     df = OverSampling_SMOTE_v2(df)
    # with timer("Run LightGBM UnderSampling"):
    #     df = UnderSampling(df)
    # with timer("Run LightGBM OverSampling"):
    #     df = OverSampling(df)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()

# df = application_train_test(None)
# df = Create_New_Ext_Source_Features(df)
# bureau = bureau_and_balance(None)
# df = df.join(bureau, how='left', on='SK_ID_CURR')
# df = df.reset_index(drop=True)
# df.loc[(df['Bureau_balance_existornot'].isnull()), 'Bureau_balance_existornot'] = 0
# update_indices1 = df[df.Bureau_balance_existornot == 0].index
# df.iloc[update_indices1, 69:-1] = -1
# prev = previous_applications(None)
# df = df.merge(prev, how='left', on='SK_ID_CURR')
# df.loc[(df['Prev_appl_existornot'].isnull()), 'Prev_appl_existornot'] = 0
# update_indices = df[df.Prev_appl_existornot == 0].index
# df.iloc[update_indices, 144:-1] = -1
# pos = pos_cash(None)
# df = df.join(pos, how='left', on='SK_ID_CURR')
# ins = installments_payments(None)
# df = df.join(ins, how='left', on='SK_ID_CURR')
# cc = credit_card_balance(None)
# df = df.join(cc, how='left', on='SK_ID_CURR')



# df = OverSampling_SMOTE(df)

# df = pd.read_csv(r'xxx.csv')
# df.to_csv('xxx.csv')
# df.to_excel('xxx')
#
# df.columns
#
# df.head()
# df.index
# df.drop('SK_ID_CURR;TARGET', axis=1, inplace=True)
# df['SK_ID_CURR'] = df['SK_ID_CURR;TARGET'].apply(lambda x:  x.split(';')[0])
# df['TARGET'] = df['SK_ID_CURR;TARGET'].apply(lambda x:  x.split(';')[1])

# Fold 1 - 1st Valid's auc's
# 487 columns - valid_1's auc: 0.780799
# 406 columns - valid_1's auc: 0.779411
# 324 columns - valid_1's auc: 0.780142
# 245 COLUMNS - valid_1's auc: 0.77977
# 224 columns - valid_1's auc: 0.780529
# 220 columns - valid_1's auc: 0.779357
# 217 columns - valid_1's auc: 0.780664 - 0.790763
# 216 columns - valid_1's auc: 0.779838
# 214 columns - valid_1's auc: 0.779839
# 210 columns - valid_1's auc: 0.779928
# 197 columns - valid_1's auc: 0.779854
# 149 columns - valid_1's auc: 0.779464
# 132 columns - valid_1's auc: 0.779376
# 108 columns - valid_1's auc: 0.776626
# 66  columns - valid_1's auc: 0.77116