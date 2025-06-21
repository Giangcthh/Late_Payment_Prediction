import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from collections import Counter
import joblib
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = './data/' 
MODEL_SAVE_PATH = './saved_model/'
CUSTOMER_CSV = DATA_PATH + 'customer.csv'
LOAN_CSV = DATA_PATH + 'loan.csv'
STATE_REGION_CSV = DATA_PATH + 'state_region.csv'
JOB_MAPPING_XLSX = DATA_PATH + 'job_mapping.xlsx' 

SAVED_PREPROCESSOR_PATH = MODEL_SAVE_PATH + 'preprocessor.joblib'
SAVED_MODEL_PATH = MODEL_SAVE_PATH + 'model.joblib'
SAVED_PARAMS_PATH = MODEL_SAVE_PATH + 'feature_engineering_params.json'

# Fixed date for 'loan_age_days' calculation, consistent with notebook
CURRENT_DATE_FOR_AGE_CALC = datetime(year=2019, month=12, day=31)


# --- Helper Functions from Notebook (Data Cleaning & Feature Engineering) ---
def clean_emp_length(val):
    if pd.isna(val):
        return np.nan
    if val == '< 1 year':
        return 0
    if val == '10+ years':
        return 10
    return int(val.split()[0])

def clean_type(val):
    if pd.isna(val):
        return 'UNKNOWN'
    val = val.strip().upper()
    if val in ['INDIVIDUAL']:
        return 'INDIVIDUAL'
    elif val in ['JOINT', 'JOINT APP']:
        return 'JOINT'
    elif val == 'DIRECT_PAY':
        return 'DIRECT_PAY'
    else:
        return 'OTHER'

def map_loan_status(status):
    bad_statuses = ['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)']
    good_statuses = ['Fully Paid', 'Current', 'In Grace Period']
    if status in bad_statuses:
        return 1
    elif status in good_statuses:
        return 0
    else:
        return np.nan

def engineer_features(df, params=None, is_training=True):
    """
    Applies all feature engineering steps.
    If is_training=True, it calculates and returns params.
    If is_training=False, it uses provided params.
    """
    engineered_df = df.copy()
    
    # Date transformations
    engineered_df['issue_d'] = pd.to_datetime(engineered_df['issue_d'], format='%b-%y', errors='coerce')
    engineered_df['issue_month'] = engineered_df['issue_d'].dt.month
    engineered_df['issue_quarter'] = engineered_df['issue_d'].dt.quarter
    engineered_df['issue_year_num'] = engineered_df['issue_d'].dt.year

    term_num_map = {' 36 months': 36, ' 60 months': 60}
    engineered_df['term_num'] = engineered_df['term'].map(term_num_map)
    
    # Calculate maturity date: issue_d + term_num months
    # Approximate months as 30 days for simplicity
    engineered_df['maturity_date'] = engineered_df.apply(
        lambda row: row['issue_d'] + pd.DateOffset(months=row['term_num']) if pd.notnull(row['issue_d']) and pd.notnull(row['term_num']) else pd.NaT,
        axis=1
    )
    
    engineered_df['loan_age_days'] = (CURRENT_DATE_FOR_AGE_CALC - engineered_df['issue_d']).dt.days
    engineered_df['is_maturity'] = np.where(engineered_df['maturity_date'] < CURRENT_DATE_FOR_AGE_CALC, 'yes', 'no')
    
    # Clean emp_length
    engineered_df['emp_length'] = engineered_df['emp_length'].apply(clean_emp_length)
    
    # Clean type
    engineered_df['type'] = engineered_df['type'].apply(clean_type) 

    # Total income
    engineered_df['total_inc'] = np.where(
        engineered_df['annual_inc_joint'].isna(),
        engineered_df['annual_inc'],
        engineered_df['annual_inc'] + engineered_df['annual_inc_joint']
    )
    engineered_df['have_inc_joint'] = engineered_df['annual_inc_joint'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')

    # Mappings (ensure these are consistent with notebook)
    engineered_df['home_ownership'] = engineered_df['home_ownership'].replace(['NONE', 'ANY', 'OTHER'], 'OTHERS')
    
    profession_mapping = {
        'Admin/Assistant/Support/Services': 'Corporate Admin & Support',
        'Agent/Legal/Insuarance': 'Business, Finance & HR',
        'Civil Servant': 'Public Service & Education',
        'Clerk': 'Corporate Admin & Support',
        'Consultant': 'Business, Finance & HR',
        'Convenience Services': 'Sales, Service & F&B',
        'Coordinator': 'Corporate Admin & Support',
        'Counselor/Therapist': 'Public Service & Education',
        'Educator/Teaching': 'Public Service & Education',
        'Financial/Accounting/Analyst': 'Business, Finance & HR',
        'Food and Beverage': 'Sales, Service & F&B',
        'Healthcare/Medical': 'Healthcare',
        'Human Resources': 'Business, Finance & HR',
        'IT/Technician/Engineer': 'Tech, Engineering & Logistics',
        'Logistics/Delivery/Driver': 'Tech, Engineering & Logistics',
        'Management/Specialist/Supervisor': 'Leadership & Management',
        'Manufacture/Distributor': 'Production & Skilled Trade',
        'Mechanic/Maintenance': 'Tech, Engineering & Logistics',
        'No Information': 'No Information',
        'Operations': 'Corporate Admin & Support',
        'Others': 'Others',
        'Production/Assembler': 'Production & Skilled Trade',
        'Representative/Relations': 'Sales, Service & F&B',
        'Sales/Marketing': 'Sales, Service & F&B',
        'Security': 'Production & Skilled Trade',
        'Worker': 'Production & Skilled Trade'
}
    engineered_df['profession'] = engineered_df['profession'].map(profession_mapping).fillna('Others') 

    engineered_df['type'] = engineered_df['type'].replace(['DIRECT_PAY', 'JOINT'], 'NON_INDIVIDUAL') 

    purpose_mapping = {
        'debt_consolidation': 'debt_consolidation', 'credit_card': 'credit_card', 'home_improvement': 'home_related',
        'house': 'home_related', 'medical': 'health', 'car': 'auto', 'wedding': 'personal_lifestyle',
        'vacation': 'personal_lifestyle', 'moving': 'personal_lifestyle', 'major_purchase': 'major_purchase',
        'small_business': 'business', 'other': 'other', 'renewable_energy': 'other'
    }
    engineered_df['purpose'] = engineered_df['purpose'].map(purpose_mapping).fillna('other')

    # New numerical features
    # Ensure denominators are not zero
    safe_total_inc_monthly = np.where(engineered_df['total_inc'] > 0, engineered_df['total_inc'] / 12, np.nan)
    safe_total_inc = np.where(engineered_df['total_inc'] > 0, engineered_df['total_inc'], np.nan)
    safe_loan_amount = np.where(engineered_df['loan_amount'] > 0, engineered_df['loan_amount'], np.nan)
    safe_tot_cur_bal = np.where(engineered_df['Tot_cur_bal'] > 0, engineered_df['Tot_cur_bal'], 1) # Use 1 to avoid div by zero, impact should be minimal if 0 means no balance

    engineered_df['payment_to_income'] = engineered_df['installment'] / safe_total_inc_monthly
    engineered_df['lti'] = engineered_df['loan_amount'] / safe_total_inc
    engineered_df['interest_burden'] = engineered_df['int_rate'] * engineered_df['loan_amount'] / safe_total_inc
    engineered_df['bal_to_loan'] = engineered_df['avg_cur_bal'] / safe_loan_amount
    engineered_df['bal_to_income'] = engineered_df['avg_cur_bal'] / safe_total_inc
    
    local_params = {}
    if is_training:
        local_params['grade_avg_rate_map'] = engineered_df.groupby('grade')['int_rate'].mean().to_dict()
        local_params['grade_avg_loan_map'] = engineered_df.groupby('grade')['loan_amount'].mean().to_dict()
    else: # Use provided params for prediction
        if not params or 'grade_avg_rate_map' not in params or 'grade_avg_loan_map' not in params:
            raise ValueError("Missing grade_avg_rate_map or grade_avg_loan_map in params for prediction.")
        local_params = params

    engineered_df['grade_avg_rate'] = engineered_df['grade'].map(local_params['grade_avg_rate_map'])
    engineered_df['rate_premium'] = engineered_df['int_rate'] - engineered_df['grade_avg_rate']
    
    engineered_df['total_debt_ratio'] = engineered_df['Tot_cur_bal'] / safe_total_inc
    
    engineered_df['grade_avg_loan'] = engineered_df['grade'].map(local_params['grade_avg_loan_map'])
    engineered_df['relative_loan_size'] = engineered_df['loan_amount'] / np.where(engineered_df['grade_avg_loan'] > 0, engineered_df['grade_avg_loan'], np.nan)
    
    engineered_df['monthly_interest'] = (engineered_df['int_rate'] / 100 / 12) * engineered_df['loan_amount']
    engineered_df['interest_to_payment_ratio'] = engineered_df['monthly_interest'] / np.where(engineered_df['installment'] > 0, engineered_df['installment'], np.nan)
    engineered_df['loan_to_balance_ratio'] = engineered_df['loan_amount'] / safe_tot_cur_bal
    
    # Fill NaNs that might have been created by division by zero/NaN
    cols_to_fill_na = ['payment_to_income', 'lti', 'interest_burden', 'bal_to_loan', 'bal_to_income', 
                       'rate_premium', 'total_debt_ratio', 'relative_loan_size', 'monthly_interest', 
                       'interest_to_payment_ratio', 'loan_to_balance_ratio']
    for col in cols_to_fill_na:
        if col in engineered_df:
             # In training, calculate median, in testing use training median (passed via params)
            if is_training:
                median_val = engineered_df[col].median()
                local_params[f'{col}_median'] = median_val
                engineered_df[col].fillna(median_val, inplace=True)
            else:
                if params and f'{col}_median' in params:
                    engineered_df[col].fillna(params[f'{col}_median'], inplace=True)
                else: # Fallback if median not in params (should not happen with proper saving)
                    engineered_df[col].fillna(0, inplace=True) # Or raise error


    # Outlier Capping (must be done AFTER all features that are capped are created)
    cols_to_cap = ['avg_cur_bal', 'Tot_cur_bal', 'loan_amount', 'int_rate', 'installment', 'total_inc'] 
    if is_training:
        local_params['outlier_caps'] = {}
        for col in cols_to_cap:
            if col in engineered_df:
                Q1 = engineered_df[col].quantile(0.10)
                Q3 = engineered_df[col].quantile(0.90)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                local_params['outlier_caps'][col] = {'lower': lower_bound, 'upper': upper_bound, 'skew': engineered_df[col].skew()}
                
                # For simplicity, direct capping here.
                engineered_df[col] = engineered_df[col].clip(lower=lower_bound, upper=upper_bound)
    else: # Prediction
        if not params or 'outlier_caps' not in params:
            raise ValueError("Missing outlier_caps in params for prediction.")
        for col in cols_to_cap:
            if col in engineered_df and col in params['outlier_caps']:
                col_caps = params['outlier_caps'][col]
                engineered_df[col] = engineered_df[col].clip(lower=col_caps['lower'], upper=col_caps['upper'])
    
    # Drop columns 
    cols_to_drop_post_feature_eng = [
        'customer_id', 'loan_id', 'emp_title', 'issue_d', 'issue_date', 'issue_year', 
        'funded_amount', 'addr_state', 'notes', 'description', 'zip_code', 'state', 'region',
        'annual_inc', 'annual_inc_joint', 'term_num', 'maturity_date', 'grade_avg_rate', 'grade_avg_loan',
        'emp_title', 'type' 
    ]
    engineered_df.drop(columns=[col for col in cols_to_drop_post_feature_eng if col in engineered_df.columns], inplace=True, errors='ignore')


    if is_training:
        return engineered_df, local_params
    return engineered_df


# --- Main Training Logic ---
def train():
    print("Starting model training process...")

    # 1. Load Data
    print("Loading data...")
    customer_df = pd.read_csv(CUSTOMER_CSV)
    loan_df = pd.read_csv(LOAN_CSV)
    state_df = pd.read_csv(STATE_REGION_CSV)

    try:
        job_df = pd.read_excel(JOB_MAPPING_XLSX)
        # 'emp_title' in customer_df needs to be mapped to 'emp_job_mapping'
    except Exception as e:
        print(f"Error reading job_mapping.xlsx: {e}. Please ensure it's accessible and 'openpyxl' is installed if needed.")
        print("Attempting to load 'job_mapping.csv' instead...")
        try:
            job_df = pd.read_csv(DATA_PATH + 'job_mapping.csv') # Fallback to CSV
        except Exception as e_csv:
            print(f"Error reading job_mapping.csv: {e_csv}. Job mapping might be incomplete.")
            # Create a dummy job_df if loading fails, so script can proceed with a warning
            job_df = pd.DataFrame(columns=['emp_title', 'job_level', 'profession'])


    # Merge data 
    merged_df = pd.merge(customer_df, loan_df, on='customer_id', how='left')
    merged_df = pd.merge(merged_df, state_df, on='state', how='left')
    merged_df = pd.merge(merged_df, job_df, on='emp_title', how='left')


    # Filter data as in notebook
    merged_df['issue_d_datetime'] = pd.to_datetime(merged_df['issue_d'], format='%b-%y', errors='coerce')
    merged_df = merged_df[merged_df['issue_d_datetime'] < pd.to_datetime('2016-01-01')] # issue_year < 2016
    
    # Map target variable
    merged_df['loan_status'] = merged_df['loan_status'].apply(map_loan_status)
    merged_df.dropna(subset=['loan_status'], inplace=True) # Remove rows where target is NaN
    merged_df['loan_status'] = merged_df['loan_status'].astype(int)

    # 2. Feature Engineering
    print("Engineering features...")
    
    df_processed, feature_eng_params = engineer_features(merged_df, params=None, is_training=True)
    feature_eng_params['current_date_for_age_calc'] = CURRENT_DATE_FOR_AGE_CALC.isoformat()


    # 3. Data Split (Temporal)
    print("Splitting data...")
    df_train = df_processed[(df_processed['issue_year_num'] < 2015) | 
                            ((df_processed['issue_year_num'] == 2015) & (df_processed['issue_month'] <= 6))]
    # df_test = df_processed[(df_processed['issue_year_num'] == 2015) & (df_processed['issue_month'] > 6)] # Not used for training final model

    target_col = 'loan_status'
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    # Define feature types for ColumnTransformer 
    num_cols = ['emp_length', 'avg_cur_bal', 'Tot_cur_bal', 'loan_amount', 'int_rate', 'installment', 'issue_month', 'issue_quarter', 'issue_year_num', 'loan_age_days', 'total_inc', 'payment_to_income', 'lti', 'interest_burden', 'bal_to_loan', 'bal_to_income', 'rate_premium', 'total_debt_ratio', 'relative_loan_size', 'monthly_interest', 'interest_to_payment_ratio', 'loan_to_balance_ratio']

    cat_cols_onehot = ['home_ownership', 'verification_status', 'term', 'purpose', 'subregion', 'job_level', 'profession']

    ordinal_cols = ['grade']
    
    # Ensure no overlap and all columns are covered or intentionally dropped
    processed_cols = num_cols + cat_cols_onehot + ordinal_cols
    for col in X_train.columns:
        if col not in processed_cols:
            print(f"Warning: Column '{col}' from X_train is not in num_cols, cat_cols_onehot, or ordinal_cols. It will be dropped by ColumnTransformer.")


    # 4. Preprocessor Definition 
    print("Defining preprocessor...")
    grade_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] 

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[grade_categories], handle_unknown='use_encoded_value', unknown_value=-1)) # handle unknown for grade
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer_onehot, cat_cols_onehot),
            ('ord_cat', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop', # Explicitly drop columns not specified
        verbose_feature_names_out=False
    )

    # 5. Model Definition (Weighted Voting Ensemble from notebook)
    print("Defining model...")
    class_counts = Counter(y_train)
    scale_pos_weight_val = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

    lgb_clf = lgb.LGBMClassifier(
        random_state=42, is_unbalance=True, 
        class_weight={0: 1, 1: scale_pos_weight_val}, 
        objective='binary', metric='auc', boost_from_average=False,
        n_estimators=100, num_leaves=31, learning_rate=0.05
    )
    cat_clf = CatBoostClassifier(
        random_seed=42, auto_class_weights='Balanced', 
        loss_function='Logloss', eval_metric='AUC', iterations=100,
        depth=6, learning_rate=0.05, verbose=False
    )
    xgb_clf = XGBClassifier(
        random_state=42, scale_pos_weight=scale_pos_weight_val,
        max_depth=5, learning_rate=0.1, n_estimators=100,
        use_label_encoder=False, eval_metric='aucpr', gamma=2.0 
    )
    
    voting_clf = VotingClassifier(
        estimators=[('lgb', lgb_clf), ('cat', cat_clf), ('xgb', xgb_clf)],
        voting='soft',
        weights=[1, 4, 3] 
    )

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])

    # 6. Train Model
    print("Training full pipeline...")
    full_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 7. Save Model, Preprocessor, and Params
    print("Saving artifacts...")
    # Create directory if it doesn't exist
    import os
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    joblib.dump(full_pipeline.named_steps['preprocessor'], SAVED_PREPROCESSOR_PATH)
    joblib.dump(full_pipeline.named_steps['classifier'], SAVED_MODEL_PATH) # Save only the classifier part

    with open(SAVED_PARAMS_PATH, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        for key, value in feature_eng_params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                         for s_sub_key, s_sub_value in sub_value.items():
                            if isinstance(s_sub_value, (np.int64, np.float64)):
                                feature_eng_params[key][sub_key][s_sub_key] = s_sub_value.item()
                    elif isinstance(sub_value, (np.int64, np.float64)):
                         feature_eng_params[key][sub_key] = sub_value.item()
            elif isinstance(value, (np.int64, np.float64)):
                 feature_eng_params[key] = value.item()
        json.dump(feature_eng_params, f, indent=4)

    print(f"Preprocessor saved to {SAVED_PREPROCESSOR_PATH}")
    print(f"Model saved to {SAVED_MODEL_PATH}")
    print(f"Feature engineering parameters saved to {SAVED_PARAMS_PATH}")
    print("Training script finished successfully.")

if __name__ == '__main__':
    # Create dummy data files if they don't exist for testing the script structure
    import os
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Create minimal dummy files if they are missing, just to allow script to run without file not found.
    # This is for placeholder purposes only.
    dummy_customer_content = "customer_id,emp_title,state,annual_inc,annual_inc_joint,type,home_ownership\nC1,Teacher,CA,60000,,INDIVIDUAL,MORTGAGE"
    dummy_loan_content = "loan_id,customer_id,loan_amount,term,int_rate,grade,loan_status,issue_d,purpose,installment,avg_cur_bal,tot_cur_bal\nL1,C1,10000, 36 months,10.0,B,Fully Paid,Jan-15,debt_consolidation,300,5000,15000"
    dummy_state_content = "state,region,subregion\nCA,West,Pacific"
    dummy_job_content = "emp_title,job_level,profession\nTeacher,Entry Level,Educator/Teaching" 

    if not os.path.exists(CUSTOMER_CSV): pd.DataFrame([x.split(',') for x in dummy_customer_content.split('\n')[1:]], columns=dummy_customer_content.split('\n')[0].split(',')).to_csv(CUSTOMER_CSV, index=False)
    if not os.path.exists(LOAN_CSV): pd.DataFrame([x.split(',') for x in dummy_loan_content.split('\n')[1:]], columns=dummy_loan_content.split('\n')[0].split(',')).to_csv(LOAN_CSV, index=False)
    if not os.path.exists(STATE_REGION_CSV): pd.DataFrame([x.split(',') for x in dummy_state_content.split('\n')[1:]], columns=dummy_state_content.split('\n')[0].split(',')).to_csv(STATE_REGION_CSV, index=False)
    if not os.path.exists(JOB_MAPPING_XLSX) and not os.path.exists(DATA_PATH+'job_mapping.csv'): # Check for CSV too
        pd.DataFrame([x.split(',') for x in dummy_job_content.split('\n')[1:]], columns=dummy_job_content.split('\n')[0].split(',')).to_csv(DATA_PATH+'job_mapping.csv', index=False)
        print(f"Created dummy job_mapping.csv as {JOB_MAPPING_XLSX} was not found.")

    train()
