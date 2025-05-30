import pika
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime 
import os
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_QUEUE = 'loan_application_queue'
MODEL_PATH = './saved_model/' # Directory for saved artifacts

PREPROCESSOR_PATH = MODEL_PATH + 'preprocessor.joblib'
MODEL_PATH_CLASSIFIER = MODEL_PATH + 'model.joblib' # Classifier part
PARAMS_PATH = MODEL_PATH + 'feature_engineering_params.json'

# --- Load saved artifacts ---
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH_CLASSIFIER) # Classifier part
    with open(PARAMS_PATH, 'r') as f:
        feature_eng_params = json.load(f)
    
    # Convert current_date_for_age_calc back to datetime
    feature_eng_params['current_date_for_age_calc_dt'] = datetime.fromisoformat(feature_eng_params['current_date_for_age_calc'])

except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Ensure train_model.py has been run successfully.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading artifacts: {e}")
    exit()


# --- Helper Functions 

def clean_emp_length_predict(val): # Renamed to avoid conflict if imported
    if pd.isna(val) or val == '': return np.nan
    if val == '< 1 year': return 0
    if val == '10+ years': return 10
    try: return int(str(val).split()[0]) # Convert to string first for safety
    except: return np.nan

def clean_type_predict(val):
    if pd.isna(val) or val == '': return 'UNKNOWN'
    val = str(val).strip().upper()
    if val in ['INDIVIDUAL']: return 'INDIVIDUAL'
    if val in ['JOINT', 'JOINT APP']: return 'JOINT'
    if val == 'DIRECT_PAY': return 'DIRECT_PAY'
    return 'OTHER'

def ensure_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer_features_predict(data_dict, params):
    """
    Applies feature engineering to a single data instance (dict).
    Uses pre-calculated params from training.
    """
    # Convert dict to DataFrame to use pandas functionalities easily
    df = pd.DataFrame([data_dict])
    engineered_df = df.copy()

    # Date transformations
    engineered_df['issue_d'] = pd.to_datetime(engineered_df['issue_d'], format='%b-%y', errors='coerce')
    engineered_df['issue_month'] = engineered_df['issue_d'].dt.month
    engineered_df['issue_quarter'] = engineered_df['issue_d'].dt.quarter
    engineered_df['issue_year_num'] = engineered_df['issue_d'].dt.year
    
    term_num_map = {' 36 months': 36, ' 60 months': 60} # Ensure spaces match data
    engineered_df['term_num'] = engineered_df['term'].map(term_num_map)
    
    engineered_df['maturity_date'] = engineered_df.apply(
        lambda row: row['issue_d'] + pd.DateOffset(months=row['term_num']) if pd.notnull(row['issue_d']) and pd.notnull(row['term_num']) else pd.NaT,
        axis=1
    )
    
    current_date_for_calc = params['current_date_for_age_calc_dt']
    engineered_df['loan_age_days'] = (current_date_for_calc - engineered_df['issue_d']).dt.days
    engineered_df['is_maturity'] = np.where(engineered_df['maturity_date'] < current_date_for_calc, 'yes', 'no')
    
    engineered_df['emp_length'] = engineered_df['emp_length'].apply(clean_emp_length_predict)
    engineered_df['type'] = engineered_df['type'].apply(clean_type_predict)
    
    numeric_cols = ['Tot_cur_bal', 'annual_inc', 'annual_inc_joint', 'loan_amount', 'int_rate']
    engineered_df = ensure_numeric(engineered_df, numeric_cols)

    engineered_df['total_inc'] = np.where(
        engineered_df['annual_inc_joint'].isna() | (engineered_df['annual_inc_joint'] == ''),
        engineered_df['annual_inc'],
        engineered_df['annual_inc'].fillna(0) + engineered_df['annual_inc_joint'].fillna(0) # Handle potential NaNs before sum
    )
    engineered_df['have_inc_joint'] = engineered_df['annual_inc_joint'].apply(lambda x: 'yes' if pd.notnull(x) and x != '' else 'no')

    engineered_df['home_ownership'] = engineered_df['home_ownership'].replace(['NONE', 'ANY', 'OTHER'], 'OTHERS')
    
    # job_df_subset = pd.read_excel('./data/job_mapping.xlsx') 
    # state_df_subset = pd.read_csv('./data/state_region.csv')
    # engineered_df = pd.merge(engineered_df, job_df_subset, on='emp_title', how='left')
    # engineered_df = pd.merge(engineered_df, state_df_subset, on='state', how='left')
    if 'profession' not in engineered_df.columns: engineered_df['profession'] = 'Others' # Placeholder
    if 'job_level' not in engineered_df.columns: engineered_df['job_level'] = 'Entry Level' # Placeholder
    if 'subregion' not in engineered_df.columns: engineered_df['subregion'] = 'Unknown' # Placeholder

    profession_mapping = params.get('profession_mapping_from_train', { 
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
    })
    engineered_df['profession'] = engineered_df['profession'].map(profession_mapping).fillna('Others')

    engineered_df['type'] = engineered_df['type'].replace(['DIRECT_PAY', 'JOINT'], 'NON_INDIVIDUAL')

    purpose_mapping = params.get('purpose_mapping_from_train', { # Load from params if saved
        'debt_consolidation': 'debt_consolidation', 'credit_card': 'credit_card', 'home_improvement': 'home_related',
        'house': 'home_related', 'medical': 'health', 'car': 'auto', 'wedding': 'personal_lifestyle',
        'vacation': 'personal_lifestyle', 'moving': 'personal_lifestyle', 'major_purchase': 'major_purchase',
        'small_business': 'business', 'other': 'other', 'renewable_energy': 'other'
    })
    engineered_df['purpose'] = engineered_df['purpose'].map(purpose_mapping).fillna('other')

    # New numerical features
    safe_total_inc_monthly = np.where(engineered_df['total_inc'] > 0, engineered_df['total_inc'] / 12, np.nan)
    safe_total_inc = np.where(engineered_df['total_inc'] > 0, engineered_df['total_inc'], np.nan)
    safe_loan_amount = np.where(engineered_df['loan_amount'] > 0, engineered_df['loan_amount'], np.nan)
    safe_tot_cur_bal = np.where(engineered_df['Tot_cur_bal'] > 0, engineered_df['Tot_cur_bal'], 1)


    engineered_df['payment_to_income'] = engineered_df['installment'] / safe_total_inc_monthly
    engineered_df['lti'] = engineered_df['loan_amount'] / safe_total_inc
    engineered_df['interest_burden'] = engineered_df['int_rate'] * engineered_df['loan_amount'] / safe_total_inc
    engineered_df['bal_to_loan'] = engineered_df['avg_cur_bal'] / safe_loan_amount
    engineered_df['bal_to_income'] = engineered_df['avg_cur_bal'] / safe_total_inc
    
    grade_avg_rate_map = params['grade_avg_rate_map']
    grade_avg_loan_map = params['grade_avg_loan_map']

    engineered_df['grade_avg_rate'] = engineered_df['grade'].map(grade_avg_rate_map)
    engineered_df['rate_premium'] = engineered_df['int_rate'] - engineered_df['grade_avg_rate']
    
    engineered_df['total_debt_ratio'] = engineered_df['Tot_cur_bal'] / safe_total_inc
    
    engineered_df['grade_avg_loan'] = engineered_df['grade'].map(grade_avg_loan_map)
    engineered_df['relative_loan_size'] = engineered_df['loan_amount'] / np.where(engineered_df['grade_avg_loan'] > 0, engineered_df['grade_avg_loan'], np.nan)
    
    engineered_df['monthly_interest'] = (engineered_df['int_rate'] / 100 / 12) * engineered_df['loan_amount']
    engineered_df['interest_to_payment_ratio'] = engineered_df['monthly_interest'] / np.where(engineered_df['installment'] > 0, engineered_df['installment'], np.nan)
    engineered_df['loan_to_balance_ratio'] = engineered_df['loan_amount'] / safe_tot_cur_bal

    cols_to_fill_na = ['payment_to_income', 'lti', 'interest_burden', 'bal_to_loan', 'bal_to_income', 
                       'rate_premium', 'total_debt_ratio', 'relative_loan_size', 'monthly_interest', 
                       'interest_to_payment_ratio', 'loan_to_balance_ratio']
    for col in cols_to_fill_na:
        if col in engineered_df:
            median_val = params.get(f'{col}_median', 0) # Use saved median, or 0 as fallback
            engineered_df[col] = engineered_df[col].fillna(median_val)


    # Outlier Capping
    cols_to_cap = ['avg_cur_bal', 'Tot_cur_bal', 'loan_amount', 'int_rate', 'installment', 'total_inc']
    outlier_caps = params['outlier_caps']
    for col in cols_to_cap:
        if col in engineered_df and col in outlier_caps:
            col_caps = outlier_caps[col]
            engineered_df[col] = engineered_df[col].clip(lower=col_caps['lower'], upper=col_caps['upper'])

    # Drop columns not needed by preprocessor 
    cols_to_drop_post_feature_eng_predict = [
        'customer_id', 'loan_id', 'emp_title', 'issue_d', 'issue_date', 'issue_year', 
        'funded_amount', 'addr_state', 'notes', 'description', 'zip_code', 'state', 'region',
        'annual_inc', 'annual_inc_joint', 'term_num', 'maturity_date', 'grade_avg_rate', 'grade_avg_loan', 'type'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop_post_feature_eng_predict if col in engineered_df.columns]
    for col in existing_cols_to_drop:
        del engineered_df[col]

    return engineered_df


def predict_loan_default(application_data_dict):
    """
    Takes a dictionary of raw application data, preprocesses it, and returns prediction.
    """
    # 1. Engineer features
    try:
        engineered_df = engineer_features_predict(application_data_dict, feature_eng_params)
    except Exception as e:
        print(f"Error during feature engineering for prediction: {e}")
        # Potentially return an error indicator or raise
        return {"error": "Feature engineering failed", "details": str(e)}

    
    # 2. Preprocess data
    try:
        processed_data = preprocessor.transform(engineered_df) 
    except Exception as e:
        print(f"Error during data preprocessing (transform): {e}")
        print(f"Engineered columns: {engineered_df.columns.tolist()}")
        # Check if all columns expected by preprocessor are present in engineered_df
        # This often happens if feature engineering output doesn't match training
        return {"error": "Preprocessing failed", "details": str(e)}

    # 3. Make prediction
    try:
        prediction = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        return {
            "customer_id": application_data_dict.get("customer_id", "N/A"),
            "prediction_label": int(prediction[0]), # 0: On-time, 1: Default
            "probability_default": float(probabilities[0][1]), # Prob of class 1
            "probability_ontime": float(probabilities[0][0])  # Prob of class 0
        }
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return {"error": "Prediction failed", "details": str(e)}


def callback(ch, method, properties, body):
    print(f"\n [x] Received application data from queue")
    try:
        application_data = json.loads(body.decode())
        print(f"     Data: {application_data.get('customer_id', 'Unknown ID')}")

        # Make prediction
        prediction_result = predict_loan_default(application_data)
        
        print(f" [o] Prediction Result: {prediction_result}")

    except json.JSONDecodeError:
        print(" [!] Error decoding JSON message body.")
    except Exception as e:
        print(f" [!] An unexpected error occurred in callback: {e}")
    finally:
        # Acknowledge message processing
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [>] Message acknowledged for {application_data.get('customer_id', 'Unknown ID')}")


def main():
    print(f"Consumer starting. Waiting for messages from '{RABBITMQ_QUEUE}' on host '{RABBITMQ_HOST}'.")
    
    retry_interval = 5  # seconds
    while True:
        try:
            connection_params = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300)
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()

            # Declare durable queue (good practice, ensures it exists)
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            print(f" [*] Successfully connected to RabbitMQ. Waiting for messages in '{RABBITMQ_QUEUE}'. To exit press CTRL+C")

            # Fair dispatch: Don't give more than one message to a worker at a time.
            channel.basic_qos(prefetch_count=1) 
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback)
            
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection to RabbitMQ failed: {e}. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        except KeyboardInterrupt:
            print("Consumer stopped by user.")
            if 'channel' in locals() and channel.is_open:
                channel.stop_consuming()
            if 'connection' in locals() and connection.is_open:
                connection.close()
            break
        except Exception as e:
            print(f"An unexpected error occurred in consumer main loop: {e}")
            # Depending on the error, you might want to break or retry
            time.sleep(retry_interval)


if __name__ == '__main__':
    import time # for retries
    main()
