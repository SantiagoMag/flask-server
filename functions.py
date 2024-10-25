from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def transform_loan_grade(data):
    return data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], [1, 2, 3, 4, 5, 6, 7])

def preprocessing(df):

    scaler = joblib.load('scaler.pkl')
    lb_encoder_home_ownership = joblib.load('lb_encoder_home_ownership.pkl')
    lb_encoder_loan_intent = joblib.load('lb_encoder_loan_intent.pkl')
    lb_encoder_cb_person_default_on_file = joblib.load('lb_encoder_cb_person_default_on_file.pkl')
    
    df["home_ownership"] = lb_encoder_home_ownership.transform(df["home_ownership"])  
    df["loan_intent"] = lb_encoder_loan_intent.transform(df["loan_intent"])  
    df["cb_person_default_on_file"] = lb_encoder_cb_person_default_on_file.transform(df["cb_person_default_on_file"])  
    df["loan_grade"] = transform_loan_grade(df["loan_grade"])  

    df_ = df[
        ['person_income', 'cb_person_default_on_file', 'person_home_ownership', 'loan_int_rate', 'loan_grade', 'loan_percent_income']
        ]
    df_ = scaler.transform(df_)
    
    return df_