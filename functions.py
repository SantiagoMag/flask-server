from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib



def preprocessing(df):
    df_ = df[['person_income','loan_amnt', 'person_age', 'loan_int_rate']]
    scaler = joblib.load('scaler.pkl')
    df_ = scaler.transform(df_)
    return df_