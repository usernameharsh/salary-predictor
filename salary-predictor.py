import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

salary_data = pd.read_csv('archive/salaries.csv')

salary_data = salary_data.drop(['salary', 'salary_currency'], axis=1)

y = salary_data['salary_in_usd'].values
top_10_title = salary_data['job_title'].value_counts().index[:10].tolist()
salary_data['job_title'] = salary_data['job_title'].apply(lambda x: x if x in top_10_title else 'Other')



categorical_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']


encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    salary_data[col] = le.fit_transform(salary_data[col].astype(str))
    encoders[col] = le 



X = salary_data.drop('salary_in_usd', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(n_estimators=200, objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.1)
xg_reg.fit(X_train, y_train)


preds = xg_reg.predict(X_test)
rmse = mean_squared_error(preds, y_test) ** 0.5
print(f"RMSE: {rmse}")


joblib.dump(xg_reg, "salarypredictor.pkl")


preprocess_data = {
    'top_10_title': top_10_title,
    'encoders': encoders
}
joblib.dump(preprocess_data, "preprocessor.pkl")