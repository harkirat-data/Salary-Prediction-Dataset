import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
 
df = pd.read_csv("job_salary_prediction_dataset.csv")
print(f"Dataset loaded: {df.shape}")
 
cat_cols   = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
 
X = df_encoded.drop('salary', axis=1)
y = df_encoded['salary']
 
feature_cols = X.columns.tolist()

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

xgb = XGBRegressor( n_estimators = 100, max_depth = 10 , random_state= 42 ,tree_method = 'hist')

xgb.fit(X_train,y_train)

print("Model Trained")

y_pred = xgb.predict(X_test)

metrics = {
    'R_Squared': round(r2_score(y_test,y_pred)),
    'MAE': round(mean_absolute_error(y_test,y_pred)),
    'MSE': round(mean_squared_error(y_test,y_pred))
}

print('Metrics :', metrics)

payload = {
    'model': xgb,
    'feature_cols': feature_cols,
    'metrics': metrics
}

with open("model.pkl", 'wb') as file:
    pickle.dump(payload, file)

