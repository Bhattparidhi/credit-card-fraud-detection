import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('transactions.csv')  # your CSV

# One-hot encode transaction types
df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
print("\nOne-Hot Encoding Output:")
print(df[['type', 'type_CASH_OUT', 'type_TRANSFER']].head(10))


# Now select only the 8 features
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_TRANSFER']
X = df[features]
y = df['isFraud']  # replace with your target column

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'rf_model_8features.pkl')
joblib.dump(scaler, 'scaler_8features.pkl')

print("Retraining done. Model and scaler saved.")
