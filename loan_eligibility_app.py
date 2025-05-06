import os
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.dropna(inplace=True)

# Drop loan_id column if it exists
if 'loan_id' in df.columns:
    df.drop('loan_id', axis=1, inplace=True)

# Encode target column and store mapping
target_le = LabelEncoder()
df['loan_status'] = target_le.fit_transform(df['loan_status'])
target_mapping = dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))

# Automatically find the encoded value for "Approved"
approved_class_value = None
for label, value in target_mapping.items():
    if label.strip().lower() == 'approved':
        approved_class_value = value
        break

if approved_class_value is None:
    st.error("Could not find 'Approved' label in target column.")
    st.stop()

# Label encode categorical features (excluding target)
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature/target split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ----------- Streamlit UI -------------
st.title("Loan Eligibility Predictor")

# Input interface
user_input = {}
select_fields = ["education", "self_employed"]

for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_.tolist()

        # Add "Select" option to specific fields
        if col in select_fields:
            options = ["Select"] + options
            user_value = st.selectbox(col.replace('_', ' ').capitalize(), options, index=0)
        else:
            user_value = st.selectbox(col.replace('_', ' ').capitalize(), options, index=0)

        user_input[col] = user_value
    else:
        user_input[col] = st.number_input(col.replace('_', ' ').capitalize(), value=0, step=1, format="%d")

# Show warning if "Select" is still chosen
if any(user_input[col] == "Select" for col in select_fields):
    st.warning("Please select valid options for Education and Self Employed fields.")
else:
    # Convert input into dataframe
    input_df = pd.DataFrame([user_input])

    # Encode categorical input
    for col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Ensure numeric input
    for col in input_df.columns:
        if col not in label_encoders:
            input_df[col] = input_df[col].astype(int)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict and show result
    if st.button("Check Eligibility"):
        prediction = model.predict(input_scaled)[0]
        if prediction == approved_class_value:
            st.success("Eligible")
        else:
            st.error("Not Eligible")