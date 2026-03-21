import streamlit as st
import joblib
import numpy as np 
import pandas as pd

# Load the trained model. 
model = joblib.load("Income_Classification.pkl")

# Load the scaler. 
scaler = joblib.load("Scaler.pkl")

# Load the list of model columns. 
model_columns = joblib.load("Model_Columns.pkl") # The list of 100+ columns. 

st.title("Income Classification App")

st.write("Please Enter the Following Details to Classify Income:")

# Create input fields for user to enter data.
Age = st.number_input("Age", min_value = 0, step = 1)
Workclass = st.selectbox("Workclass", options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
Final_Weight = st.number_input("Final Weight (fnlwgt)", min_value = 0.0, step = 100.0)
Education = st.selectbox("Education", options = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
Education_Num = st.number_input("Education Number (education-num)", min_value = 0, step = 1)
Marital_Status = st.selectbox("Marital Status", options = ["Married-civ spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
Occupation = st.selectbox("Occupation", options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
Relationship = st.selectbox("Relationship", options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
Race = st.selectbox("Race", options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
Sex = st.selectbox("Sex", options = ["Male", "Female"])
Capital_Gain = st.number_input("Capital Gain", min_value = 0.0, step = 100.0)   
Capital_Loss = st.number_input("Capital Loss", min_value = 0.0, step = 100.0)   
Hours_Per_Week = st.number_input("Hours Per Week", min_value = 0, step = 1) 
Native_Country = st.selectbox("Native Country", options = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand"])

# When the user clicks the "Classify" button, make a classification using the model.
if st.button("Classify"):
    # 1. Create a dictionary of inputs (matching the RAW column names from your CSV)
    raw_data = {
        "age": Age,
        "workclass": Workclass,
        "fnlwgt": Final_Weight,
        "education": Education,
        "education-num": Education_Num,
        "marital-status": Marital_Status,
        "occupation": Occupation,
        "relationship": Relationship,
        "race": Race,
        "sex": Sex,
        "capital-gain": Capital_Gain,
        "capital-loss": Capital_Loss,
        "hours-per-week": Hours_Per_Week,
        "native-country": Native_Country
    }

    # Convert to DataFrame.
    df_input = pd.DataFrame([raw_data])

    # Feature Engineering: Create the Binary Flags.
    df_input["has_capital_gain"] = (df_input["capital-gain"] > 0).astype(int)
    df_input["has_capital_loss"] = (df_input["capital-loss"] > 0).astype(int)

    # Apply Log Transformation.
    df_input["fnlwgt"] = np.log1p(df_input["fnlwgt"])
    df_input["capital-gain"] = np.log1p(df_input["capital-gain"])
    df_input["capital-loss"] = np.log1p(df_input["capital-loss"])

    # One-Hot Encoding.
    df_encoded = pd.get_dummies(df_input)

    # Data Alignment. 
    # Reindex fills missing columns with 0 and ensures the order matches exactly what the model saw.
    df_final = df_encoded.reindex(columns = model_columns, fill_value = 0)

    # Scaling.
    # Scaler expects a 2D array, which df_final provides.
    df_scaled = scaler.transform(df_final)

    # Prediction.
    prediction = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)

    # Display Results.
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"Target Income: **>50K** 💰")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.info(f"Target Income: **<=50K** ⬇️")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")