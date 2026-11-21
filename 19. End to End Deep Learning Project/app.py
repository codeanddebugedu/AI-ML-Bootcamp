import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf


# label, onehot, scaler
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.keras")
    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)
    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()
st.title("Customer Churn Prediction")
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_df = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])
geo_encoded = onehot_encoder_geo.transform(input_df[["Geography"]])
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_df)


prediction_sigmoid_value = model.predict(input_scaled, verbose=0)[0][0]
st.write(f"Churn Probability: {prediction_sigmoid_value:.2f}")
if prediction_sigmoid_value > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
