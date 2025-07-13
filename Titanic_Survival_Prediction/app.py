import streamlit as st
import joblib
import json
import pandas as pd
import base64

def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        /* Title Styling */
        h1 {{
            color: #5C4033 !important;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }}

        /* Transparent input components */
        .stSelectbox > div, .stNumberInput > div, .stSlider > div {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 12px;
            padding: 10px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.3);
        }}

        /* Make labels white */
        label, .stSlider > label, .stTextInput > label {{
            color: white !important;
            font-weight: 600;
        }}

        /* Remove top bar locally */
        header {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Call it at the start of app
set_bg("background.jpg")

# Load model and preprocessor
model = joblib.load("titanic_model.pkl")
preprocessor = joblib.load("titanic_preprocessor.pkl")
with open("titanic_model_columns.json", "r") as f:
    model_columns = json.load(f)

# UI
st.title("🚢 Titanic Survival Prediction")

# Input fields
sex = st.selectbox("Sex", ['male', 'female'])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouse Aboard", 0, 5, 0)
parch = st.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])
deck = st.selectbox("Deck", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'])
title = st.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

# Prepare input
input_dict = {
    'Sex': sex,
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked,
    'Deck': deck,
    'Title': title
}

# Predict button
if st.button("Predict Survival"):
    input_df = pd.DataFrame([input_dict])
    input_processed = preprocessor.transform(input_df)
    result = model.predict(input_processed)
    if result[0] == 1:
        st.success("🎉 This passenger **would have survived** in the Titanic!")
    else:
        st.error("💀 Unfortunately, this passenger **would not have survived.**")