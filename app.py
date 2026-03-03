import streamlit as st
import pickle
import re
import numpy as np

model = pickle.load(open("model/phishing_model.pkl", "rb"))

st.set_page_config(page_title="Cyber Shield", layout="centered")

st.markdown(
    """
    <style>
    body {background-color: #0E1117; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ AI Phishing URL Detector")

url = st.text_input("Enter Website URL")

def extract_features(url):
    return np.array([
        len(url),
        1 if "https" in url else 0,
        url.count("."),
        url.count("-"),
        url.count("@")
    ]).reshape(1, -1)

if st.button("Check"):
    features = extract_features(url)
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ This URL is Phishing!")
    else:
        st.success("✅ This URL is Safe")
