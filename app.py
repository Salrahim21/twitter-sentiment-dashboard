# app.py

import streamlit as st
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt

# ----------------------------
# CLEAN TEXT FUNCTION
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# LOAD PRE-TRAINED MODEL
# ----------------------------
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.title("💬 Twitter Sentiment Analysis (Pre-trained Model)")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV with a 'Text' column", type=["csv"])

if uploaded_file:
    df_val = pd.read_csv(uploaded_file)

    if "Text" not in df_val.columns:
        st.error("❌ Uploaded CSV must contain a 'Text' column.")
    else:
        df_val['clean_text'] = df_val['Text'].apply(clean_text)

        X_val = vectorizer.transform(df_val['clean_text'])

        preds = model.predict(X_val)
        df_val['Predicted_Sentiment'] = preds

        st.success("✅ Predictions complete!")

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(df_val['Predicted_Sentiment'].value_counts())

        st.subheader("📝 Example Tweets per Sentiment")
        sentiment_choice = st.selectbox("Choose a sentiment", df_val['Predicted_Sentiment'].unique())

        examples = df_val[df_val['Predicted_Sentiment'] == sentiment_choice]['Text'].head(5).to_list()
        for tweet in examples:
            st.write(f"• {tweet}")

        csv = df_val.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
