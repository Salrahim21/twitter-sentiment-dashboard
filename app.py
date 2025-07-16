# app.py

import streamlit as st
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt

# ----------------------------
# TEXT CLEANER CLASS
# ----------------------------
class TextCleaner:
    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

cleaner = TextCleaner()

# ----------------------------
# LOAD TUNED PIPELINE MODEL
# ----------------------------
@st.cache_resource
def load_pipeline():
    with open("rf_pipeline_tuned.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.title("💬 Twitter Sentiment Analysis — Tuned Random Forest Pipeline")

uploaded_file = st.file_uploader("Upload a CSV with a 'Text' column", type=["csv"])

if uploaded_file:
    df_val = pd.read_csv(uploaded_file)

    if "Text" not in df_val.columns:
        st.error("❌ Uploaded CSV must contain a 'Text' column.")
    else:
        df_val['clean_text'] = df_val['Text'].apply(cleaner.clean)

        # Pipeline handles vectorization and prediction
        preds = pipeline.predict(df_val['clean_text'])
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
