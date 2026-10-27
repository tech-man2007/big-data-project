import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="NLP Sentiment Dashboard", page_icon="📊", layout="centered"
)


# 1. Train and cache the NLP model pipeline
@st.cache_resource
def get_trained_model():
  training_texts = [
      "This is a fantastic, wonderful, and amazing product!",
      "Terrible experience, completely broken and useless.",
      "Amazing quality, highly recommend it to everyone.",
      "Worst purchase ever, absolute waste of money.",
      "Love it, works like absolute charm.",
      "Awful customer service and poor build quality.",
  ] * 20

  training_labels = [1, 0, 1, 0, 1, 0] * 20

  # Pipeline: TF-IDF Vectorizer -> Logistic Regression Classifier
  model = Pipeline([
      ("tfidf", TfidfVectorizer()),
      ("clf", LogisticRegression()),
  ])
  model.fit(training_texts, training_labels)

  # DataFrame for statistics
  training_df = pd.DataFrame(
      {"text": training_texts, "label": training_labels}
  )
  return model, training_df


model, training_df = get_trained_model()

# UI Layout
st.title("📊 NLP Sentiment Analyzer Dashboard")
st.write(
    "A live web app powered by **Machine Learning (TF-IDF, Logistic"
    " Regression)** and **Streamlit**."
)

# User Input Text
st.sidebar.header("✍️ Test a Sentence")
user_text = st.sidebar.text_area(
    "Enter a review or sentence:", "This product is fantastic and works great!"
)

if st.button("Analyze Sentiment", type="primary"):
  # Predict input text
  pred_class = model.predict([user_text])[0]
  prob_vector = model.predict_proba([user_text])[0]
  conf_score = prob_vector[pred_class]

  st.divider()
  col1, col2 = st.columns(2)
  with col1:
    if pred_class == 1:
      st.success("**Prediction: Positive Sentiment 😊**")
    else:
      st.error("**Prediction: Negative Sentiment 😞**")
  with col2:
    st.metric(label="Confidence Score", value=f"{conf_score:.2%}")

  # Visualization 1: Prediction Confidence Bar Chart
  st.subheader("📈 Prediction Probability Breakdown")
  chart_data = pd.DataFrame(
      {
          "Sentiment Category": ["Negative (0)", "Positive (1)"],
          "Probability": prob_vector,
      }
  )
  st.bar_chart(chart_data.set_index("Sentiment Category"))

# Visualization 2: Dataset Statistics Chart
st.divider()
st.subheader("📊 Training Dataset Class Distribution")
class_counts = (
    training_df["label"].value_counts().rename({0: "Negative", 1: "Positive"})
)
st.bar_chart(class_counts)
