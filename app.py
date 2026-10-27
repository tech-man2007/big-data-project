import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="IT Incident & Log Analyzer", page_icon="💻", layout="centered"
)


# 1. Train and cache the IT text classification model
@st.cache_resource
def get_trained_model():
  training_logs = [
      "Database connection pool exhausted and timeout occurred on master node",
      "SQL syntax error while executing user query on reporting schema",
      "Network latency spike detected on firewall gateway interface 3",
      "Packet loss and high jitter on primary router uplink",
      "User authentication failed due to invalid LDAP token expiration",
      "Multiple failed SSH login attempts detected from unauthorized IP",
      "Disk storage usage exceeded 95 percent threshold on volume var",
      "Out of memory error triggered by background worker process",
  ] * 15

  # Categories: 0: Database, 1: Network, 2: Security, 3: Infrastructure/System
  training_labels = [
      "Database Error",
      "Database Error",
      "Network Issue",
      "Network Issue",
      "Security Alert",
      "Security Alert",
      "Infrastructure Warning",
      "Infrastructure Warning",
  ] * 15

  # Pipeline: TF-IDF Vectorizer -> Logistic Regression Classifier
  model = Pipeline([
      ("tfidf", TfidfVectorizer()),
      ("clf", LogisticRegression()),
  ])
  model.fit(training_logs, training_labels)

  training_df = pd.DataFrame(
      {"log_text": training_logs, "category": training_labels}
  )
  return model, training_df


model, training_df = get_trained_model()

# UI Layout
st.title("💻 IT Incident & Log Classification Dashboard")
st.write(
    "A live web application powered by **Machine Learning (TF-IDF & Logistic"
    " Regression)** to automatically categorize technical support tickets and"
    " server error logs."
)

# User Input Text
st.sidebar.header("🔍 Test a System Log / Ticket")
user_input = st.sidebar.text_area(
    "Enter system log message or ticket description:",
    "Connection refused when attempting to reach the primary database cluster",
)

if st.button("Classify Incident", type="primary"):
  # Predict category
  predicted_category = model.predict([user_input])[0]
  probabilities = model.predict_proba([user_input])[0]
  confidence = max(probabilities)

  st.divider()
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("🏷️ Classification Result")
    st.info(f"**Detected Category:** {predicted_category}")
  with col2:
    st.metric(label="Model Confidence", value=f"{confidence:.2%}")

  # Visualization: Prediction Probability across all IT Categories
  st.subheader("📊 Category Probability Distribution")
  prob_df = pd.DataFrame(
      {"Category": model.classes_, "Probability": probabilities}
  )
  st.bar_chart(prob_df.set_index("Category"))

# Dataset Analytics Section
st.divider()
st.subheader("📈 Training Dataset Incident Distribution")
category_counts = training_df["category"].value_counts()
st.bar_chart(category_counts)
