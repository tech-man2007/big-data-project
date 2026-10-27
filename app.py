import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="IT Incident & Log Analyzer", page_icon="💻", layout="centered"
)


# 1. Train and cache an expanded, distinct IT text classification model
@st.cache_resource
def get_trained_model():
  training_logs = [
      # Database Errors
      (
          "Database connection pool exhausted and timeout occurred on master"
          " node"
      ),
      (
          "SQL syntax error while executing user query on reporting schema"
          " database"
      ),
      "Deadlock detected on transaction table, connection dropped",
      # Network Issues
      "Network latency spike detected on firewall gateway interface 3",
      "Packet loss and high jitter on primary router uplink connection",
      "DNS resolution failure when reaching external API gateway",
      # Security Alerts
      "Multiple failed SSH login attempts detected from unauthorized IP address",
      (
          "User authentication failed due to invalid LDAP token expiration and"
          " brute force suspicion"
      ),
      "Unauthorized firewall rule modification detected on perimeter node",
      # Infrastructure Warnings
      (
          "Disk storage usage exceeded 95 percent threshold on volume var"
          " partition"
      ),
      "Out of memory error triggered by background worker container process",
      "CPU thermal throttling event logged on hypervisor node 4",
  ]

  training_labels = [
      "Database Error",
      "Database Error",
      "Database Error",
      "Network Issue",
      "Network Issue",
      "Network Issue",
      "Security Alert",
      "Security Alert",
      "Security Alert",
      "Infrastructure Warning",
      "Infrastructure Warning",
      "Infrastructure Warning",
  ] * 5  # Multiply to give the classifier more training weight

  # Pipeline: TF-IDF Vectorizer -> Logistic Regression Classifier
  model = Pipeline([
      ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
      ("clf", LogisticRegression()),
  ])
  model.fit(training_logs, training_labels)

  training_df = pd.DataFrame(
      {"log_text": training_logs, "category": training_labels}
  )
  return model, training_df


model, training_df = get_trained_model()

# UI Layout
st.title("💻 IT Incident & Log Classification System")
st.markdown(
    "An AI-powered dashboard designed to automatically categorize incoming"
    " technical support tickets and server infrastructure logs using **Machine"
    " Learning (TF-IDF & Logistic Regression)**."
)

# User Input Text
st.sidebar.header("🔍 Test a System Log / Ticket")
user_input = st.sidebar.text_area(
    "Enter system log message or ticket description:",
    "Connection refused when attempting to reach the infrastructure server",
)

if st.button("Classify Incident", type="primary"):
  # Predict category
  predicted_category = model.predict([user_input])[0]
  probabilities = model.predict_proba([user_input])[0]
  confidence = max(probabilities)

  st.divider()

  # Top Metric Cards
  col1, col2 = st.columns(2)
  with col1:
    st.metric(
        label="Predicted Incident Category", value=predicted_category
    )
  with col2:
    st.metric(label="Model Confidence Score", value=f"{confidence:.2%}")

  st.markdown("### 📊 Interactive Probability Breakdown")

  # Visualization 1: Plotly Horizontal Bar Chart for Confidence Distribution
  prob_df = pd.DataFrame(
      {"Category": model.classes_, "Probability": probabilities}
  )
  fig_bar = px.bar(
      prob_df,
      x="Probability",
      y="Category",
      orientation="h",
      text_auto=".1%",
      color="Probability",
      color_continuousscale="Blues",
      title="Confidence per Category",
  )
  fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
  st.plotly_chart(fig_bar, use_container_width=True)

# Dataset Analytics Section
st.divider()
st.subheader("📈 Training Dataset Distribution")
category_counts = (
    training_df["category"].value_counts().reset_index()
)
category_counts.columns = ["Category", "Count"]

# Visualization 2: Plotly Donut Chart for Dataset Balance
fig_donut = px.pie(
    category_counts,
    names="Category",
    values="Count",
    hole=0.4,
    title="Proportion of Training Logs per IT Category",
)
st.plotly_chart(fig_donut, use_container_width=True)
