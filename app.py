import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession

# Page configuration
st.set_page_config(
    page_title="Spark NLP Sentiment Dashboard", page_icon="📊", layout="centered"
)


# 1. Initialize Spark Session (cached)
@st.cache_resource
def get_spark():
  return (
      SparkSession.builder.appName("SparkNLP_StreamlitApp")
      .master("local[*]")
      .getOrCreate()
  )


spark = get_spark()


# 2. Train Spark NLP Pipeline (Tokenizer -> HashingTF -> LogisticRegression)
@st.cache_resource
def get_trained_pipeline():
  training_data = spark.createDataFrame(
      [
          ("This is a fantastic, wonderful, and amazing product!", 1.0),
          ("Terrible experience, completely broken and useless.", 0.0),
          ("Amazing quality, highly recommend it to everyone.", 1.0),
          ("Worst purchase ever, absolute waste of money.", 0.0),
          ("Love it, works like absolute charm.", 1.0),
          ("Awful customer service and poor build quality.", 0.0),
      ]
      * 20
  ,
      ["text", "label"],
  )

  tokenizer = Tokenizer(inputCol="text", outputCol="words")
  hashingTF = HashingTF(
      inputCol="words", outputCol="features", numFeatures=1000
  )
  lr = LogisticRegression(maxIter=10, regParam=0.01)

  pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
  model = pipeline.fit(training_data)
  return model, training_data


model, training_df = get_trained_pipeline()

# UI Layout
st.title("📊 Spark MLlib NLP Sentiment Analyzer")
st.write(
    "A live web app powered by **Apache Spark MLlib Pipelines (Tokenizer,"
    " HashingTF, Logistic Regression)** and **Streamlit**."
)

# User Input Text
st.sidebar.header("✍️ Test a Sentence")
user_text = st.sidebar.text_area(
    "Enter a review or sentence:", "This product is fantastic and works great!"
)

if st.button("Analyze Sentiment", type="primary"):
  # Predict input text
  test_df = spark.createDataFrame([(user_text,)], ["text"])
  prediction_df = model.transform(test_df)
  result = prediction_df.collect()[0]

  pred_class = int(result["prediction"])
  prob_vector = result["probability"].toArray()
  conf_score = float(prob_vector[pred_class])

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
pdf = training_df.toPandas()
class_counts = (
    pdf["label"].value_counts().rename({0.0: "Negative", 1.0: "Positive"})
)
st.bar_chart(class_counts)
