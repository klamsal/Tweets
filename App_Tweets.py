
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Load data function
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# File path configuration
DATA_URL = "Tweets.csv"

# Application title and description
st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis Dashboard")
st.markdown(
    "This application is a Streamlit dashboard to analyze sentiments "
    "of tweets related to US Airlines."
)

# Load data
data = load_data(DATA_URL)

# Sidebar options
st.sidebar.subheader("Visualization Selector")
sentiment_count = st.sidebar.checkbox("Show Sentiment Count")
word_cloud = st.sidebar.checkbox("Generate Word Cloud")
tweet_length_histogram = st.sidebar.checkbox("Show Tweet Length Histogram")

# Data preview
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data.head())

# Sentiment Count Visualization
if sentiment_count:
    st.subheader("Number of Tweets by Sentiment")
    sentiment_counts = data['airline_sentiment'].value_counts()
    sentiment_counts_df = pd.DataFrame({
        'Sentiment': sentiment_counts.index,
        'Tweets': sentiment_counts.values
    })
    fig = px.bar(sentiment_counts_df, x='Sentiment', y='Tweets', color='Sentiment', title="Tweet Sentiment Count")
    st.plotly_chart(fig)

# Word Cloud Visualization
if word_cloud:
    st.subheader("Word Cloud for Selected Sentiment")
    sentiment = st.sidebar.selectbox("Choose a Sentiment", data['airline_sentiment'].unique())
    filtered_data = data[data['airline_sentiment'] == sentiment]
    words = " ".join(filtered_data['text'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Histogram for Tweet Lengths
if tweet_length_histogram:
    st.subheader("Histogram of Tweet Lengths")
    data['tweet_length'] = data['text'].apply(len)
    fig = px.histogram(data, x='tweet_length', nbins=30, title="Distribution of Tweet Lengths")
    st.plotly_chart(fig)
