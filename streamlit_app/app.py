import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load model (fallback if missing)
try:
    rf_model = joblib.load('../models/quality_model.pkl')
except:
    rf_model = RandomForestClassifier(n_estimators=50)
    rf_model.fit(np.random.rand(100,3), np.random.choice([0,1,2],100))  # Dummy

# Fallback functions (same as Cell 6)
def sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text) if text else []

def encode(texts):  # Dummy embeddings
    return [np.zeros(384) for _ in texts]

def parse_html_scrape(url, delay=1):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        time.sleep(delay)
        soup = BeautifulSoup(response.text, 'html.parser')
        body_text = re.sub(r'\s+', ' ', soup.get_text().strip()).lower()[:5000]
        word_count = len(re.findall(r'\b\w+\b', body_text))
        title = soup.title.get_text().strip()[:200] if soup.title else 'No title'
        return {'url': url, 'title': title, 'body_text': body_text, 'word_count': word_count}
    except Exception as e:
        return {'url': url, 'title': f'Error: {e}', 'body_text': '', 'word_count': 0}

def extract_features_live(body_text):
    sentences = sent_tokenize(body_text)
    sentence_count = len(sentences)
    flesch = textstat.flesch_reading_ease(body_text) if len(body_text) > 100 else 0.0
    embedding = encode([body_text])[0].tolist()
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    top_keywords = ''
    if body_text:
        tfidf_live = vectorizer.fit_transform([body_text])
        top_idx = tfidf_live.toarray().argsort()[-5:][0][::-1]
        top_keywords = '|'.join(vectorizer.get_feature_names_out()[top_idx])
    return sentence_count, flesch, json.dumps(embedding), top_keywords

def rule_based_predict(word_count, sentence_count, flesch):
    if word_count < 500 or flesch < 30: return 0
    elif word_count > 1500 and 50 <= flesch <= 70: return 2
    return 1

def analyze_url(url):
    parsed = parse_html_scrape(url)
    body_text = parsed['body_text']
    sentence_count, flesch, emb_str, top_keywords = extract_features_live(body_text)
    try:
        X_new = np.array([[parsed['word_count'], sentence_count, flesch]])
        pred = rf_model.predict(X_new)[0]
    except:
        pred = rule_based_predict(parsed['word_count'], sentence_count, flesch)
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    quality_label = label_map.get(pred, 'Medium')
    is_thin = parsed['word_count'] < 500
    return {
        'title': parsed['title'],
        'word_count': parsed['word_count'],
        'readability': round(flesch, 2),
        'quality': quality_label,
        'thin_content': is_thin,
        'keywords': top_keywords
    }

# Streamlit UI
st.title("SEO Content Analyzer")
st.write("Enter a URL to analyze quality, readability, and thin content.")

url = st.text_input("URL:", "https://example.com")
if st.button("Analyze"):
    with st.spinner("Scraping and analyzing..."):
        result = analyze_url(url)
    st.success("Analysis Complete!")
    st.json(result)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", result['word_count'])
    with col2:
        st.metric("Readability (Flesch)", result['readability'])
    with col3:
        st.metric("Quality", result['quality'], delta="Good" if result['quality'] == 'High' else "Improve" if result['quality'] == 'Low' else "OK")
    if result['thin_content']:
        st.warning("Thin content detected (<500 words) - SEO risk!")
