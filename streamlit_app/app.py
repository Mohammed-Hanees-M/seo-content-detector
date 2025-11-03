import streamlit as st
import requests
from bs4 import BeautifulSoup
import textstat
import re
import time
import joblib  # For model loading
import numpy as np
from collections import Counter
import os  # For path checks
import pandas as pd  # For CSV/export/table
import plotly.express as px  # For charts
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import string


# Config for wide, professional layout (MUST be first Streamlit command)
st.set_page_config(page_title="SEO Content Analyzer Pro", layout="wide", page_icon="🔍")


# NLTK initialization (downloads data on first run, cached)
@st.cache_data
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
init_nltk()

# Semantic keywords via NLTK (noun extraction)
def extract_semantics(text, max_keywords=5):
    if not text:
        return []
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in string.punctuation and w not in stop_words]
    pos_tags = pos_tag(tokens)
    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    counter = Counter(nouns)
    return [word for word, _ in counter.most_common(max_keywords)]


# Load or init model (robust fallback for corruption/no file)
@st.cache_resource
def load_model():
    model_path = "../models/quality_model.pkl"  # Relative to streamlit_app/
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.sidebar.warning("No trained model found—using rule-based fallback for predictions.")
    except (ImportError, EOFError, FileNotFoundError, Exception) as e:
        st.sidebar.warning(f"Model load issue ({str(e)}) — switched to rule-based predictions.")
    
    # Fallback: Rule-based classifier (weighted: 30% wc, 70% readability)
    class RuleClassifier:
        def predict_proba(self, X):
            probs = np.zeros((len(X), 3))  # Low=0, Medium=1, High=2
            for i, row in enumerate(X):
                wc, read = row[0], row[1] if len(row) > 1 else 50  # Default read if missing
                read = max(0, min(100, read))  # Clamp 0-100 for negatives/extremes
                score = 0.3 * (min(wc / 1000, 1)) + 0.7 * (read / 100)  # Normalized score 0-1
                if score > 0.7: probs[i, 2] = 1  # High
                elif score > 0.4: probs[i, 1] = 1  # Medium
                else: probs[i, 0] = 1  # Low
            return probs
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
    return RuleClassifier()


model = load_model()


# Extract text and title (cached for speed)
@st.cache_data(ttl=3600)  # Cache 1hr; easy speed boost
def extract_content(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script/style for clean text
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        title = soup.title.string.strip() if soup.title else "N/A"
        return text, title
    except Exception as e:
        st.error(f"❌ Scrape error for {url}: {str(e)}. Check URL or network.")
        return "", "Error"


# Extract top 5 keywords
def extract_keywords(text):
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common = Counter(words).most_common(5)
    return [word for word, _ in common]


# Feature extraction (wc, readability)
def get_features(text):
    if not text:
        return 0, 0
    wc = len(re.findall(r'\w+', text))  # Robust word count
    read = textstat.flesch_reading_ease(text[:2000]) if wc > 10 else 0  # Limit for speed/clamp
    return wc, read


# Custom CSS for professional theme (enhanced for dark mode)
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stMetric {background-color: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .high {color: #10B981; font-weight: bold;}
    .medium {color: #F59E0B; font-weight: bold;}
    .low {color: #EF4444; font-weight: bold;}
    .metric-label {font-size: 0.9em; color: gray;}
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    </style>
""", unsafe_allow_html=True)


# Header
st.title("🔍 SEO Content Analyzer Pro")
st.markdown("**Professional tool for evaluating webpage SEO quality: Length, readability, thin content, and AI predictions.**")
st.markdown("---")


# Sidebar for settings (enhanced with dark toggle & batch)
with st.sidebar:
    st.header("⚙️ Analyzer Settings")
    dark_mode = st.checkbox("🌙 Dark Mode")  # Easy: Toggle theme
    if dark_mode:
        st.markdown("""
            <style>
            /* Global Dark Theme */
            .stApp { 
                background-color: #0e1117 !important; 
                color: #f9fafb !important; 
            }
            .main { 
                background-color: #0e1117 !important; 
                color: #f9fafb !important; 
            }
            .block-container {
                background-color: #0e1117 !important;
                color: #f9fafb !important;
            }
            /* Titles and Headings */
            h1, h2, h3, h4, h5, h6 {
                color: #f9fafb !important;
            }
            /* Paragraphs and Text */
            p, div, span, li, td, th {
                color: #f9fafb !important;
            }
            .stMarkdown {
                color: #f9fafb !important;
            }
            /* Sidebar */
            [data-testid="stSidebar"] { 
                background-color: #111827 !important; 
                color: #f9fafb !important; 
            }
            section[data-testid="stSidebar"] div[role="heading"] {
                color: #f9fafb !important;
            }
            /* Metrics */
            .stMetric { 
                background-color: #1f2937 !important; 
                color: #f9fafb !important; 
                border: 1px solid #374151 !important; 
            }
            .stMetric > label {
                color: #d1d5db !important;
            }
            .stMetric > div > div {
                color: #f9fafb !important;
            }
            /* Inputs and Text Areas */
            .stTextInput > div > div > input, 
            .stTextArea > div > div > textarea {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
                border: 1px solid #374151 !important;
            }
            .stTextInput > div > div > input::placeholder,
            .stTextArea > div > div > textarea::placeholder {
                color: #9ca3af !important;
            }
            .stTextInput label, .stTextArea label {
                color: #f9fafb !important;
            }
            /* Buttons */
            .stButton > button {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
                border: 1px solid #374151 !important;
            }
            .stButton > button:hover {
                background-color: #374151 !important;
                color: #f9fafb !important;
            }
            /* Sliders and Checkboxes */
            .stSlider > div > div > div {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            .stCheckbox > label {
                color: #f9fafb !important;
            }
            /* Expanders */
            .stExpander {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            .stExpander > label {
                color: #f9fafb !important;
            }
            .stExpander > div[role="button"] {
                color: #f9fafb !important;
            }
            /* DataFrames and Tables */
            .stDataFrame {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            .dataframe {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            .dataframe th, .dataframe td {
                color: #f9fafb !important;
                border-color: #374151 !important;
            }
            /* Plots and Charts */
            .stPlotlyChart {
                background-color: #1f2937 !important;
            }
            /* JSON and Raw Outputs */
            pre, code {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            /* Download Buttons */
            .stDownloadButton > button {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
                border: 1px solid #374151 !important;
            }
            /* Warnings, Infos, Errors, Success */
            .stWarning, .stInfo, .stError, .stSuccess {
                color: #f9fafb !important;
                background-color: #374151 !important;
                border-color: #4b5563 !important;
            }
            .stAlert > div {
                color: #f9fafb !important;
            }
            /* Captions and Small Text */
            .stCaption {
                color: #9ca3af !important;
            }
            /* Progress Bars */
            .stProgress > div > div {
                background-color: #1f2937 !important;
            }
            /* Spinner */
            .stSpinner > div {
                color: #f9fafb !important;
            }
            /* Footer and General */
            .css-1d391kg {
                color: #f9fafb !important;
            }
            footer {
                color: #f9fafb !important;
            }
            </style>
        """, unsafe_allow_html=True)
    max_words = st.slider("Max words to process:", 500, 5000, 2000, help="Limits analysis for long pages.")
    show_raw = st.checkbox("📄 Show raw JSON output", value=True)
    batch_mode = st.checkbox("📈 Batch Analysis (up to 3 URLs)")
    if batch_mode:
        batch_urls_input = st.text_area("Enter URLs (one per line):", height=100, placeholder="https://example1.com\nhttps://example2.com").splitlines()
        batch_urls = [u.strip() for u in batch_urls_input if u.strip()]
        if len(batch_urls) > 3:
            st.warning("Limited to 3 URLs for performance.")
            batch_urls = batch_urls[:3]
        if not batch_urls:
            st.warning("Add at least one valid URL for batch analysis.")
    else:
        batch_urls = []
    st.markdown("---")
    st.caption("*Built for SEO insights | Rule-based fallback active if no model.*")


# Main input row (responsive container)
with st.container():  # Easy: Responsive wrapper
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("📝 Enter URL to analyze:", placeholder="https://surferseo.com/blog/seo-content-writing-examples/", help="Paste a valid webpage URL.")
    with col2:
        analyze = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# Batch or single analysis
if analyze or batch_mode:
    # Validate inputs
    if not batch_mode and not url.strip():
        st.error("❌ Please enter a valid URL before running analysis.")
    elif batch_mode and not batch_urls:
        st.error("❌ Please add at least one valid URL in batch mode.")
    else:
        with st.spinner("🔄 Scraping and analyzing... This may take 10-20 seconds per URL."):
            progress_bar = st.progress(0.0)
            results = []
            urls_to_process = [url] if analyze and url and not batch_mode else batch_urls
            
            for idx, process_url in enumerate(urls_to_process):
                # Fixed Progress: Use fraction 0-1 per step (scrape ~0.3, analyze ~0.7 of per-URL progress)
                url_progress = (idx / len(urls_to_process)) + (0.3 / len(urls_to_process))  # Incremental for scrape
                progress_bar.progress(min(url_progress, 1.0))
                
                text, title = extract_content(process_url)  # Cached
                
                analyze_progress = (idx / len(urls_to_process)) + (0.7 / len(urls_to_process))  # Incremental for analyze
                progress_bar.progress(min(analyze_progress, 1.0))
                
                if text and len(text.strip()) > 50:  # Valid content check
                    text = text[:max_words]  # Truncate
                    wc, read = get_features(text)
                    keywords = extract_keywords(text)
                    semantics = extract_semantics(text)  # Updated: NLTK-based
                    features = np.array([[wc, read]])
                    probs = model.predict_proba(features)[0]
                    quality_idx = np.argmax(probs)
                    quality_map = ["Low", "Medium", "High"]
                    quality = quality_map[quality_idx]
                    thin = wc < 500
                    delta = "🚀 Excellent!" if quality == "High" else "⚠️ Improve" if quality == "Low" else "✅ Solid"
                    results.append({"url": process_url, "title": title, "wc": wc, "read": read, "keywords": keywords, "semantics": semantics, "quality": quality, "thin": thin, "delta": delta})
                else:
                    st.warning(f"⚠️ Skipped {process_url}: No valid content extracted (too short or scrape failed).")
            
            # Final progress
            progress_bar.progress(1.0)
            st.success("✅ Analysis Complete!")
            
            if not results:
                st.error("❌ No valid results. Check URLs and try again.")
            elif batch_mode and len(results) > 1:
                # Medium: Batch table comparison
                df = pd.DataFrame(results)
                st.markdown("### 📊 Batch Comparison Table")
                st.table(df[["url", "title", "wc", "read", "quality", "thin"]].round(1))
            else:
                # Single analysis
                result = results[0]
                wc, read, keywords, semantics, quality, thin, delta = result["wc"], result["read"], result["keywords"], result["semantics"], result["quality"], result["thin"], result["delta"]
                title = result["title"]
                
                # Metrics row (responsive columns)
                with st.container():  # Easy: Responsive
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown("<h3>📊 Word Count</h3>", unsafe_allow_html=True)
                        st.metric("Words", wc, delta, delta_color="normal")
                    with col_b:
                        st.markdown("<h3>📖 Readability</h3>", unsafe_allow_html=True)
                        color_class = "low" if read < 30 else "medium" if read < 60 else "high"
                        st.markdown(f'<p class="{color_class}">Flesch Score: {read:.1f}</p>', unsafe_allow_html=True)
                        st.caption("*Higher = Easier to read (60-70 ideal for web)*")
                    with col_c:
                        st.markdown("<h3>⭐ Overall Quality</h3>", unsafe_allow_html=True)
                        color_class = "low" if quality == "Low" else "medium" if quality == "Medium" else "high"
                        st.markdown(f'<p class="{color_class}">{quality}</p>', unsafe_allow_html=True)
                        st.caption("*AI prediction via rules/RF model*")
                
                # Medium: Progress Metrics with Charts (pie for weights)
                st.markdown("### 📈 Quality Score Breakdown")
                wc_contrib = 0.3 * min(wc / 1000, 1) * 100
                read_contrib = 0.7 * (read / 100) * 100
                fig = px.pie(values=[wc_contrib, read_contrib], names=["Word Count (30%)", "Readability (70%)"], 
                             title=f"Contributions to {quality} Score (Total: {wc_contrib + read_contrib:.0f}%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Keywords section
                with st.expander("🔑 Extracted Keywords (Top 5)", expanded=False):
                    if keywords:
                        st.write("**Key Terms:** " + " | ".join(keywords))
                        st.caption(f"Based on frequency in {wc} words.")
                    else:
                        st.info("No significant keywords found.")
                
                # Medium: New Semantics Expander
                with st.expander("🔬 Semantic Terms (NLP Insights)", expanded=False):
                    if semantics:
                        st.write("**Related Nouns:** " + " | ".join(semantics))
                        st.caption("Extracted via NLTK for topical authority (e.g., LSI keywords).")
                    else:
                        st.info("No semantics found.")
                
                # Thin content alert
                if thin:
                    st.warning("⚠️ **Thin Content Alert:** Under 500 words—may rank poorly in search engines like Google. Add more value!")
                
                # Recommendations (unchanged)
                st.markdown("### 💡 Quick Recommendations")
                recs = []
                if wc < 500: recs.append("💡 Increase content length to 800+ words.")
                if read < 60: recs.append("💡 Simplify language for better readability (aim 60-70).")
                if quality == "Low": recs.append("💡 Optimize keywords and structure.")
                if not recs: recs.append("💡 Great job—content is SEO-ready!")
                for rec in recs:
                    st.markdown(f"- {rec}")
                
                # Raw JSON
                if show_raw:
                    with st.expander("📄 Raw Analysis JSON", expanded=False):
                        output = {
                            "url": url,
                            "title": title,
                            "word_count": wc,
                            "readability_score": read,
                            "quality_prediction": quality,
                            "is_thin_content": thin,
                            "top_keywords": keywords[:5],
                            "semantics": semantics,
                            "recommendations": recs,
                            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.json(output)
                
                # Easy: PDF/CSV Export
                st.markdown("### 📥 Export Report")
                df_export = pd.DataFrame({
                    "Metric": ["Word Count", "Readability", "Quality", "Thin Content", "Top Keywords"],
                    "Value": [wc, f"{read:.1f}", quality, thin, " | ".join(keywords[:3])]
                })
                csv = df_export.to_csv(index=False)
                st.download_button("📥 Download CSV Audit", csv, "seo_audit.csv", "text/csv")


# Basic history (Easy: Session state; expands if needed)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results' in locals() and results and len(results) > 0:  # Safe append only if valid results
    result = results[0]
    st.session_state.history.append({"url": result['url'], "quality": result['quality'], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
with st.sidebar.expander("📋 Analysis History"):
    for h in st.session_state.history[-5:]:  # Last 5
        st.write(f"- {h['url']}: {h['quality']} ({h['timestamp']})")


# Footer
st.markdown("---")
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("*Powered by Streamlit & AI | For SEO optimization demos.*")
with col_right:
    st.markdown("*© 2025 Mohammed Hanees M*")
