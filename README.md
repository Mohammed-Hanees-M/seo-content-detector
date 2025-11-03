# SEO Content Quality & Duplicate Detector

## Project Overview
This project builds a machine learning pipeline to analyze web content for SEO quality and detect duplicates, using pre-scraped HTML from cybersecurity/SEO sites. It extracts features like readability and keywords, flags thin/duplicate content, and scores quality with a Random Forest model, plus a real-time analysis function. A bonus Streamlit app provides an interactive demo.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/yourusername/seo-content-detector`
2. Install dependencies: `cd seo-content-detector && pip install -r requirements.txt`
3. Download dataset: Place `data.csv` in `data/` (Kaggle primary dataset).
4. Run notebook: `jupyter notebook notebooks/seo_pipeline.ipynb`

## Quick Start
Open `seo_pipeline.ipynb` in Jupyter/VS Code, run cells sequentially for parsing, features, duplicates, model training, and demo. For bonus app: `streamlit run streamlit_app/app.py` (deploy to Streamlit Cloud for URL).

## Deployed Streamlit URL
[Insert URL after deployment, e.g., https://your-app.streamlit.app]

## Key Decisions
- **Libraries**: BeautifulSoup for robust HTML parsing; SentenceTransformers for semantic embeddings; Random Forest for interpretable classification.
- **Parsing**: Target <p>/<article> tags to extract clean body text, handling errors by skipping malformed rows.
- **Similarity Threshold**: 0.80 for duplicates—balances SEO near-duplicates without over-flagging (based on TF-IDF/embedding norms).
- **Model**: Random Forest over Logistic for non-linear feature interactions; synthetic labels ensure clear evaluation.
- **Bonus Modularity**: Utils in Streamlit for reuse, enabling easy deployment.

## Results Summary
- Model Accuracy: [e.g., 0.78], F1: [e.g., 0.75] (vs. baseline 0.64).
- Duplicates Found: [e.g., 3 pairs].
- Sample Scores: High quality pages emphasize readability 50-70; 10% thin content.
- Total Analyzed: ~65 pages.

## Limitations
- Synthetic labels may not capture real SEO nuances (no human annotation).
- Embeddings limited to MiniLM; larger models could improve similarity but increase compute.
- No advanced scraping retries; assumes primary dataset availability.
