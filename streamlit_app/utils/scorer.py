import json
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from .features import extract_features_for_url
from .parser import parse_html
import requests
import time
import textstat
from sentence_transformers import SentenceTransformer

# Load globals (for app)
model = joblib.load('../models/quality_model.pkl')  # Relative
features_df = pd.read_csv('../data/features.csv')
dataset_embeddings = np.array([json.loads(e) for e in features_df['embedding']])

def analyze_url(url, delay=1):
    # Full code as in Cell 6 main function
    pass  # Paste the def here, using globals
