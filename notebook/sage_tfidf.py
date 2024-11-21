# pip install pandas numpy scikit-learn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class POIRecommenderTFIDF:
    def __init__(self):
        self.df = None
        self.tfidfMatrix = None
        self.vectorizer = None

    def clean_categories(self, categories):
        if isinstance(categories, str):
            categories = categories.replace('"', '').replace('[', '').replace(']', '')
            categories = categories.split(';') if ';' in categories else [categories]
            categories = [cat.strip().lower() for cat in categories]
            return ' '.join(categories)
        return ''

    def train(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['processed_categories'] = self.df['categories'].apply(self.clean_categories)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidfMatrix = self.vectorizer.fit_transform(self.df['processed_categories'])
        print(f"Model trained on {len(self.df)} points of interest")

    def get_recommendations(self, interests, n_recommendations=5):
        if self.tfidfMatrix is None:
            raise ValueError("Model needs to be trained.")
        interests = self.clean_categories(interests)
        interest_vector = self.vectorizer.transform([interests])
        similarity_scores = cosine_similarity(interest_vector, self.tfidfMatrix)
        top_indices = similarity_scores[0].argsort()[-n_recommendations:][::-1]
        return [
            {'name': self.df.iloc[idx]['name'],
             'categories': self.df.iloc[idx]['categories'],
             'similarity_score': similarity_scores[0][idx],
             'latitude': self.df.iloc[idx]['latitude_radian'],
             'longitude': self.df.iloc[idx]['longitude_radian']}
            for idx in top_indices
        ]
