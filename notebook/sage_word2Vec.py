# pip install pandas numpy gensim scikit-learn
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re


class POIRecommenderWord2Vec:
    def __init__(self):
        self.df = None
        self.word2vec_model = None
        self.category_vectors = None
        self.vector_size = 100
        self.window = 5
        self.min_count = 1

    def clean_categories(self, categories):
        if isinstance(categories, str):
            categories = categories.replace('"', '').replace('[', '').replace(']', '')
            categories = re.sub(r'[^\w\s;]', '', categories)
            categories = categories.split(';') if ';' in categories else [categories]
            categories = [cat.strip().lower() for cat in categories if cat.strip()]
            return categories
        return []

    def get_category_vector(self, categories):
        vectors = []
        for category in categories:
            words = category.split()
            word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def train(self, csv_path):
        self.df = pd.read_csv(csv_path)
        category_lists = [
            [word for cat in self.clean_categories(categories) for word in cat.split()]
            for categories in self.df['categories']
        ]
        self.word2vec_model = Word2Vec(sentences=category_lists, vector_size=self.vector_size, window=self.window,
                                       min_count=self.min_count, workers=4)
        self.category_vectors = np.array([
            self.get_category_vector(self.clean_categories(categories)) for categories in self.df['categories']
        ])
        print(f"Model trained on {len(self.df)} points of interest")

    def get_recommendations(self, interests, n_recommendations=5):
        if self.word2vec_model is None:
            raise ValueError("Model needs to be trained.")
        clean_interests = self.clean_categories(interests)
        if not clean_interests:
            raise ValueError("No valid interests provided.")
        interest_vector = self.get_category_vector(clean_interests).reshape(1, -1)
        similarity_scores = cosine_similarity(interest_vector, self.category_vectors)[0]
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        return [
            {'name': self.df.iloc[idx]['name'],
             'categories': self.df.iloc[idx]['categories'],
             'similarity_score': similarity_scores[idx],
             'latitude': self.df.iloc[idx]['latitude_radian'],
             'longitude': self.df.iloc[idx]['longitude_radian']}
            for idx in top_indices
        ]
