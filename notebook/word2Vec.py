# pip install pandas numpy gensim scikit-learn
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re

class POIRecommender:
    def __init__(self):
        # void++
        self.df = None
        self.word2vec_model = None
        self.category_vectors = None
        # hyperparameters for word2vec
        self.vector_size = 100
        self.window = 5
        self.min_count = 1
        
    def clean_categories(self, categories):
        if isinstance(categories, str):
            # clean stuff (now with more cleaning)
            categories = categories.replace('"', '').replace('[', '').replace(']', '')
            categories = re.sub(r'[^\w\s;]', '', categories)  # remove special chars
            # split stuff
            categories = categories.split(';') if ';' in categories else [categories]
            # strip spaces, lowercase it, keep it together (unlike my life)
            categories = [cat.strip().lower() for cat in categories if cat.strip()]
            return categories  # return as list this time
        return []

    def get_category_vector(self, categories):
        # get vector for each word and average them
        vectors = []
        for category in categories:
            # split into individual words
            words = category.split()
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            if word_vectors:
                # average the word vectors for this category
                vectors.append(np.mean(word_vectors, axis=0))
        
        # if we got any vectors, average them
        if vectors:
            return np.mean(vectors, axis=0)
        # otherwise return zero vector
        return np.zeros(self.vector_size)

    def train(self, csv_path):
        # read csv, error checking is for babies
        self.df = pd.read_csv(csv_path)
        
        # create a list of lists of categories for training
        category_lists = []
        for categories in self.df['categories']:
            clean_cats = self.clean_categories(categories)
            # split each category into words for better training
            words = []
            for cat in clean_cats:
                words.extend(cat.split())
            category_lists.append(words)
            
        # train word2vec model 
        self.word2vec_model = Word2Vec(
            sentences=category_lists,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        
        # pre-compute category vectors for all POIs
        self.category_vectors = []
        for categories in self.df['categories']:
            clean_cats = self.clean_categories(categories)
            vector = self.get_category_vector(clean_cats)
            self.category_vectors.append(vector)
        
        # convert to numpy array for faster similarity computation
        self.category_vectors = np.array(self.category_vectors)
        
        print(f"Model trained on {len(self.df)} points of interest")

        
        # print("\nSome example similar categories:")
        # try:
        #     sample_word = next(word for word in self.word2vec_model.wv.index_to_key 
        #             if word in ['restaurant', 'cafe', 'museum', 'park'])
        #     similar_words = self.word2vec_model.wv.most_similar(sample_word, topn=3)
        #     print(f"Words similar to '{sample_word}':")
        #     for word, score in similar_words:
        #         print(f"  - {word}: {score:.3f}")
        # except StopIteration:
        #     pass
        
    def get_recommendations(self, interests, n_recommendations=5):
        # stop if u didn't train first. pls
        if self.word2vec_model is None:
            raise ValueError("Model needs to be trained dumb dumb")
            
        # clean user input and get vector
        clean_interests = self.clean_categories(interests)
        if not clean_interests:
            raise ValueError("No valid interests provided")
            
        interest_vector = self.get_category_vector(clean_interests)
        
        # reshape for sklearn
        interest_vector = interest_vector.reshape(1, -1)
        category_vectors = self.category_vectors
        
        # calculate similarity between interest vector and all POI vectors
        similarity_scores = cosine_similarity(interest_vector, category_vectors)[0]
        
        # get top N indices
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        
        # prepare recommendations
        recommendations = []
        for idx in top_indices:
            poi = self.df.iloc[idx]
            recommendations.append({
                'name': poi['name'],
                'categories': poi['categories'],
                'similarity_score': similarity_scores[idx],
                'latitude': poi['latitude_radian'],
                'longitude': poi['longitude_radian']
            })
            
        return recommendations
    
    def print_recommendations(self, recommendations):
        print("\nRecommended Points of Interest:")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"Categories: {rec['categories']}")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Location: ({rec['latitude']:.6f}, {rec['longitude']:.6f})")
            print("-" * 50)

# everything explodes here
if __name__ == "__main__":
    recommender = POIRecommender()
    # let's train this bad boy
    recommender.train('poiTrainingData.csv')
    # get user interests (now with semantic understanding!)
    userInterests = input("What are you interested in? = ")
    # plz work TT
    recommendations = recommender.get_recommendations(userInterests, n_recommendations=3)
    recommender.print_recommendations(recommendations)