# pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# pain :(
import re 

# Big recommender energy
class POIRecommender:
    def __init__(self):

        # void
        self.df = None
        self.tfidfMatrix = None
        self.vectorizer = None
        
    def clean_categories(self, categories):
        if isinstance(categories, str):
            # clean stuff
            categories = categories.replace('"', '').replace('[', '').replace(']', '')
            # split stuff
            categories = categories.split(';') if ';' in categories else [categories]
            # strip spaces, lowercase it, keep it together (unlike my life)
            categories = [cat.strip().lower() for cat in categories]
            return ' '.join(categories) # superglue
        return ''
        
    def train(self, csv_path):
        # read csv, error checking is for babies
        self.df = pd.read_csv(csv_path)

        # create a new column for 'processed_categories'
        self.df['processed_categories'] = self.df['categories'].apply(self.clean_categories)

        # I have no clue what is going on
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidfMatrix = self.vectorizer.fit_transform(self.df['processed_categories'])
        
        print(f"Model trained on {len(self.df)} points of interest") # yay?

    def get_recommendations(self, interests, n_recommendations=5):

        # stop if u didn’t train first. pls
        if self.tfidfMatrix is None:
            raise ValueError("Model needs to be trained dumb dumb")

        # clean user garbage
        interests = self.clean_categories(interests)

        # make it a vector
        interest_vector = self.vectorizer.transform([interests])

        # calculate similarity aka how close is this thing to ur random words
        similarity_scores = cosine_similarity(interest_vector, self.tfidfMatrix)

        # sort scores, highest first
        top_indices = similarity_scores[0].argsort()[-n_recommendations:][::-1]

        # tell the user how to live their life
        recommendations = []
        for idx in top_indices:
            poi = self.df.iloc[idx]
            recommendations.append({
                'name': poi['name'], 
                'categories': poi['categories'],
                'similarity_score': similarity_scores[0][idx],
                'latitude': poi['latitude_radian'],
                'longitude': poi['longitude_radian']
            })
            
        return recommendations
    
    def print_recommendations(self, recommendations):
        print("\nRecommended Points of Interest:")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):

            # output stuff
            print(f"\n{i}. {rec['name']}")
            print(f"Categories: {rec['categories']}")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Location: ({rec['latitude']:.6f}, {rec['longitude']:.6f})")
            print("-" * 50)

# everything explodes here
if __name__ == "__main__":
    recommender = POIRecommender()

    # rocky moment
    recommender.train('poiTrainingData.csv')
    # glaciers mountains nature
    userInterests = input("User Interests = ")
    recommendations = recommender.get_recommendations(userInterests, n_recommendations=3)
    recommender.print_recommendations(recommendations)
