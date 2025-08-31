"""
Music Recommendation System - Collaborative Filtering
Implements user-based and item-based collaborative filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilteringRecommender:
    def __init__(self, method='user_based'):
        """
        Initialize collaborative filtering recommender
        
        Args:
            method (str): 'user_based', 'item_based', or 'matrix_factorization'
        """
        self.method = method
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_means = None
        self.is_fitted = False
        
    def create_user_item_matrix(self, interactions_df):
        """Create user-item interaction matrix"""
        # Use rating as the interaction strength
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='song_id', 
            values='rating', 
            fill_value=0
        )
        
        return user_item_matrix
    
    def fit(self, interactions_df):
        """Train the collaborative filtering model"""
        print(f"Training {self.method} collaborative filtering model...")
        
        # Create user-item matrix
        self.user_item_matrix = self.create_user_item_matrix(interactions_df)
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
        if self.method == 'user_based':
            self._fit_user_based()
        elif self.method == 'item_based':
            self._fit_item_based()
        elif self.method == 'matrix_factorization':
            self._fit_matrix_factorization()
        
        self.is_fitted = True
        print("Model training completed!")
    
    def _fit_user_based(self):
        """Fit user-based collaborative filtering"""
        # Calculate user means for mean-centered ratings
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Mean-center the ratings
        user_item_centered = self.user_item_matrix.sub(self.user_means, axis=0)
        user_item_centered = user_item_centered.fillna(0)
        
        # Calculate user similarity using cosine similarity
        self.user_similarity = cosine_similarity(user_item_centered)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
    
    def _fit_item_based(self):
        """Fit item-based collaborative filtering"""
        # Calculate item similarity using cosine similarity
        # Transpose to get items as rows
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
    
    def _fit_matrix_factorization(self):
        """Fit matrix factorization using SVD"""
        # Convert to sparse matrix for efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.svd_model.fit(sparse_matrix)
        
        # Calculate user means for prediction adjustment
        self.user_means = self.user_item_matrix.mean(axis=1)
    
    def predict_rating(self, user_id, song_id):
        """Predict rating for a user-song pair"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.method == 'user_based':
            return self._predict_user_based(user_id, song_id)
        elif self.method == 'item_based':
            return self._predict_item_based(user_id, song_id)
        elif self.method == 'matrix_factorization':
            return self._predict_matrix_factorization(user_id, song_id)
    
    def _predict_user_based(self, user_id, song_id):
        """Predict rating using user-based collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()  # Global average
        
        if song_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]  # User average
        
        # Get users who rated this song
        song_ratings = self.user_item_matrix[song_id]
        rated_users = song_ratings[song_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.user_means[user_id]
        
        # Get similarities with users who rated this song
        similarities = self.user_similarity.loc[user_id, rated_users]
        
        # Remove self-similarity and get top similar users
        similarities = similarities[similarities.index != user_id]
        top_similar = similarities.nlargest(20)  # Top 20 similar users
        
        if len(top_similar) == 0:
            return self.user_means[user_id]
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_user, similarity in top_similar.items():
            if similarity > 0:  # Only consider positive similarities
                user_rating = self.user_item_matrix.loc[similar_user, song_id]
                user_mean = self.user_means[similar_user]
                
                numerator += similarity * (user_rating - user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        return max(1, min(5, predicted_rating))  # Clamp between 1 and 5
    
    def _predict_item_based(self, user_id, song_id):
        """Predict rating using item-based collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if song_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        # Get songs rated by this user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_songs = user_ratings[user_ratings > 0].index
        
        if len(rated_songs) == 0:
            return self.user_item_matrix.mean().mean()
        
        # Get similarities with songs rated by this user
        similarities = self.item_similarity.loc[song_id, rated_songs]
        top_similar = similarities.nlargest(20)  # Top 20 similar songs
        
        if len(top_similar) == 0:
            return self.user_item_matrix.mean().mean()
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_song, similarity in top_similar.items():
            if similarity > 0:
                song_rating = self.user_item_matrix.loc[user_id, similar_song]
                numerator += similarity * song_rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_item_matrix.mean().mean()
        
        predicted_rating = numerator / denominator
        return max(1, min(5, predicted_rating))
    
    def _predict_matrix_factorization(self, user_id, song_id):
        """Predict rating using matrix factorization"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if song_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]
        
        # Get user and item indices
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        song_idx = self.user_item_matrix.columns.get_loc(song_id)
        
        # Transform user vector
        user_vector = self.user_item_matrix.iloc[user_idx:user_idx+1].values
        user_transformed = self.svd_model.transform(user_vector)
        
        # Get item vector from SVD components
        item_vector = self.svd_model.components_[:, song_idx]
        
        # Predict rating
        predicted_rating = np.dot(user_transformed[0], item_vector)
        predicted_rating += self.user_means[user_id]  # Add user bias
        
        return max(1, min(5, predicted_rating))
    
    def recommend_songs(self, user_id, n_recommendations=10, exclude_rated=True):
        """Recommend songs for a user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular songs
            return self._recommend_popular_songs(n_recommendations)
        
        # Get all songs
        all_songs = self.user_item_matrix.columns
        
        if exclude_rated:
            # Exclude songs already rated by the user
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_songs = user_ratings[user_ratings == 0].index
        else:
            unrated_songs = all_songs
        
        # Predict ratings for unrated songs
        predictions = []
        for song_id in unrated_songs:
            predicted_rating = self.predict_rating(user_id, song_id)
            predictions.append((song_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in predictions[:n_recommendations]]
    
    def _recommend_popular_songs(self, n_recommendations):
        """Recommend popular songs for new users"""
        # Calculate song popularity based on average rating and number of ratings
        song_stats = self.user_item_matrix.apply(lambda x: {
            'avg_rating': x[x > 0].mean() if len(x[x > 0]) > 0 else 0,
            'num_ratings': len(x[x > 0])
        }, axis=0)
        
        # Create popularity score (weighted by number of ratings)
        popularity_scores = []
        for song_id in song_stats.index:
            stats = song_stats[song_id]
            if stats['num_ratings'] >= 5:  # Minimum 5 ratings
                score = stats['avg_rating'] * np.log(1 + stats['num_ratings'])
                popularity_scores.append((song_id, score))
        
        # Sort by popularity score
        popularity_scores.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in popularity_scores[:n_recommendations]]
    
    def get_similar_users(self, user_id, n_users=10):
        """Get most similar users to a given user"""
        if not self.is_fitted or self.method != 'user_based':
            raise ValueError("User similarity only available for user-based method")
        
        if user_id not in self.user_similarity.index:
            return []
        
        similarities = self.user_similarity.loc[user_id]
        similarities = similarities[similarities.index != user_id]  # Exclude self
        top_similar = similarities.nlargest(n_users)
        
        return [(user, sim) for user, sim in top_similar.items()]
    
    def get_similar_songs(self, song_id, n_songs=10):
        """Get most similar songs to a given song"""
        if not self.is_fitted or self.method != 'item_based':
            raise ValueError("Item similarity only available for item-based method")
        
        if song_id not in self.item_similarity.index:
            return []
        
        similarities = self.item_similarity.loc[song_id]
        similarities = similarities[similarities.index != song_id]  # Exclude self
        top_similar = similarities.nlargest(n_songs)
        
        return [(song, sim) for song, sim in top_similar.items()]
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'method': self.method,
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'svd_model': self.svd_model,
            'user_means': self.user_means,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity = model_data['user_similarity']
        self.item_similarity = model_data['item_similarity']
        self.svd_model = model_data['svd_model']
        self.user_means = model_data['user_means']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")

def main():
    """Demo of collaborative filtering recommender"""
    # Load data
    interactions_df = pd.read_csv('../data/interactions.csv')
    songs_df = pd.read_csv('../data/songs.csv')
    
    print("=== Collaborative Filtering Demo ===")
    print(f"Loaded {len(interactions_df)} interactions")
    
    # Train different models
    methods = ['user_based', 'item_based', 'matrix_factorization']
    
    for method in methods:
        print(f"\n--- {method.replace('_', ' ').title()} Method ---")
        
        # Initialize and train model
        recommender = CollaborativeFilteringRecommender(method=method)
        recommender.fit(interactions_df)
        
        # Test recommendations for a sample user
        test_user = interactions_df['user_id'].iloc[0]
        recommendations = recommender.recommend_songs(test_user, n_recommendations=5)
        
        print(f"Recommendations for User {test_user}:")
        for i, song_id in enumerate(recommendations, 1):
            song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
            predicted_rating = recommender.predict_rating(test_user, song_id)
            print(f"{i}. {song_info['title']} by {song_info['artist']} "
                  f"(Genre: {song_info['genre']}, Predicted Rating: {predicted_rating:.2f})")
        
        # Save model
        model_path = f'../models/collaborative_filtering_{method}.pkl'
        recommender.save_model(model_path)

if __name__ == "__main__":
    main()

