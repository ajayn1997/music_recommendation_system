"""
Music Recommendation System - Hybrid Recommender
Combines collaborative filtering and content-based filtering for improved recommendations
"""

import pandas as pd
import numpy as np
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender
import pickle
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        """
        Initialize hybrid recommender
        
        Args:
            cf_weight (float): Weight for collaborative filtering (0-1)
            cb_weight (float): Weight for content-based filtering (0-1)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        # Ensure weights sum to 1
        total_weight = cf_weight + cb_weight
        self.cf_weight = cf_weight / total_weight
        self.cb_weight = cb_weight / total_weight
        
        # Initialize component recommenders
        self.cf_recommender = CollaborativeFilteringRecommender(method='matrix_factorization')
        self.cb_recommender = ContentBasedRecommender()
        
        self.is_fitted = False
        
    def fit(self, songs_df, interactions_df):
        """Train both collaborative and content-based models"""
        print("Training hybrid recommender...")
        print(f"Weights: CF={self.cf_weight:.2f}, CB={self.cb_weight:.2f}")
        
        # Train collaborative filtering model
        print("\n1. Training Collaborative Filtering component...")
        self.cf_recommender.fit(interactions_df)
        
        # Train content-based model
        print("\n2. Training Content-Based component...")
        self.cb_recommender.fit(songs_df, interactions_df)
        
        self.is_fitted = True
        print("\nHybrid model training completed!")
    
    def predict_rating(self, user_id, song_id):
        """Predict rating using hybrid approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from both models
        cf_prediction = self.cf_recommender.predict_rating(user_id, song_id)
        cb_prediction = self.cb_recommender.predict_rating(user_id, song_id)
        
        # Combine predictions using weighted average
        hybrid_prediction = (self.cf_weight * cf_prediction + 
                           self.cb_weight * cb_prediction)
        
        return max(1, min(5, hybrid_prediction))
    
    def recommend_songs(self, user_id, n_recommendations=10, exclude_rated=True):
        """Generate hybrid recommendations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        cf_recommendations = self.cf_recommender.recommend_songs(
            user_id, n_recommendations=n_recommendations*2, exclude_rated=exclude_rated
        )
        cb_recommendations = self.cb_recommender.recommend_songs(
            user_id, n_recommendations=n_recommendations*2, exclude_rated=exclude_rated
        )
        
        # Combine and score all unique songs
        all_songs = set(cf_recommendations + cb_recommendations)
        
        song_scores = []
        for song_id in all_songs:
            # Get rank-based scores (higher rank = lower score)
            cf_score = 0
            if song_id in cf_recommendations:
                cf_rank = cf_recommendations.index(song_id) + 1
                cf_score = 1.0 / cf_rank
            
            cb_score = 0
            if song_id in cb_recommendations:
                cb_rank = cb_recommendations.index(song_id) + 1
                cb_score = 1.0 / cb_rank
            
            # Combine scores
            hybrid_score = self.cf_weight * cf_score + self.cb_weight * cb_score
            song_scores.append((song_id, hybrid_score))
        
        # Sort by hybrid score and return top N
        song_scores.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in song_scores[:n_recommendations]]
    
    def recommend_with_explanation(self, user_id, n_recommendations=10):
        """Generate recommendations with explanations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = self.recommend_songs(user_id, n_recommendations)
        
        explanations = []
        for song_id in recommendations:
            # Get predictions from both models
            cf_prediction = self.cf_recommender.predict_rating(user_id, song_id)
            cb_prediction = self.cb_recommender.predict_rating(user_id, song_id)
            hybrid_prediction = self.predict_rating(user_id, song_id)
            
            # Determine primary reason for recommendation
            if cf_prediction > cb_prediction:
                primary_reason = "collaborative"
                reason_text = "Users with similar taste also liked this song"
            else:
                primary_reason = "content"
                reason_text = "This song matches your music preferences"
            
            explanation = {
                'song_id': song_id,
                'hybrid_rating': hybrid_prediction,
                'cf_rating': cf_prediction,
                'cb_rating': cb_prediction,
                'primary_reason': primary_reason,
                'explanation': reason_text
            }
            explanations.append(explanation)
        
        return explanations
    
    def recommend_diverse(self, user_id, n_recommendations=10, diversity_factor=0.3):
        """Generate diverse recommendations by balancing accuracy and diversity"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get a larger pool of candidates
        candidates = self.recommend_songs(user_id, n_recommendations=n_recommendations*3)
        
        # Load song data for diversity calculation
        songs_df = self.cb_recommender.songs_df
        
        selected_songs = []
        selected_features = []
        
        for song_id in candidates:
            if len(selected_songs) >= n_recommendations:
                break
            
            # Get song features
            song_info = songs_df[songs_df['song_id'] == song_id]
            if len(song_info) == 0:
                continue
            
            song_features = song_info.iloc[0]
            
            # Calculate diversity score
            if len(selected_songs) == 0:
                diversity_score = 1.0  # First song is always diverse
            else:
                # Calculate average similarity to already selected songs
                similarities = []
                for selected_feature in selected_features:
                    # Simple diversity based on genre and audio features
                    genre_sim = 1.0 if song_features['genre'] == selected_feature['genre'] else 0.0
                    
                    audio_features = ['danceability', 'energy', 'valence', 'acousticness']
                    audio_sim = np.mean([
                        abs(song_features[feat] - selected_feature[feat]) 
                        for feat in audio_features
                    ])
                    
                    similarity = 0.5 * genre_sim + 0.5 * (1 - audio_sim)
                    similarities.append(similarity)
                
                diversity_score = 1 - np.mean(similarities)
            
            # Get accuracy score (predicted rating)
            accuracy_score = self.predict_rating(user_id, song_id) / 5.0
            
            # Combine accuracy and diversity
            final_score = (1 - diversity_factor) * accuracy_score + diversity_factor * diversity_score
            
            # Add song if it meets threshold or we need more songs
            if final_score > 0.5 or len(selected_songs) < n_recommendations // 2:
                selected_songs.append(song_id)
                selected_features.append(song_features)
        
        return selected_songs
    
    def get_recommendation_stats(self, user_id, recommendations):
        """Get statistics about the recommendations"""
        if not self.is_fitted:
            return {}
        
        songs_df = self.cb_recommender.songs_df
        
        # Get song information
        rec_songs = songs_df[songs_df['song_id'].isin(recommendations)]
        
        stats = {
            'total_recommendations': len(recommendations),
            'genre_distribution': rec_songs['genre'].value_counts().to_dict(),
            'avg_popularity': rec_songs['popularity'].mean(),
            'avg_energy': rec_songs['energy'].mean(),
            'avg_valence': rec_songs['valence'].mean(),
            'year_range': {
                'min': rec_songs['release_year'].min(),
                'max': rec_songs['release_year'].max(),
                'avg': rec_songs['release_year'].mean()
            }
        }
        
        return stats
    
    def compare_methods(self, user_id, n_recommendations=5):
        """Compare recommendations from different methods"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparison")
        
        # Get recommendations from each method
        cf_recs = self.cf_recommender.recommend_songs(user_id, n_recommendations)
        cb_recs = self.cb_recommender.recommend_songs(user_id, n_recommendations)
        hybrid_recs = self.recommend_songs(user_id, n_recommendations)
        
        songs_df = self.cb_recommender.songs_df
        
        comparison = {
            'collaborative_filtering': [],
            'content_based': [],
            'hybrid': []
        }
        
        for method, recs in [('collaborative_filtering', cf_recs), 
                           ('content_based', cb_recs), 
                           ('hybrid', hybrid_recs)]:
            for song_id in recs:
                song_info = songs_df[songs_df['song_id'] == song_id]
                if len(song_info) > 0:
                    song_data = song_info.iloc[0]
                    comparison[method].append({
                        'song_id': song_id,
                        'title': song_data['title'],
                        'artist': song_data['artist'],
                        'genre': song_data['genre'],
                        'predicted_rating': self.predict_rating(user_id, song_id)
                    })
        
        return comparison
    
    def adjust_weights(self, cf_weight, cb_weight):
        """Adjust the weights of collaborative and content-based components"""
        total_weight = cf_weight + cb_weight
        self.cf_weight = cf_weight / total_weight
        self.cb_weight = cb_weight / total_weight
        
        print(f"Updated weights: CF={self.cf_weight:.2f}, CB={self.cb_weight:.2f}")
    
    def save_model(self, filepath):
        """Save the hybrid model"""
        model_data = {
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'cf_recommender': self.cf_recommender,
            'cb_recommender': self.cb_recommender,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Hybrid model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a hybrid model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.cf_weight = model_data['cf_weight']
        self.cb_weight = model_data['cb_weight']
        self.cf_recommender = model_data['cf_recommender']
        self.cb_recommender = model_data['cb_recommender']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Hybrid model loaded from {filepath}")

def main():
    """Demo of hybrid recommender"""
    # Load data
    songs_df = pd.read_csv('../data/songs.csv')
    interactions_df = pd.read_csv('../data/interactions.csv')
    
    print("=== Hybrid Recommender Demo ===")
    print(f"Loaded {len(songs_df)} songs and {len(interactions_df)} interactions")
    
    # Initialize and train hybrid model
    hybrid_recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    hybrid_recommender.fit(songs_df, interactions_df)
    
    # Test recommendations for a sample user
    test_user = interactions_df['user_id'].iloc[0]
    
    print(f"\n=== Recommendations for User {test_user} ===")
    
    # Basic recommendations
    recommendations = hybrid_recommender.recommend_songs(test_user, n_recommendations=5)
    print("\nHybrid Recommendations:")
    for i, song_id in enumerate(recommendations, 1):
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        predicted_rating = hybrid_recommender.predict_rating(test_user, song_id)
        print(f"{i}. {song_info['title']} by {song_info['artist']} "
              f"(Genre: {song_info['genre']}, Rating: {predicted_rating:.2f})")
    
    # Recommendations with explanations
    print("\n=== Recommendations with Explanations ===")
    explanations = hybrid_recommender.recommend_with_explanation(test_user, n_recommendations=3)
    for i, exp in enumerate(explanations, 1):
        song_info = songs_df[songs_df['song_id'] == exp['song_id']].iloc[0]
        print(f"{i}. {song_info['title']} by {song_info['artist']}")
        print(f"   Hybrid Rating: {exp['hybrid_rating']:.2f} "
              f"(CF: {exp['cf_rating']:.2f}, CB: {exp['cb_rating']:.2f})")
        print(f"   Reason: {exp['explanation']}")
    
    # Diverse recommendations
    print("\n=== Diverse Recommendations ===")
    diverse_recs = hybrid_recommender.recommend_diverse(test_user, n_recommendations=5)
    for i, song_id in enumerate(diverse_recs, 1):
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        print(f"{i}. {song_info['title']} by {song_info['artist']} "
              f"(Genre: {song_info['genre']})")
    
    # Method comparison
    print("\n=== Method Comparison ===")
    comparison = hybrid_recommender.compare_methods(test_user, n_recommendations=3)
    
    for method, recs in comparison.items():
        print(f"\n{method.replace('_', ' ').title()}:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['title']} by {rec['artist']} "
                  f"(Genre: {rec['genre']}, Rating: {rec['predicted_rating']:.2f})")
    
    # Recommendation statistics
    print("\n=== Recommendation Statistics ===")
    stats = hybrid_recommender.get_recommendation_stats(test_user, recommendations)
    print(f"Average Popularity: {stats['avg_popularity']:.1f}")
    print(f"Average Energy: {stats['avg_energy']:.2f}")
    print(f"Average Valence: {stats['avg_valence']:.2f}")
    print(f"Year Range: {stats['year_range']['min']}-{stats['year_range']['max']}")
    print(f"Top Genres: {list(stats['genre_distribution'].keys())[:3]}")
    
    # Save model
    hybrid_recommender.save_model('../models/hybrid_recommender.pkl')

if __name__ == "__main__":
    main()

