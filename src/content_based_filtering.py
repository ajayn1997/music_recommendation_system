"""
Music Recommendation System - Content-Based Filtering
Recommends songs based on audio features and user preferences
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self):
        """Initialize content-based recommender"""
        self.scaler = StandardScaler()
        self.genre_encoder = LabelEncoder()
        self.feature_matrix = None
        self.songs_df = None
        self.user_profiles = {}
        self.rating_predictor = None
        self.is_fitted = False
        
        # Audio features to use for similarity
        self.audio_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness'
        ]
        
        # Additional features
        self.additional_features = ['popularity', 'duration_ms', 'release_year']
    
    def prepare_features(self, songs_df):
        """Prepare feature matrix from song data"""
        features_df = songs_df.copy()
        
        # Encode categorical features
        features_df['genre_encoded'] = self.genre_encoder.fit_transform(features_df['genre'])
        
        # Normalize release year
        features_df['release_year_norm'] = (features_df['release_year'] - features_df['release_year'].min()) / \
                                          (features_df['release_year'].max() - features_df['release_year'].min())
        
        # Normalize duration
        features_df['duration_norm'] = (features_df['duration_ms'] - features_df['duration_ms'].min()) / \
                                      (features_df['duration_ms'].max() - features_df['duration_ms'].min())
        
        # Select features for similarity calculation
        feature_columns = self.audio_features + ['popularity', 'genre_encoded', 'release_year_norm', 'duration_norm']
        feature_matrix = features_df[feature_columns].values
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix_scaled, features_df
    
    def fit(self, songs_df, interactions_df):
        """Train the content-based model"""
        print("Training content-based filtering model...")
        
        self.songs_df = songs_df.copy()
        
        # Prepare feature matrix
        self.feature_matrix, self.processed_songs = self.prepare_features(songs_df)
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        # Build user profiles
        self._build_user_profiles(interactions_df)
        
        # Train rating predictor
        self._train_rating_predictor(interactions_df)
        
        self.is_fitted = True
        print("Content-based model training completed!")
    
    def _build_user_profiles(self, interactions_df):
        """Build user profiles based on their listening history"""
        print("Building user profiles...")
        
        # Merge interactions with song features
        interaction_features = interactions_df.merge(
            self.processed_songs, 
            left_on='song_id', 
            right_on='song_id'
        )
        
        # Build profile for each user
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interaction_features[interaction_features['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                continue
            
            # Weight features by rating and listening duration
            weights = user_interactions['rating'] * (user_interactions['completion_rate'] + 0.1)
            
            # Calculate weighted average of features
            feature_columns = self.audio_features + ['popularity', 'genre_encoded', 'release_year_norm', 'duration_norm']
            
            profile = {}
            for feature in feature_columns:
                if feature in user_interactions.columns:
                    weighted_avg = np.average(user_interactions[feature], weights=weights)
                    profile[feature] = weighted_avg
            
            # Calculate genre preferences
            genre_preferences = user_interactions.groupby('genre').agg({
                'rating': 'mean',
                'song_id': 'count'
            }).rename(columns={'song_id': 'count'})
            
            # Weight by both rating and frequency
            genre_preferences['score'] = genre_preferences['rating'] * np.log(1 + genre_preferences['count'])
            profile['genre_preferences'] = genre_preferences['score'].to_dict()
            
            # Calculate temporal preferences (recent vs old music)
            current_year = 2024
            user_interactions['age'] = current_year - user_interactions['release_year']
            age_preferences = user_interactions.groupby(pd.cut(user_interactions['age'], 
                                                             bins=[0, 5, 15, 30, 100], 
                                                             labels=['Recent', 'Modern', 'Classic', 'Vintage']))['rating'].mean()
            profile['age_preferences'] = age_preferences.to_dict()
            
            self.user_profiles[user_id] = profile
        
        print(f"Built profiles for {len(self.user_profiles)} users")
    
    def _train_rating_predictor(self, interactions_df):
        """Train a model to predict ratings based on user-song feature similarity"""
        print("Training rating predictor...")
        
        # Prepare training data
        training_data = []
        
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            song_id = interaction['song_id']
            rating = interaction['rating']
            
            if user_id not in self.user_profiles:
                continue
            
            # Get song features
            song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index
            if len(song_idx) == 0:
                continue
            
            song_features = self.feature_matrix[song_idx[0]]
            
            # Get user profile
            user_profile = self.user_profiles[user_id]
            
            # Calculate similarity features
            feature_columns = self.audio_features + ['popularity', 'genre_encoded', 'release_year_norm', 'duration_norm']
            user_feature_vector = [user_profile.get(feature, 0) for feature in feature_columns]
            
            # Cosine similarity between user profile and song
            similarity = cosine_similarity([user_feature_vector], [song_features])[0][0]
            
            # Additional features
            song_info = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0]
            genre_preference = user_profile['genre_preferences'].get(song_info['genre'], 0)
            
            # Combine features
            features = list(song_features) + [similarity, genre_preference]
            training_data.append(features + [rating])
        
        # Convert to DataFrame
        feature_names = (self.audio_features + ['popularity', 'genre_encoded', 'release_year_norm', 'duration_norm'] + 
                        ['user_similarity', 'genre_preference'])
        
        training_df = pd.DataFrame(training_data, columns=feature_names + ['rating'])
        
        # Train Random Forest model
        X = training_df[feature_names]
        y = training_df['rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rating_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rating_predictor.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.rating_predictor.score(X_train, y_train)
        test_score = self.rating_predictor.score(X_test, y_test)
        print(f"Rating predictor - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    def predict_rating(self, user_id, song_id):
        """Predict rating for a user-song pair"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle new users
        if user_id not in self.user_profiles:
            return self._predict_for_new_user(song_id)
        
        # Get song features
        song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index
        if len(song_idx) == 0:
            return 3.0  # Default rating
        
        song_features = self.feature_matrix[song_idx[0]]
        
        # Get user profile
        user_profile = self.user_profiles[user_id]
        
        # Calculate features for prediction
        feature_columns = self.audio_features + ['popularity', 'genre_encoded', 'release_year_norm', 'duration_norm']
        user_feature_vector = [user_profile.get(feature, 0) for feature in feature_columns]
        
        # Cosine similarity
        similarity = cosine_similarity([user_feature_vector], [song_features])[0][0]
        
        # Genre preference
        song_info = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0]
        genre_preference = user_profile['genre_preferences'].get(song_info['genre'], 0)
        
        # Combine features
        features = list(song_features) + [similarity, genre_preference]
        
        # Predict rating
        predicted_rating = self.rating_predictor.predict([features])[0]
        return max(1, min(5, predicted_rating))
    
    def _predict_for_new_user(self, song_id):
        """Predict rating for new user based on song popularity"""
        song_info = self.songs_df[self.songs_df['song_id'] == song_id]
        if len(song_info) == 0:
            return 3.0
        
        # Use popularity as a proxy for new user preference
        popularity = song_info.iloc[0]['popularity']
        # Convert popularity (0-100) to rating (1-5)
        rating = 1 + (popularity / 100) * 4
        return rating
    
    def get_similar_songs(self, song_id, n_songs=10):
        """Get songs similar to a given song based on audio features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar songs")
        
        # Get song features
        song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index
        if len(song_idx) == 0:
            return []
        
        song_features = self.feature_matrix[song_idx[0]]
        
        # Calculate similarities with all songs
        similarities = cosine_similarity([song_features], self.feature_matrix)[0]
        
        # Get top similar songs (excluding the song itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_songs+1]
        
        similar_songs = []
        for idx in similar_indices:
            similar_song_id = self.songs_df.iloc[idx]['song_id']
            similarity_score = similarities[idx]
            similar_songs.append((similar_song_id, similarity_score))
        
        return similar_songs
    
    def recommend_songs(self, user_id, n_recommendations=10, exclude_rated=True):
        """Recommend songs for a user based on content"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get all songs
        all_songs = self.songs_df['song_id'].tolist()
        
        if exclude_rated and user_id in self.user_profiles:
            # This would require interaction data to exclude rated songs
            # For now, we'll recommend from all songs
            pass
        
        # Predict ratings for all songs
        predictions = []
        for song_id in all_songs:
            predicted_rating = self.predict_rating(user_id, song_id)
            predictions.append((song_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [song_id for song_id, _ in predictions[:n_recommendations]]
    
    def recommend_by_genre(self, user_id, genre, n_recommendations=10):
        """Recommend songs from a specific genre"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Filter songs by genre
        genre_songs = self.songs_df[self.songs_df['genre'] == genre]['song_id'].tolist()
        
        if len(genre_songs) == 0:
            return []
        
        # Predict ratings for genre songs
        predictions = []
        for song_id in genre_songs:
            predicted_rating = self.predict_rating(user_id, song_id)
            predictions.append((song_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [song_id for song_id, _ in predictions[:n_recommendations]]
    
    def get_user_profile_summary(self, user_id):
        """Get a summary of user's music preferences"""
        if user_id not in self.user_profiles:
            return "User profile not found"
        
        profile = self.user_profiles[user_id]
        
        summary = {
            'top_genres': sorted(profile['genre_preferences'].items(), 
                               key=lambda x: x[1], reverse=True)[:5],
            'audio_preferences': {
                'danceability': profile.get('danceability', 0),
                'energy': profile.get('energy', 0),
                'valence': profile.get('valence', 0),
                'acousticness': profile.get('acousticness', 0)
            },
            'era_preferences': profile.get('age_preferences', {})
        }
        
        return summary
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'scaler': self.scaler,
            'genre_encoder': self.genre_encoder,
            'feature_matrix': self.feature_matrix,
            'songs_df': self.songs_df,
            'user_profiles': self.user_profiles,
            'rating_predictor': self.rating_predictor,
            'processed_songs': self.processed_songs,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Content-based model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.genre_encoder = model_data['genre_encoder']
        self.feature_matrix = model_data['feature_matrix']
        self.songs_df = model_data['songs_df']
        self.user_profiles = model_data['user_profiles']
        self.rating_predictor = model_data['rating_predictor']
        self.processed_songs = model_data['processed_songs']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Content-based model loaded from {filepath}")

def main():
    """Demo of content-based recommender"""
    # Load data
    songs_df = pd.read_csv('../data/songs.csv')
    interactions_df = pd.read_csv('../data/interactions.csv')
    
    print("=== Content-Based Filtering Demo ===")
    print(f"Loaded {len(songs_df)} songs and {len(interactions_df)} interactions")
    
    # Initialize and train model
    recommender = ContentBasedRecommender()
    recommender.fit(songs_df, interactions_df)
    
    # Test recommendations for a sample user
    test_user = interactions_df['user_id'].iloc[0]
    recommendations = recommender.recommend_songs(test_user, n_recommendations=5)
    
    print(f"\nRecommendations for User {test_user}:")
    for i, song_id in enumerate(recommendations, 1):
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        predicted_rating = recommender.predict_rating(test_user, song_id)
        print(f"{i}. {song_info['title']} by {song_info['artist']} "
              f"(Genre: {song_info['genre']}, Predicted Rating: {predicted_rating:.2f})")
    
    # Show user profile
    print(f"\nUser {test_user} Profile Summary:")
    profile_summary = recommender.get_user_profile_summary(test_user)
    print(f"Top Genres: {profile_summary['top_genres'][:3]}")
    print(f"Audio Preferences: {profile_summary['audio_preferences']}")
    
    # Test similar songs
    test_song = songs_df.iloc[0]['song_id']
    similar_songs = recommender.get_similar_songs(test_song, n_songs=3)
    print(f"\nSongs similar to '{songs_df.iloc[0]['title']}':")
    for song_id, similarity in similar_songs:
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        print(f"- {song_info['title']} by {song_info['artist']} (Similarity: {similarity:.3f})")
    
    # Save model
    recommender.save_model('../models/content_based_filtering.pkl')

if __name__ == "__main__":
    main()

