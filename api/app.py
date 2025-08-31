"""
Music Recommendation System - Flask API
Simple REST API for serving music recommendations
"""

from flask import Flask, request, jsonify
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_recommender import HybridRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender

app = Flask(__name__)

# Global variables for models and data
recommender = None
songs_df = None
users_df = None
interactions_df = None

def load_data():
    """Load data and models"""
    global songs_df, users_df, interactions_df
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    songs_df = pd.read_csv(os.path.join(data_dir, 'songs.csv'))
    users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
    
    print(f"Loaded {len(songs_df)} songs, {len(users_df)} users, {len(interactions_df)} interactions")

def initialize_recommender():
    """Initialize and train the recommender"""
    global recommender
    
    print("Initializing hybrid recommender...")
    recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    recommender.fit(songs_df, interactions_df)
    print("Recommender initialized successfully!")

@app.route('/')
def home():
    """API home page"""
    return jsonify({
        "message": "Music Recommendation System API",
        "version": "1.0",
        "endpoints": {
            "/recommend/<user_id>": "Get recommendations for a user",
            "/recommend/<user_id>/<int:n>": "Get N recommendations for a user",
            "/song/<song_id>": "Get song information",
            "/user/<user_id>": "Get user information",
            "/similar/<song_id>": "Get similar songs",
            "/predict/<user_id>/<song_id>": "Predict rating for user-song pair"
        }
    })

@app.route('/recommend/<int:user_id>')
@app.route('/recommend/<int:user_id>/<int:n>')
def get_recommendations(user_id, n=10):
    """Get recommendations for a user"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Get recommendations
        recommendations = recommender.recommend_songs(user_id, n_recommendations=n)
        
        # Get song details
        recommended_songs = []
        for song_id in recommendations:
            song_info = songs_df[songs_df['song_id'] == song_id]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                predicted_rating = recommender.predict_rating(user_id, song_id)
                
                recommended_songs.append({
                    "song_id": int(song_id),
                    "title": song['title'],
                    "artist": song['artist'],
                    "genre": song['genre'],
                    "release_year": int(song['release_year']),
                    "popularity": float(song['popularity']),
                    "predicted_rating": float(predicted_rating)
                })
        
        return jsonify({
            "user_id": user_id,
            "num_recommendations": len(recommended_songs),
            "recommendations": recommended_songs
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_with_explanation/<int:user_id>')
@app.route('/recommend_with_explanation/<int:user_id>/<int:n>')
def get_recommendations_with_explanation(user_id, n=5):
    """Get recommendations with explanations"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Get recommendations with explanations
        explanations = recommender.recommend_with_explanation(user_id, n_recommendations=n)
        
        # Get song details and format response
        recommendations_with_explanations = []
        for exp in explanations:
            song_info = songs_df[songs_df['song_id'] == exp['song_id']]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                
                recommendations_with_explanations.append({
                    "song_id": int(exp['song_id']),
                    "title": song['title'],
                    "artist": song['artist'],
                    "genre": song['genre'],
                    "hybrid_rating": float(exp['hybrid_rating']),
                    "cf_rating": float(exp['cf_rating']),
                    "cb_rating": float(exp['cb_rating']),
                    "primary_reason": exp['primary_reason'],
                    "explanation": exp['explanation']
                })
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations_with_explanations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/song/<int:song_id>')
def get_song_info(song_id):
    """Get information about a specific song"""
    try:
        song_info = songs_df[songs_df['song_id'] == song_id]
        
        if len(song_info) == 0:
            return jsonify({"error": "Song not found"}), 404
        
        song = song_info.iloc[0]
        
        # Get interaction statistics
        song_interactions = interactions_df[interactions_df['song_id'] == song_id]
        
        return jsonify({
            "song_id": int(song_id),
            "title": song['title'],
            "artist": song['artist'],
            "genre": song['genre'],
            "release_year": int(song['release_year']),
            "duration_ms": float(song['duration_ms']),
            "popularity": float(song['popularity']),
            "audio_features": {
                "danceability": float(song['danceability']),
                "energy": float(song['energy']),
                "valence": float(song['valence']),
                "acousticness": float(song['acousticness']),
                "instrumentalness": float(song['instrumentalness']),
                "liveness": float(song['liveness']),
                "speechiness": float(song['speechiness']),
                "tempo": float(song['tempo']),
                "loudness": float(song['loudness'])
            },
            "statistics": {
                "total_interactions": len(song_interactions),
                "average_rating": float(song_interactions['rating'].mean()) if len(song_interactions) > 0 else 0,
                "unique_listeners": int(song_interactions['user_id'].nunique()) if len(song_interactions) > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user/<int:user_id>')
def get_user_info(user_id):
    """Get information about a specific user"""
    try:
        user_info = users_df[users_df['user_id'] == user_id]
        
        if len(user_info) == 0:
            return jsonify({"error": "User not found"}), 404
        
        user = user_info.iloc[0]
        
        # Get user interaction statistics
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # Get listening history
        recent_songs = user_interactions.nlargest(10, 'timestamp')
        listening_history = []
        
        for _, interaction in recent_songs.iterrows():
            song_info = songs_df[songs_df['song_id'] == interaction['song_id']]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                listening_history.append({
                    "song_id": int(interaction['song_id']),
                    "title": song['title'],
                    "artist": song['artist'],
                    "genre": song['genre'],
                    "rating": int(interaction['rating']),
                    "timestamp": interaction['timestamp']
                })
        
        return jsonify({
            "user_id": int(user_id),
            "age": int(user['age']),
            "age_group": user['age_group'],
            "country": user['country'],
            "preferred_genres": user['preferred_genres'],
            "listening_hours_per_day": float(user['listening_hours_per_day']),
            "discovery_factor": float(user['discovery_factor']),
            "statistics": {
                "total_interactions": len(user_interactions),
                "unique_songs": int(user_interactions['song_id'].nunique()),
                "average_rating": float(user_interactions['rating'].mean()) if len(user_interactions) > 0 else 0,
                "favorite_genres": user_interactions.merge(songs_df, on='song_id')['genre'].value_counts().head(5).to_dict()
            },
            "recent_listening": listening_history
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similar/<int:song_id>')
@app.route('/similar/<int:song_id>/<int:n>')
def get_similar_songs(song_id, n=10):
    """Get songs similar to a given song"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Get similar songs using content-based approach
        similar_songs = recommender.cb_recommender.get_similar_songs(song_id, n_songs=n)
        
        # Get song details
        similar_songs_info = []
        for similar_song_id, similarity in similar_songs:
            song_info = songs_df[songs_df['song_id'] == similar_song_id]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                similar_songs_info.append({
                    "song_id": int(similar_song_id),
                    "title": song['title'],
                    "artist": song['artist'],
                    "genre": song['genre'],
                    "similarity_score": float(similarity)
                })
        
        # Get original song info
        original_song = songs_df[songs_df['song_id'] == song_id]
        original_info = {}
        if len(original_song) > 0:
            song = original_song.iloc[0]
            original_info = {
                "song_id": int(song_id),
                "title": song['title'],
                "artist": song['artist'],
                "genre": song['genre']
            }
        
        return jsonify({
            "original_song": original_info,
            "similar_songs": similar_songs_info
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<int:user_id>/<int:song_id>')
def predict_rating(user_id, song_id):
    """Predict rating for a user-song pair"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Get prediction
        predicted_rating = recommender.predict_rating(user_id, song_id)
        
        # Get component predictions
        cf_rating = recommender.cf_recommender.predict_rating(user_id, song_id)
        cb_rating = recommender.cb_recommender.predict_rating(user_id, song_id)
        
        # Get song and user info
        song_info = songs_df[songs_df['song_id'] == song_id]
        user_info = users_df[users_df['user_id'] == user_id]
        
        song_data = {}
        if len(song_info) > 0:
            song = song_info.iloc[0]
            song_data = {
                "title": song['title'],
                "artist": song['artist'],
                "genre": song['genre']
            }
        
        user_data = {}
        if len(user_info) > 0:
            user = user_info.iloc[0]
            user_data = {
                "age_group": user['age_group'],
                "preferred_genres": user['preferred_genres']
            }
        
        return jsonify({
            "user_id": user_id,
            "song_id": song_id,
            "predicted_rating": float(predicted_rating),
            "component_ratings": {
                "collaborative_filtering": float(cf_rating),
                "content_based": float(cb_rating)
            },
            "song_info": song_data,
            "user_info": user_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            "total_songs": len(songs_df),
            "total_users": len(users_df),
            "total_interactions": len(interactions_df),
            "genres": songs_df['genre'].value_counts().to_dict(),
            "avg_interactions_per_user": float(len(interactions_df) / len(users_df)),
            "avg_interactions_per_song": float(len(interactions_df) / len(songs_df)),
            "rating_distribution": interactions_df['rating'].value_counts().sort_index().to_dict(),
            "recommender_weights": {
                "collaborative_filtering": float(recommender.cf_weight) if recommender else 0,
                "content_based": float(recommender.cb_weight) if recommender else 0
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading data...")
    load_data()
    
    print("Initializing recommender...")
    initialize_recommender()
    
    print("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

