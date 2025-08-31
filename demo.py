#!/usr/bin/env python3
"""
Music Recommendation System Demo

This script demonstrates the key features of the music recommendation system
by running through data generation, model training, and recommendation generation.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def main():
    print_header("Music Recommendation System Demo")
    print("This demo showcases the complete recommendation system pipeline.")
    print("Starting demo at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Step 1: Generate Data
    print_section("Step 1: Generating Synthetic Dataset")
    
    try:
        from data_generator import MusicDataGenerator
        
        generator = MusicDataGenerator()
        
        # Check if data already exists
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        songs_file = os.path.join(data_dir, 'songs.csv')
        
        if os.path.exists(songs_file):
            print("✓ Dataset already exists, loading existing data...")
            songs_df = pd.read_csv(songs_file)
            users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
            interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
        else:
            print("Generating new synthetic dataset...")
            songs_df = generator.generate_songs(1000)
            users_df = generator.generate_users(500)
            interactions_df = generator.generate_interactions(songs_df, users_df, 50000)
            
            # Save data
            os.makedirs(data_dir, exist_ok=True)
            songs_df.to_csv(songs_file, index=False)
            users_df.to_csv(os.path.join(data_dir, 'users.csv'), index=False)
            interactions_df.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
            print("✓ Dataset generated and saved!")
        
        print(f"Dataset Summary:")
        print(f"  - Songs: {len(songs_df):,}")
        print(f"  - Users: {len(users_df):,}")
        print(f"  - Interactions: {len(interactions_df):,}")
        print(f"  - Genres: {songs_df['genre'].nunique()}")
        print(f"  - Average rating: {interactions_df['rating'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return
    
    # Step 2: Train Collaborative Filtering
    print_section("Step 2: Training Collaborative Filtering Models")
    
    try:
        from collaborative_filtering import CollaborativeFilteringRecommender
        
        # Train matrix factorization model (fastest for demo)
        cf_recommender = CollaborativeFilteringRecommender(method='matrix_factorization')
        cf_recommender.fit(interactions_df)
        print("✓ Collaborative filtering model trained!")
        
        # Test recommendations
        test_user = 1
        cf_recs = cf_recommender.recommend_songs(test_user, n_recommendations=5)
        print(f"\nCollaborative Filtering Recommendations for User {test_user}:")
        
        for i, song_id in enumerate(cf_recs, 1):
            song_info = songs_df[songs_df['song_id'] == song_id]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                rating = cf_recommender.predict_rating(test_user, song_id)
                print(f"  {i}. {song['title']} by {song['artist']} (Genre: {song['genre']}, Rating: {rating:.2f})")
        
    except Exception as e:
        print(f"❌ Error training collaborative filtering: {e}")
        return
    
    # Step 3: Train Content-Based Filtering
    print_section("Step 3: Training Content-Based Filtering Model")
    
    try:
        from content_based_filtering import ContentBasedRecommender
        
        cb_recommender = ContentBasedRecommender()
        cb_recommender.fit(songs_df, interactions_df)
        print("✓ Content-based filtering model trained!")
        
        # Test recommendations
        cb_recs = cb_recommender.recommend_songs(test_user, n_recommendations=5)
        print(f"\nContent-Based Recommendations for User {test_user}:")
        
        for i, song_id in enumerate(cb_recs, 1):
            song_info = songs_df[songs_df['song_id'] == song_id]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                rating = cb_recommender.predict_rating(test_user, song_id)
                print(f"  {i}. {song['title']} by {song['artist']} (Genre: {song['genre']}, Rating: {rating:.2f})")
        
        # Show user profile
        if test_user in cb_recommender.user_profiles:
            profile = cb_recommender.user_profiles[test_user]
            print(f"\nUser {test_user} Profile Summary:")
            if 'genre_preferences' in profile:
                top_genres = sorted(profile['genre_preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top Genres: {top_genres}")
            
            audio_prefs = {k: v for k, v in profile.items() if k in ['danceability', 'energy', 'valence', 'acousticness']}
            print(f"  Audio Preferences: {audio_prefs}")
        
    except Exception as e:
        print(f"❌ Error training content-based filtering: {e}")
        return
    
    # Step 4: Train Hybrid System
    print_section("Step 4: Training Hybrid Recommendation System")
    
    try:
        from hybrid_recommender import HybridRecommender
        
        hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
        hybrid.fit(songs_df, interactions_df)
        print("✓ Hybrid recommendation system trained!")
        
        # Test recommendations with explanations
        explanations = hybrid.recommend_with_explanation(test_user, n_recommendations=5)
        print(f"\nHybrid Recommendations with Explanations for User {test_user}:")
        
        for i, exp in enumerate(explanations, 1):
            song_info = songs_df[songs_df['song_id'] == exp['song_id']]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                print(f"  {i}. {song['title']} by {song['artist']} (Genre: {song['genre']})")
                print(f"     Hybrid Rating: {exp['hybrid_rating']:.2f} (CF: {exp['cf_rating']:.2f}, CB: {exp['cb_rating']:.2f})")
                print(f"     Reason: {exp['explanation']}")
        
    except Exception as e:
        print(f"❌ Error training hybrid system: {e}")
        return
    
    # Step 5: Evaluation
    print_section("Step 5: Model Evaluation")
    
    try:
        from evaluation import RecommenderEvaluator
        
        evaluator = RecommenderEvaluator()
        train_data, test_data = evaluator.split_data(interactions_df, test_size=0.2)
        
        print(f"Data split: {len(train_data):,} training, {len(test_data):,} test interactions")
        
        # Retrain on training data only
        hybrid_eval = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
        hybrid_eval.fit(songs_df, train_data)
        
        # Evaluate rating prediction
        rating_results = evaluator.evaluate_rating_prediction(hybrid_eval, test_data)
        print(f"\nRating Prediction Performance:")
        print(f"  RMSE: {rating_results['rmse']:.3f}")
        print(f"  MAE: {rating_results['mae']:.3f}")
        print(f"  Coverage: {rating_results['coverage']:.3f}")
        
        # Evaluate diversity
        diversity_results = evaluator.evaluate_diversity(hybrid_eval, user_ids=test_data['user_id'].unique()[:20])
        print(f"\nDiversity Metrics:")
        print(f"  Catalog Coverage: {diversity_results['catalog_coverage']:.3f}")
        print(f"  Average Genre Diversity: {diversity_results['avg_genre_diversity']:.3f}")
        print(f"  Average Feature Diversity: {diversity_results['avg_feature_diversity']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
    
    # Step 6: Similar Songs Demo
    print_section("Step 6: Similar Songs Discovery")
    
    try:
        # Pick a random song and find similar ones
        sample_song = songs_df.sample(1).iloc[0]
        similar_songs = cb_recommender.get_similar_songs(sample_song['song_id'], n_songs=5)
        
        print(f"Songs similar to '{sample_song['title']}' by {sample_song['artist']} ({sample_song['genre']}):")
        
        for song_id, similarity in similar_songs:
            song_info = songs_df[songs_df['song_id'] == song_id]
            if len(song_info) > 0:
                song = song_info.iloc[0]
                print(f"  • {song['title']} by {song['artist']} ({song['genre']}) - Similarity: {similarity:.3f}")
        
    except Exception as e:
        print(f"❌ Error finding similar songs: {e}")
    
    # Step 7: API Demo Information
    print_section("Step 7: REST API Information")
    
    print("To start the REST API server:")
    print("  1. cd api")
    print("  2. python app.py")
    print("  3. Visit http://localhost:5000 for API documentation")
    print("\nExample API calls:")
    print("  curl http://localhost:5000/recommend/1")
    print("  curl http://localhost:5000/recommend_with_explanation/1/5")
    print("  curl http://localhost:5000/similar/42")
    print("  curl http://localhost:5000/song/100")
    print("  curl http://localhost:5000/stats")
    
    print_header("Demo Complete!")
    print("✓ All components successfully demonstrated")
    print("✓ Models trained and saved")
    print("✓ Recommendations generated")
    print("✓ System ready for use")
    print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  - Explore the API endpoints")
    print("  - Modify the code to experiment with different approaches")
    print("  - Try the system with your own data")
    print("  - Read the comprehensive tutorial blog post")

if __name__ == "__main__":
    main()

