"""
Music Recommendation System - Data Generator
Generates realistic sample data for the music recommendation tutorial
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MusicDataGenerator:
    def __init__(self):
        self.genres = [
            'Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 
            'Country', 'R&B', 'Indie', 'Alternative', 'Folk', 'Blues',
            'Reggae', 'Punk', 'Metal', 'Funk', 'Soul', 'Disco'
        ]
        
        self.artists = [
            'The Midnight Echoes', 'Luna Rodriguez', 'Digital Dreams', 'Jazz Collective',
            'Mountain Folk', 'Electric Storm', 'Sarah Chen', 'The Wanderers',
            'Neon Lights', 'Acoustic Soul', 'Urban Beats', 'Classical Ensemble',
            'Rock Legends', 'Pop Stars', 'Hip-Hop Masters', 'Electronic Vibes',
            'Country Roads', 'R&B Smooth', 'Indie Artists', 'Alternative Rock',
            'Folk Tales', 'Blues Brothers', 'Reggae Rhythms', 'Punk Revolution',
            'Metal Thunder', 'Funk Groove', 'Soul Sisters', 'Disco Fever',
            'Ambient Sounds', 'Experimental Music'
        ]
        
        self.song_adjectives = [
            'Beautiful', 'Dark', 'Bright', 'Mysterious', 'Energetic', 'Calm',
            'Wild', 'Gentle', 'Powerful', 'Dreamy', 'Intense', 'Peaceful',
            'Vibrant', 'Melancholic', 'Uplifting', 'Haunting', 'Joyful', 'Serene'
        ]
        
        self.song_nouns = [
            'Sunset', 'Ocean', 'Mountain', 'City', 'Dreams', 'Love', 'Journey',
            'Stars', 'Rain', 'Fire', 'Wind', 'Heart', 'Soul', 'Mind', 'Spirit',
            'Night', 'Day', 'Moon', 'Sun', 'River', 'Forest', 'Desert', 'Sky'
        ]
        
    def generate_songs(self, num_songs=1000):
        """Generate realistic song data"""
        songs = []
        
        for song_id in range(1, num_songs + 1):
            # Generate song title
            title = f"{random.choice(self.song_adjectives)} {random.choice(self.song_nouns)}"
            if random.random() < 0.3:  # 30% chance of adding a subtitle
                title += f" ({random.choice(['Remix', 'Acoustic Version', 'Live', 'Extended Mix'])})"
            
            # Select artist and genre
            artist = random.choice(self.artists)
            genre = random.choice(self.genres)
            
            # Generate audio features (similar to Spotify's audio features)
            song = {
                'song_id': song_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'duration_ms': np.random.normal(210000, 60000),  # ~3.5 minutes average
                'release_year': np.random.randint(1960, 2024),
                'popularity': np.random.beta(2, 5) * 100,  # Skewed towards lower popularity
                'danceability': np.random.beta(2, 2),
                'energy': np.random.beta(2, 2),
                'valence': np.random.beta(2, 2),  # Musical positivity
                'acousticness': np.random.beta(1, 3),
                'instrumentalness': np.random.beta(1, 9),  # Most songs have vocals
                'liveness': np.random.beta(1, 9),  # Most songs are studio recordings
                'speechiness': np.random.beta(1, 9),  # Most songs are not speech-like
                'tempo': np.random.normal(120, 30),  # BPM
                'loudness': np.random.normal(-8, 4),  # dB
            }
            
            # Ensure realistic ranges
            song['duration_ms'] = max(30000, min(600000, song['duration_ms']))  # 30s to 10min
            song['tempo'] = max(60, min(200, song['tempo']))
            song['loudness'] = max(-20, min(0, song['loudness']))
            
            songs.append(song)
        
        return pd.DataFrame(songs)
    
    def generate_users(self, num_users=500):
        """Generate user data with preferences"""
        users = []
        
        age_groups = {
            'Gen Z': (16, 25),
            'Millennial': (26, 40),
            'Gen X': (41, 55),
            'Boomer': (56, 75)
        }
        
        for user_id in range(1, num_users + 1):
            # Assign age group and corresponding music preferences
            age_group = np.random.choice(list(age_groups.keys()), 
                                       p=[0.3, 0.35, 0.25, 0.1])
            age_range = age_groups[age_group]
            age = np.random.randint(age_range[0], age_range[1] + 1)
            
            # Genre preferences based on age group
            if age_group == 'Gen Z':
                preferred_genres = np.random.choice(
                    ['Pop', 'Hip-Hop', 'Electronic', 'Indie', 'Alternative'], 
                    size=np.random.randint(2, 4), replace=False
                ).tolist()
            elif age_group == 'Millennial':
                preferred_genres = np.random.choice(
                    ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Indie', 'Alternative'], 
                    size=np.random.randint(2, 5), replace=False
                ).tolist()
            elif age_group == 'Gen X':
                preferred_genres = np.random.choice(
                    ['Rock', 'Pop', 'Country', 'R&B', 'Jazz', 'Blues'], 
                    size=np.random.randint(2, 4), replace=False
                ).tolist()
            else:  # Boomer
                preferred_genres = np.random.choice(
                    ['Rock', 'Jazz', 'Classical', 'Country', 'Blues', 'Folk'], 
                    size=np.random.randint(2, 4), replace=False
                ).tolist()
            
            user = {
                'user_id': user_id,
                'age': age,
                'age_group': age_group,
                'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR'], 
                                          p=[0.4, 0.15, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05]),
                'preferred_genres': preferred_genres,
                'listening_hours_per_day': np.random.gamma(2, 1.5),  # Average ~3 hours
                'discovery_factor': np.random.beta(2, 3),  # How much they explore new music
            }
            
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, songs_df, users_df, num_interactions=50000):
        """Generate user-song interaction data"""
        interactions = []
        
        # Create interaction types with different weights
        interaction_types = ['play', 'skip', 'like', 'dislike', 'playlist_add']
        interaction_weights = [0.6, 0.25, 0.08, 0.02, 0.05]
        
        for _ in range(num_interactions):
            user = users_df.sample(1).iloc[0]
            
            # Users are more likely to interact with songs from their preferred genres
            if random.random() < 0.7:  # 70% chance of genre preference influence
                preferred_songs = songs_df[songs_df['genre'].isin(user['preferred_genres'])]
                if len(preferred_songs) > 0:
                    song = preferred_songs.sample(1).iloc[0]
                else:
                    song = songs_df.sample(1).iloc[0]
            else:
                song = songs_df.sample(1).iloc[0]
            
            # Generate interaction type
            interaction_type = np.random.choice(interaction_types, p=interaction_weights)
            
            # Generate timestamp (last 2 years)
            start_date = datetime.now() - timedelta(days=730)
            random_days = np.random.randint(0, 730)
            timestamp = start_date + timedelta(days=random_days)
            
            # Generate listening duration (for play interactions)
            if interaction_type == 'play':
                # Users more likely to listen to full songs they like
                completion_rate = np.random.beta(2, 2)
                if song['genre'] in user['preferred_genres']:
                    completion_rate = np.random.beta(3, 1)  # Higher completion for preferred genres
                listening_duration = song['duration_ms'] * completion_rate
            else:
                listening_duration = 0
            
            # Generate rating (implicit from interaction type)
            rating_map = {'play': 3, 'skip': 1, 'like': 5, 'dislike': 1, 'playlist_add': 5}
            rating = rating_map[interaction_type]
            
            # Adjust rating based on listening duration for plays
            if interaction_type == 'play':
                if listening_duration / song['duration_ms'] > 0.8:
                    rating = 4  # Listened to most of the song
                elif listening_duration / song['duration_ms'] < 0.3:
                    rating = 2  # Skipped early
            
            interaction = {
                'user_id': user['user_id'],
                'song_id': song['song_id'],
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp,
                'listening_duration_ms': listening_duration,
                'completion_rate': listening_duration / song['duration_ms'] if song['duration_ms'] > 0 else 0
            }
            
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_all_data(self, num_songs=1000, num_users=500, num_interactions=50000):
        """Generate complete dataset"""
        print("Generating songs data...")
        songs_df = self.generate_songs(num_songs)
        
        print("Generating users data...")
        users_df = self.generate_users(num_users)
        
        print("Generating interactions data...")
        interactions_df = self.generate_interactions(songs_df, users_df, num_interactions)
        
        return songs_df, users_df, interactions_df

def main():
    """Generate and save sample data"""
    generator = MusicDataGenerator()
    
    # Generate data
    songs_df, users_df, interactions_df = generator.generate_all_data()
    
    # Save to CSV files
    songs_df.to_csv('../data/songs.csv', index=False)
    users_df.to_csv('../data/users.csv', index=False)
    interactions_df.to_csv('../data/interactions.csv', index=False)
    
    # Save user preferences as JSON for easier handling
    user_preferences = {}
    for _, user in users_df.iterrows():
        user_preferences[user['user_id']] = {
            'preferred_genres': user['preferred_genres'],
            'age_group': user['age_group'],
            'discovery_factor': user['discovery_factor']
        }
    
    with open('../data/user_preferences.json', 'w') as f:
        json.dump(user_preferences, f, indent=2)
    
    print(f"Generated data saved:")
    print(f"- Songs: {len(songs_df)} records")
    print(f"- Users: {len(users_df)} records") 
    print(f"- Interactions: {len(interactions_df)} records")
    print(f"- Average interactions per user: {len(interactions_df) / len(users_df):.1f}")
    print(f"- Average interactions per song: {len(interactions_df) / len(songs_df):.1f}")

if __name__ == "__main__":
    main()

