# ============================================
# MUSIC COMPATIBILITY ML - USING KAGGLE DATASET
# ============================================
# This file teaches you ML concepts using a real dataset
# No API needed - works offline with downloaded data
#
# Dataset: Spotify Tracks Dataset from Kaggle
# Download from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================

def load_data(filepath='dataset.csv'):
    """
    Load the Spotify dataset.
    
    Download dataset from Kaggle and save as 'dataset.csv'
    Or use the sample data below for testing.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} tracks from {filepath}")
        return df
    except FileNotFoundError:
        print("⚠️ Dataset not found. Using sample data...")
        return create_sample_data()


def create_sample_data():
    """
    Create sample data if Kaggle dataset not available.
    This simulates what the real dataset looks like.
    """
    np.random.seed(42)
    
    # Simulate different music styles
    data = []
    
    # Pop songs (high danceability, high valence, medium energy)
    for i in range(50):
        data.append({
            'track_name': f'Pop Song {i+1}',
            'artists': f'Pop Artist {i%10}',
            'track_genre': 'pop',
            'popularity': np.random.randint(60, 100),
            'danceability': np.random.uniform(0.6, 0.9),
            'energy': np.random.uniform(0.5, 0.8),
            'valence': np.random.uniform(0.5, 0.9),
            'acousticness': np.random.uniform(0.1, 0.4),
            'instrumentalness': np.random.uniform(0.0, 0.1),
            'tempo': np.random.uniform(100, 130)
        })
    
    # Rock songs (high energy, medium danceability)
    for i in range(50):
        data.append({
            'track_name': f'Rock Song {i+1}',
            'artists': f'Rock Artist {i%10}',
            'track_genre': 'rock',
            'popularity': np.random.randint(40, 80),
            'danceability': np.random.uniform(0.3, 0.6),
            'energy': np.random.uniform(0.7, 1.0),
            'valence': np.random.uniform(0.3, 0.7),
            'acousticness': np.random.uniform(0.1, 0.3),
            'instrumentalness': np.random.uniform(0.0, 0.3),
            'tempo': np.random.uniform(110, 150)
        })
    
    # Acoustic/Indie songs (high acousticness, lower energy)
    for i in range(50):
        data.append({
            'track_name': f'Indie Song {i+1}',
            'artists': f'Indie Artist {i%10}',
            'track_genre': 'indie',
            'popularity': np.random.randint(20, 60),
            'danceability': np.random.uniform(0.3, 0.6),
            'energy': np.random.uniform(0.2, 0.5),
            'valence': np.random.uniform(0.2, 0.6),
            'acousticness': np.random.uniform(0.6, 0.95),
            'instrumentalness': np.random.uniform(0.0, 0.2),
            'tempo': np.random.uniform(80, 120)
        })
    
    # Electronic/EDM songs (high energy, high danceability, low acousticness)
    for i in range(50):
        data.append({
            'track_name': f'EDM Song {i+1}',
            'artists': f'DJ Artist {i%10}',
            'track_genre': 'electronic',
            'popularity': np.random.randint(50, 90),
            'danceability': np.random.uniform(0.7, 0.95),
            'energy': np.random.uniform(0.8, 1.0),
            'valence': np.random.uniform(0.4, 0.8),
            'acousticness': np.random.uniform(0.0, 0.1),
            'instrumentalness': np.random.uniform(0.3, 0.9),
            'tempo': np.random.uniform(120, 150)
        })
    
    return pd.DataFrame(data)


# ============================================
# ML CONCEPT 1: FEATURE ENGINEERING
# ============================================

def extract_features(df):
    """
    FEATURE ENGINEERING
    
    Converting raw data into useful numbers for ML.
    
    We select features that describe the "vibe" of a song:
    - danceability: Can you dance to it?
    - energy: How intense is it?
    - valence: Happy or sad?
    - acousticness: Acoustic or electronic?
    - instrumentalness: Has vocals?
    - tempo: How fast?
    """
    features = ['danceability', 'energy', 'valence', 'acousticness', 
                'instrumentalness', 'tempo']
    
    # Check which features exist in the dataset
    available = [f for f in features if f in df.columns]
    
    print(f"\n📊 Using features: {available}")
    
    return df[available].values, available


# ============================================
# ML CONCEPT 2: NORMALIZATION
# ============================================

def normalize_features(X):
    """
    NORMALIZATION (STANDARDIZATION)
    
    Problem: Features have different scales
    - tempo: 60-200 BPM
    - danceability: 0-1
    
    If we don't normalize, tempo will dominate because 
    its numbers are bigger.
    
    Solution: StandardScaler transforms each feature to have:
    - Mean = 0
    - Standard deviation = 1
    
    Formula: z = (x - mean) / std
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    print("\n🔧 Normalization:")
    print(f"   Before: tempo might be 120, danceability might be 0.8")
    print(f"   After: both are on same scale (roughly -2 to +2)")
    
    return X_normalized, scaler


# ============================================
# ML CONCEPT 3: USER PROFILE CREATION
# ============================================

def create_user_profile(df, liked_songs_indices):
    """
    CREATE USER PROFILE
    
    A user's taste = average of songs they like.
    
    If you like 10 songs, we average their features
    to create one vector representing your taste.
    """
    features = ['danceability', 'energy', 'valence', 'acousticness', 
                'instrumentalness', 'tempo']
    available = [f for f in features if f in df.columns]
    
    # Get features of liked songs
    liked_songs = df.iloc[liked_songs_indices][available]
    
    # Average them
    profile = liked_songs.mean().values
    
    return profile, available


# ============================================
# ML CONCEPT 4: COSINE SIMILARITY
# ============================================

def calculate_similarity(profile1, profile2):
    """
    COSINE SIMILARITY
    
    Measures angle between two vectors.
    
    - 1.0 = identical direction = same taste
    - 0.0 = perpendicular = unrelated
    - -1.0 = opposite direction = opposite taste
    
    We reshape to 2D because sklearn expects it.
    """
    p1 = profile1.reshape(1, -1)
    p2 = profile2.reshape(1, -1)
    
    similarity = cosine_similarity(p1, p2)[0][0]
    
    return similarity


# ============================================
# ML CONCEPT 5: CLUSTERING (K-MEANS)
# ============================================

def cluster_songs(X, n_clusters=5):
    """
    K-MEANS CLUSTERING
    
    Groups similar songs together automatically.
    
    How it works:
    1. Randomly place K "centroids" (cluster centers)
    2. Assign each song to nearest centroid
    3. Move centroids to center of their assigned songs
    4. Repeat until centroids stop moving
    
    Result: Songs with similar vibes are in same cluster.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    return clusters, kmeans


# ============================================
# ML CONCEPT 6: RECOMMENDATIONS
# ============================================

def recommend_songs(df, user_profile, scaler, feature_names, n_recommendations=5):
    """
    CONTENT-BASED RECOMMENDATIONS
    
    Find songs most similar to user's taste profile.
    
    Steps:
    1. Normalize user profile (same scale as songs)
    2. Calculate similarity between user and every song
    3. Return top N most similar songs
    """
    # Get all song features
    X = df[feature_names].values
    X_normalized = scaler.transform(X)
    
    # Normalize user profile
    user_normalized = scaler.transform(user_profile.reshape(1, -1))
    
    # Calculate similarity to each song
    similarities = cosine_similarity(user_normalized, X_normalized)[0]
    
    # Get top N indices
    top_indices = similarities.argsort()[::-1][:n_recommendations]
    
    recommendations = df.iloc[top_indices][['track_name', 'artists', 'track_genre']].copy()
    recommendations['similarity'] = similarities[top_indices]
    
    return recommendations


# ============================================
# MAIN: RUN EVERYTHING
# ============================================

def main():
    print("=" * 60)
    print("🎵 MUSIC ML LEARNING - Using Kaggle Dataset")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data()
    print(f"\n📁 Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Extract features
    X, feature_names = extract_features(df)
    print(f"\n📊 Feature matrix shape: {X.shape}")
    
    # Step 3: Normalize
    X_normalized, scaler = normalize_features(X)
    
    # Step 4: Create two simulated users
    print("\n" + "=" * 60)
    print("👤 SIMULATING TWO USERS")
    print("=" * 60)
    
    # User A likes pop and EDM (indices 0-9 are pop, 150-159 are EDM in sample)
    user_a_likes = list(range(0, 10)) + list(range(150, 160)) if len(df) >= 160 else list(range(0, min(20, len(df))))
    profile_a, _ = create_user_profile(df, user_a_likes)
    
    # User B likes indie and acoustic (indices 100-119 in sample)
    user_b_likes = list(range(100, 120)) if len(df) >= 120 else list(range(max(0, len(df)-20), len(df)))
    profile_b, _ = create_user_profile(df, user_b_likes)
    
    print(f"\n👤 User A Profile (Pop/EDM lover):")
    for i, name in enumerate(feature_names):
        print(f"   {name}: {profile_a[i]:.3f}")
    
    print(f"\n👤 User B Profile (Indie/Acoustic lover):")
    for i, name in enumerate(feature_names):
        print(f"   {name}: {profile_b[i]:.3f}")
    
    # Step 5: Calculate compatibility
    print("\n" + "=" * 60)
    print("💕 COMPATIBILITY CALCULATION")
    print("=" * 60)
    
    # Normalize profiles for fair comparison
    profile_a_norm = scaler.transform(profile_a.reshape(1, -1))[0]
    profile_b_norm = scaler.transform(profile_b.reshape(1, -1))[0]
    
    similarity = calculate_similarity(profile_a_norm, profile_b_norm)
    compatibility = (similarity + 1) / 2 * 100  # Convert to 0-100
    
    print(f"\n🎯 Cosine Similarity: {similarity:.3f}")
    print(f"🎯 Compatibility Score: {compatibility:.1f}%")
    
    if compatibility > 70:
        print("   Status: Great match! 🔥")
    elif compatibility > 50:
        print("   Status: Some overlap 🎵")
    else:
        print("   Status: Different tastes 🌈")
    
    # Step 6: Clustering
    print("\n" + "=" * 60)
    print("📊 CLUSTERING SONGS")
    print("=" * 60)
    
    clusters, kmeans = cluster_songs(X_normalized, n_clusters=4)
    df['cluster'] = clusters
    
    print("\n🎨 Songs grouped into 4 clusters:")
    for i in range(4):
        cluster_songs_df = df[df['cluster'] == i]
        if 'track_genre' in df.columns:
            genres = cluster_songs_df['track_genre'].value_counts().head(2)
            print(f"\n   Cluster {i}: {len(cluster_songs_df)} songs")
            print(f"   Main genres: {dict(genres)}")
    
    # Step 7: Recommendations
    print("\n" + "=" * 60)
    print("🎧 RECOMMENDATIONS FOR USER A")
    print("=" * 60)
    
    recommendations = recommend_songs(df, profile_a, scaler, feature_names, n_recommendations=5)
    print("\nTop 5 recommended songs:")
    for i, row in recommendations.iterrows():
        print(f"   {row['track_name']} by {row['artists']}")
        print(f"      Genre: {row['track_genre']}, Similarity: {row['similarity']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📚 WHAT YOU LEARNED")
    print("=" * 60)
    print("""
    1. FEATURE ENGINEERING - Converting songs to numbers
    2. NORMALIZATION - Putting all features on same scale
    3. USER PROFILES - Averaging liked songs' features
    4. COSINE SIMILARITY - Measuring how similar two vectors are
    5. K-MEANS CLUSTERING - Grouping similar items automatically
    6. RECOMMENDATIONS - Finding items similar to user's taste
    
    These are REAL ML techniques used by Spotify, Netflix, YouTube!
    """)


if __name__ == "__main__":
    main()