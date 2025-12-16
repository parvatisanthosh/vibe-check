import numpy as np

# ============================================
# WHAT IS THIS FILE?
# ============================================
# We're building a music compatibility checker.
# It compares two users' music taste and gives a score.
#
# HOW?
# 1. Each song has "audio features" (energy, danceability, etc.)
# 2. We average all songs to get a "user profile"
# 3. We compare two profiles using "cosine similarity"
# ============================================


# ============================================
# FAKE SPOTIFY DATA
# ============================================
# Spotify gives these features for every song (0 to 1):
#
# energy: How intense/loud (0=calm, 1=energetic)
# danceability: Can you dance to it (0=no, 1=yes)
# valence: Happy or sad (0=sad, 1=happy)
# acousticness: Acoustic vs electronic (0=electronic, 1=acoustic)
# instrumentalness: Has vocals? (0=vocals, 1=no vocals)

# User A: Party person - likes energetic, happy, dance music
user_a_songs = [
    {"energy": 0.8, "danceability": 0.9, "valence": 0.7, "acousticness": 0.1, "instrumentalness": 0.0},
    {"energy": 0.9, "danceability": 0.8, "valence": 0.8, "acousticness": 0.2, "instrumentalness": 0.1},
    {"energy": 0.7, "danceability": 0.85, "valence": 0.6, "acousticness": 0.15, "instrumentalness": 0.0},
    {"energy": 0.85, "danceability": 0.75, "valence": 0.9, "acousticness": 0.1, "instrumentalness": 0.0},
    {"energy": 0.75, "danceability": 0.95, "valence": 0.7, "acousticness": 0.05, "instrumentalness": 0.1},
]

# User B: Also party person (should be HIGH compatibility with A)
user_b_songs = [
    {"energy": 0.85, "danceability": 0.8, "valence": 0.75, "acousticness": 0.15, "instrumentalness": 0.0},
    {"energy": 0.9, "danceability": 0.85, "valence": 0.7, "acousticness": 0.1, "instrumentalness": 0.0},
    {"energy": 0.75, "danceability": 0.9, "valence": 0.65, "acousticness": 0.2, "instrumentalness": 0.1},
    {"energy": 0.8, "danceability": 0.7, "valence": 0.8, "acousticness": 0.1, "instrumentalness": 0.0},
    {"energy": 0.7, "danceability": 0.88, "valence": 0.75, "acousticness": 0.1, "instrumentalness": 0.05},
]

# User C: Sad acoustic lover (should be LOW compatibility with A)
user_c_songs = [
    {"energy": 0.2, "danceability": 0.3, "valence": 0.2, "acousticness": 0.9, "instrumentalness": 0.3},
    {"energy": 0.15, "danceability": 0.25, "valence": 0.1, "acousticness": 0.95, "instrumentalness": 0.4},
    {"energy": 0.3, "danceability": 0.4, "valence": 0.25, "acousticness": 0.85, "instrumentalness": 0.2},
    {"energy": 0.25, "danceability": 0.35, "valence": 0.15, "acousticness": 0.9, "instrumentalness": 0.35},
    {"energy": 0.1, "danceability": 0.2, "valence": 0.3, "acousticness": 0.92, "instrumentalness": 0.5},
]


# ============================================
# ML CONCEPT 1: FEATURE VECTOR
# ============================================
# A "feature vector" is just a list of numbers that describes something.
#
# Example: A song's feature vector
# [0.8, 0.9, 0.7, 0.1, 0.0]
#  ^    ^    ^    ^    ^
#  |    |    |    |    instrumentalness
#  |    |    |    acousticness
#  |    |    valence
#  |    danceability
#  energy
#
# We can compare vectors to see how similar they are!


# ============================================
# STEP 1: CREATE USER PROFILE
# ============================================
# Average all songs to get one "profile" for the user

def create_user_profile(songs):
    """
    Input: List of songs (each song is a dict of features)
    Output: One dict with average of each feature
    
    Example:
    songs = [{"energy": 0.8, ...}, {"energy": 0.6, ...}]
    output = {"energy": 0.7, ...}  (average)
    """
    features = ["energy", "danceability", "valence", "acousticness", "instrumentalness"]
    
    profile = {}
    for feature in features:
        # Sum all values for this feature
        total = 0
        for song in songs:
            total += song[feature]
        
        # Divide by number of songs to get average
        profile[feature] = total / len(songs)
    
    return profile


# ============================================
# STEP 2: CONVERT TO NUMPY ARRAY
# ============================================
# Math is easier with numpy arrays

def profile_to_vector(profile):
    """
    Input: {"energy": 0.8, "danceability": 0.9, ...}
    Output: [0.8, 0.9, ...] as numpy array
    """
    features = ["energy", "danceability", "valence", "acousticness", "instrumentalness"]
    return np.array([profile[f] for f in features])


# ============================================
# ML CONCEPT 2: COSINE SIMILARITY
# ============================================
# How do we compare two vectors?
#
# Imagine two arrows pointing from origin:
#
#     B
#    /
#   /  angle
#  /______ A
#
# If arrows point same direction → similar (angle = 0)
# If arrows point opposite → not similar (angle = 180)
#
# Cosine similarity = cos(angle between vectors)
# - Returns 1 if same direction (perfect match)
# - Returns 0 if perpendicular (no match)
# - Returns -1 if opposite (complete opposite)
#
# Formula: cos(θ) = (A · B) / (|A| × |B|)
# - A · B = dot product
# - |A| = magnitude (length) of A

def cosine_similarity(vec_a, vec_b):
    """
    Input: Two numpy arrays
    Output: Number between -1 and 1
    """
    # Dot product: multiply corresponding elements and sum
    # [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32
    dot_product = np.dot(vec_a, vec_b)
    
    # Magnitude: length of vector
    # |[3,4]| = sqrt(3² + 4²) = 5
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    # Cosine similarity formula
    return dot_product / (magnitude_a * magnitude_b)


# ============================================
# STEP 3: CALCULATE COMPATIBILITY
# ============================================

def calculate_compatibility(user1_songs, user2_songs):
    """
    Main function!
    Input: Two users' song lists
    Output: Compatibility percentage (0-100)
    """
    # Create profiles (average of songs)
    profile1 = create_user_profile(user1_songs)
    profile2 = create_user_profile(user2_songs)
    
    # Convert to vectors
    vec1 = profile_to_vector(profile1)
    vec2 = profile_to_vector(profile2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)
    
    # Convert to 0-100 scale
    # similarity is -1 to 1, we want 0 to 100
    compatibility = (similarity + 1) / 2 * 100
    
    return compatibility, profile1, profile2


# ============================================
# RUN THE CODE!
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🎵 VIBE MATCH - Music Compatibility Checker 🎵")
    print("=" * 50)
    
    # Test 1: Similar users
    compat_ab, prof_a, prof_b = calculate_compatibility(user_a_songs, user_b_songs)
    print(f"\n👯 User A vs User B (both party people)")
    print(f"   Compatibility: {compat_ab:.1f}%")
    
    # Test 2: Different users
    compat_ac, prof_a, prof_c = calculate_compatibility(user_a_songs, user_c_songs)
    print(f"\n😢 User A vs User C (party vs sad)")
    print(f"   Compatibility: {compat_ac:.1f}%")
    
    # Show what the profiles look like
    print(f"\n📊 User A's Music Profile:")
    for key, val in prof_a.items():
        bar = "█" * int(val * 20)
        print(f"   {key:18} {bar} {val:.2f}")
    
    print(f"\n📊 User C's Music Profile:")
    for key, val in prof_c.items():
        bar = "█" * int(val * 20)
        print(f"   {key:18} {bar} {val:.2f}")