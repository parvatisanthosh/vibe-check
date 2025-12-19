# Vibe-Check 🎵

A music compatibility web app that analyzes Spotify listening habits and calculates how similar two users' music tastes are.

## What it does

- Fetches users' top tracks from Spotify
- Uses Kaggle dataset for audio features (since Spotify deprecated audio features API)
- Compares listening patterns between two users
- Generates a compatibility score with insights

## Tech Stack

- **Frontend:** React.js
- **Backend:** Node.js, Express
- **API:** Spotify Web API
- **Data:** Kaggle Spotify dataset for audio features
- **Authentication:** Spotify OAuth 2.0

## Features

- Spotify login integration
- Audio feature analysis (energy, valence, danceability, acousticness)
- User-to-user music taste comparison
- Compatibility percentage and breakdown

## Setup

1. Clone the repo
```bash
git clone https://github.com/parvatisanthosh/vibe-check
cd vibe-check
```

2. Install dependencies
```bash
npm install
```

3. Run the app
```bash
npm start
```

## How it works

1. User logs in with Spotify
2. App fetches their top 50 tracks
3. Matches tracks with Kaggle dataset to get audio features
4. When second user connects, compares both profiles
5. Calculates similarity using cosine similarity on audio features
6. Displays compatibility score and insights



## Future improvements

- Compare with friends directly
- Playlist generation based on shared tastes
- Historical compatibility tracking

