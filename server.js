const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = 3000;

const CLIENT_ID = process.env.SPOTIFY_CLIENT_ID;
const CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET;
const REDIRECT_URI = process.env.REDIRECT_URI || 'http://127.0.0.1:3000/callback';

// Store users in memory
let usersData = {};

// ============================================
// ML FUNCTIONS
// ============================================

/**
 * JACCARD SIMILARITY
 * 
 * Measures overlap between two sets.
 * 
 * Formula: |A ∩ B| / |A ∪ B|
 * = (items in BOTH) / (all unique items)
 * 
 * Example:
 * User A: {pop, rock, indie}
 * User B: {pop, rock, jazz}
 * Intersection: {pop, rock} = 2
 * Union: {pop, rock, indie, jazz} = 4
 * Jaccard = 2/4 = 0.5 = 50%
 */
function jaccardSimilarity(setA, setB) {
    const a = new Set(setA);
    const b = new Set(setB);
    
    const intersection = new Set([...a].filter(x => b.has(x)));
    const union = new Set([...a, ...b]);
    
    if (union.size === 0) return 0;
    return intersection.size / union.size;
}

/**
 * WEIGHTED JACCARD SIMILARITY
 * 
 * Like Jaccard, but items have weights (importance).
 * Items that appear more often in your listening = higher weight.
 * 
 * This is more accurate than regular Jaccard because
 * your #1 genre matters more than your #10 genre.
 */
function weightedJaccardSimilarity(list1, list2) {
    // Create frequency maps (how often each item appears)
    const freq1 = {};
    const freq2 = {};
    
    list1.forEach((item, index) => {
        // Higher weight for items earlier in list (more listened)
        freq1[item] = (freq1[item] || 0) + (1 / (index + 1));
    });
    
    list2.forEach((item, index) => {
        freq2[item] = (freq2[item] || 0) + (1 / (index + 1));
    });
    
    // Get all unique items
    const allItems = new Set([...Object.keys(freq1), ...Object.keys(freq2)]);
    
    let minSum = 0;  // Intersection (minimum of weights)
    let maxSum = 0;  // Union (maximum of weights)
    
    for (const item of allItems) {
        const w1 = freq1[item] || 0;
        const w2 = freq2[item] || 0;
        minSum += Math.min(w1, w2);
        maxSum += Math.max(w1, w2);
    }
    
    if (maxSum === 0) return 0;
    return minSum / maxSum;
}

/**
 * COSINE SIMILARITY ON GENRE VECTORS
 * 
 * Treats genres as dimensions in a vector space.
 * Each user is a vector where each dimension = genre frequency.
 */
function genreVectorSimilarity(genres1, genres2) {
    // Count genre frequencies
    const freq1 = {};
    const freq2 = {};
    
    genres1.forEach(g => freq1[g] = (freq1[g] || 0) + 1);
    genres2.forEach(g => freq2[g] = (freq2[g] || 0) + 1);
    
    // Get all unique genres
    const allGenres = new Set([...Object.keys(freq1), ...Object.keys(freq2)]);
    
    // Create vectors
    const vec1 = [];
    const vec2 = [];
    
    for (const genre of allGenres) {
        vec1.push(freq1[genre] || 0);
        vec2.push(freq2[genre] || 0);
    }
    
    // Cosine similarity
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        mag1 += vec1[i] * vec1[i];
        mag2 += vec2[i] * vec2[i];
    }
    
    mag1 = Math.sqrt(mag1);
    mag2 = Math.sqrt(mag2);
    
    if (mag1 === 0 || mag2 === 0) return 0;
    return dotProduct / (mag1 * mag2);
}

/**
 * POPULARITY SIMILARITY
 * 
 * Compares how mainstream vs niche users' tastes are.
 * Someone who listens to obscure indie has different taste
 * than someone who only plays top 40 hits.
 */
function popularitySimilarity(pop1, pop2) {
    // Both values are 0-100
    // Closer values = more similar
    const diff = Math.abs(pop1 - pop2);
    return 1 - (diff / 100);
}

/**
 * MAIN COMPATIBILITY FUNCTION
 * 
 * Combines multiple similarity metrics with weights.
 */
function calculateCompatibility(profile1, profile2) {
    // 1. Genre Similarity (using cosine on genre vectors) - 40%
    const genreSim = genreVectorSimilarity(profile1.allGenres, profile2.allGenres);
    
    // 2. Weighted Genre Overlap - 25%
    const weightedGenreSim = weightedJaccardSimilarity(profile1.topGenres, profile2.topGenres);
    
    // 3. Artist Overlap - 20%
    const artistIds1 = profile1.topArtists.map(a => a.id);
    const artistIds2 = profile2.topArtists.map(a => a.id);
    const artistSim = jaccardSimilarity(artistIds1, artistIds2);
    
    // 4. Track Overlap - 10%
    const trackIds1 = profile1.topTracks.map(t => t.id);
    const trackIds2 = profile2.topTracks.map(t => t.id);
    const trackSim = jaccardSimilarity(trackIds1, trackIds2);
    
    // 5. Popularity Similarity - 5%
    const popSim = popularitySimilarity(profile1.avgPopularity, profile2.avgPopularity);
    
    // Weighted combination
    const overallScore = (
        genreSim * 0.40 +
        weightedGenreSim * 0.25 +
        artistSim * 0.20 +
        trackSim * 0.10 +
        popSim * 0.05
    ) * 100;
    
    // Find shared items
    const sharedGenres = profile1.topGenres.filter(g => profile2.topGenres.includes(g));
    const sharedArtists = profile1.topArtists
        .filter(a => artistIds2.includes(a.id))
        .map(a => a.name);
    const sharedTracks = profile1.topTracks
        .filter(t => trackIds2.includes(t.id))
        .map(t => `${t.name} - ${t.artist}`);
    
    return {
        overall: Math.round(overallScore),
        breakdown: {
            genres: Math.round(genreSim * 100),
            weightedGenres: Math.round(weightedGenreSim * 100),
            artists: Math.round(artistSim * 100),
            tracks: Math.round(trackSim * 100),
            popularity: Math.round(popSim * 100)
        },
        sharedGenres,
        sharedArtists,
        sharedTracks
    };
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function getTopGenres(artists) {
    const genreCount = {};
    
    for (const artist of artists) {
        for (const genre of artist.genres || []) {
            genreCount[genre] = (genreCount[genre] || 0) + 1;
        }
    }
    
    return Object.entries(genreCount)
        .sort((a, b) => b[1] - a[1])
        .map(entry => entry[0]);
}

function getAllGenres(artists) {
    const genres = [];
    for (const artist of artists) {
        genres.push(...(artist.genres || []));
    }
    return genres;
}

function getAvgPopularity(tracks) {
    if (tracks.length === 0) return 50;
    const total = tracks.reduce((sum, t) => sum + t.popularity, 0);
    return total / tracks.length;
}

// ============================================
// ROUTES
// ============================================

app.get('/', (req, res) => {
    const userCount = Object.keys(usersData).length;
    
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vibe Match</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                }
                h1 { font-size: 3rem; margin-bottom: 10px; }
                .subtitle { color: #1DB954; font-size: 1.2rem; margin-bottom: 30px; }
                .description { color: #aaa; margin-bottom: 40px; max-width: 400px; }
                .login-btn {
                    background: #1DB954;
                    color: white;
                    padding: 15px 40px;
                    border-radius: 30px;
                    text-decoration: none;
                    font-weight: bold;
                    font-size: 1.1rem;
                    display: inline-block;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .login-btn:hover {
                    transform: scale(1.05);
                    box-shadow: 0 10px 30px rgba(29, 185, 84, 0.3);
                }
                .stats { margin-top: 40px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎵 Vibe Match</h1>
                <p class="subtitle">Music Compatibility Checker</p>
                <p class="description">
                    Compare your music taste with friends using ML-powered genre and artist analysis.
                </p>
                <a href="/login" class="login-btn">Login with Spotify</a>
                <p class="stats">${userCount} users connected</p>
            </div>
        </body>
        </html>
    `);
});

app.get('/login', (req, res) => {
    const compareWith = req.query.compare || '';
    const scopes = 'user-top-read';
    const state = compareWith;
    
    const url = `https://accounts.spotify.com/authorize?` +
        `client_id=${CLIENT_ID}` +
        `&response_type=code` +
        `&redirect_uri=${encodeURIComponent(REDIRECT_URI)}` +
        `&scope=${scopes}` +
        `&state=${state}`;
    
    res.redirect(url);
});

app.get('/callback', async (req, res) => {
    const code = req.query.code;
    const compareWith = req.query.state;
    const error = req.query.error;

    if (error) {
        return res.send(`<h1>Error: ${error}</h1><a href="/">Try again</a>`);
    }

    if (!code) {
        return res.send('<h1>No authorization code received</h1><a href="/">Try again</a>');
    }

    try {
        // 1. Get access token
        const tokenResponse = await axios.post(
            'https://accounts.spotify.com/api/token',
            new URLSearchParams({
                grant_type: 'authorization_code',
                code: code,
                redirect_uri: REDIRECT_URI,
            }),
            {
                headers: {
                    'Authorization': 'Basic ' + Buffer.from(CLIENT_ID + ':' + CLIENT_SECRET).toString('base64'),
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            }
        );

        const accessToken = tokenResponse.data.access_token;

        // 2. Get user info
        const userInfo = await axios.get('https://api.spotify.com/v1/me', {
            headers: { 'Authorization': `Bearer ${accessToken}` }
        });

        const userId = userInfo.data.id;
        const userName = userInfo.data.display_name || userId;

        // 3. Get top tracks
        const topTracks = await axios.get(
            'https://api.spotify.com/v1/me/top/tracks?limit=50&time_range=medium_term',
            { headers: { 'Authorization': `Bearer ${accessToken}` } }
        );

        // 4. Get top artists
        const topArtists = await axios.get(
            'https://api.spotify.com/v1/me/top/artists?limit=50&time_range=medium_term',
            { headers: { 'Authorization': `Bearer ${accessToken}` } }
        );

        // 5. Build user profile
        const topGenres = getTopGenres(topArtists.data.items);
        const allGenres = getAllGenres(topArtists.data.items);
        
        const profile = {
            name: userName,
            topTracks: topTracks.data.items.map(t => ({
                id: t.id,
                name: t.name,
                artist: t.artists[0].name,
                popularity: t.popularity
            })),
            topArtists: topArtists.data.items.map(a => ({
                id: a.id,
                name: a.name,
                genres: a.genres,
                popularity: a.popularity
            })),
            topGenres: topGenres,
            allGenres: allGenres,
            avgPopularity: getAvgPopularity(topTracks.data.items),
            mainstreamScore: Math.round(getAvgPopularity(topTracks.data.items))
        };

        // Store user
        usersData[userId] = profile;

        // If comparing with another user
        if (compareWith && usersData[compareWith]) {
            return res.redirect(`/result/${userId}/${compareWith}`);
        }

        // Show user's profile
        res.redirect(`/profile/${userId}`);

    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
        res.send(`
            <h1>Error</h1>
            <p>${error.response?.data?.error?.message || error.message}</p>
            <a href="/">Try again</a>
        `);
    }
});

app.get('/profile/:userId', (req, res) => {
    const user = usersData[req.params.userId];
    
    if (!user) {
        return res.send('<h1>User not found</h1><a href="/">Go home</a>');
    }
    
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>${user.name}'s Profile - Vibe Match</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    min-height: 100vh;
                    color: white;
                    padding: 40px 20px;
                }
                .container { max-width: 700px; margin: 0 auto; }
                h1 { font-size: 2rem; margin-bottom: 30px; }
                h2 { font-size: 1.3rem; margin: 30px 0 15px; color: #1DB954; }
                
                .stat-card {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .stat-label { color: #aaa; font-size: 0.9rem; }
                .stat-value { font-size: 2rem; font-weight: bold; color: #1DB954; }
                
                .genre-tags { display: flex; flex-wrap: wrap; gap: 8px; }
                .genre { 
                    background: rgba(29, 185, 84, 0.2);
                    border: 1px solid #1DB954;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                }
                
                .track-list { list-style: none; }
                .track-item {
                    padding: 12px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .track-name { font-weight: bold; }
                .track-artist { color: #aaa; font-size: 0.9rem; }
                
                .share-box {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 30px;
                }
                .share-link {
                    background: rgba(255,255,255,0.1);
                    padding: 10px;
                    border-radius: 5px;
                    word-break: break-all;
                    font-family: monospace;
                    margin: 10px 0;
                }
                .copy-btn {
                    background: #1DB954;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎵 ${user.name}'s Music Profile</h1>
                
                <div class="stat-card">
                    <div class="stat-label">Mainstream Score</div>
                    <div class="stat-value">${user.mainstreamScore}/100</div>
                    <div class="stat-label" style="margin-top: 5px;">
                        ${user.mainstreamScore > 70 ? "You love popular hits!" : 
                          user.mainstreamScore > 40 ? "Nice mix of popular and indie" : 
                          "You have unique taste!"}
                    </div>
                </div>
                
                <h2>🎸 Top Genres</h2>
                <div class="genre-tags">
                    ${user.topGenres.slice(0, 12).map(g => `<span class="genre">${g}</span>`).join('')}
                </div>
                
                <h2>🎧 Top Tracks</h2>
                <ul class="track-list">
                    ${user.topTracks.slice(0, 5).map(t => `
                        <li class="track-item">
                            <div class="track-name">${t.name}</div>
                            <div class="track-artist">${t.artist}</div>
                        </li>
                    `).join('')}
                </ul>
                
                <h2>👥 Top Artists</h2>
                <ul class="track-list">
                    ${user.topArtists.slice(0, 5).map(a => `
                        <li class="track-item">
                            <div class="track-name">${a.name}</div>
                            <div class="track-artist">${a.genres.slice(0, 3).join(', ')}</div>
                        </li>
                    `).join('')}
                </ul>
                
                <div class="share-box">
                    <h2 style="margin-top: 0;">📤 Compare with Friends</h2>
                    <p>Send this link to friends to see your compatibility:</p>
                    <div class="share-link" id="shareLink">
                        ${req.protocol}://${req.get('host')}/compare/${req.params.userId}
                    </div>
                    <button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('shareLink').innerText); this.innerText='Copied!'">
                        Copy Link
                    </button>
                </div>
            </div>
        </body>
        </html>
    `);
});

app.get('/compare/:userId', (req, res) => {
    const user = usersData[req.params.userId];
    
    if (!user) {
        return res.send('<h1>User not found</h1><p>They need to login first.</p><a href="/">Go home</a>');
    }

    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compare with ${user.name} - Vibe Match</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                }
                .container { text-align: center; padding: 40px; }
                h1 { font-size: 2.5rem; margin-bottom: 20px; }
                p { color: #aaa; margin-bottom: 30px; }
                .login-btn {
                    background: #1DB954;
                    color: white;
                    padding: 15px 40px;
                    border-radius: 30px;
                    text-decoration: none;
                    font-weight: bold;
                    font-size: 1.1rem;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎵 Vibe Match</h1>
                <p>Compare your music taste with <strong>${user.name}</strong></p>
                <a href="/login?compare=${req.params.userId}" class="login-btn">Login with Spotify</a>
            </div>
        </body>
        </html>
    `);
});

app.get('/result/:user1/:user2', (req, res) => {
    const user1 = usersData[req.params.user1];
    const user2 = usersData[req.params.user2];
    
    if (!user1 || !user2) {
        return res.send('<h1>Users not found</h1><a href="/">Go home</a>');
    }

    const compatibility = calculateCompatibility(user1, user2);
    
    let message, emoji;
    if (compatibility.overall >= 80) {
        message = "You're musical soulmates!";
        emoji = "🔥";
    } else if (compatibility.overall >= 60) {
        message = "Great compatibility!";
        emoji = "✨";
    } else if (compatibility.overall >= 40) {
        message = "Some overlap in your tastes";
        emoji = "🎵";
    } else {
        message = "Opposites attract?";
        emoji = "🌈";
    }

    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compatibility Result - Vibe Match</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    min-height: 100vh;
                    color: white;
                    padding: 40px 20px;
                }
                .container { max-width: 600px; margin: 0 auto; text-align: center; }
                
                .score-circle {
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    background: conic-gradient(#1DB954 ${compatibility.overall}%, rgba(255,255,255,0.1) 0);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 30px auto;
                }
                .score-inner {
                    width: 160px;
                    height: 160px;
                    border-radius: 50%;
                    background: #1a1a2e;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                .score-number { font-size: 3rem; font-weight: bold; }
                .score-label { color: #aaa; }
                
                h1 { margin-bottom: 10px; }
                .message { font-size: 1.5rem; color: #1DB954; margin-bottom: 30px; }
                
                .breakdown {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 30px 0;
                    text-align: left;
                }
                .breakdown h2 { margin-bottom: 15px; text-align: center; }
                .breakdown-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .breakdown-item:last-child { border-bottom: none; }
                
                .shared {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    text-align: left;
                }
                .shared h2 { margin-bottom: 15px; }
                .shared-items { display: flex; flex-wrap: wrap; gap: 8px; }
                .shared-item {
                    background: rgba(29, 185, 84, 0.2);
                    border: 1px solid #1DB954;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                }
                
                .back-btn {
                    background: #1DB954;
                    color: white;
                    padding: 15px 30px;
                    border-radius: 30px;
                    text-decoration: none;
                    display: inline-block;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>${emoji} ${user1.name} & ${user2.name}</h1>
                <p class="message">${message}</p>
                
                <div class="score-circle">
                    <div class="score-inner">
                        <div class="score-number">${compatibility.overall}%</div>
                        <div class="score-label">Compatible</div>
                    </div>
                </div>
                
                <div class="breakdown">
                    <h2>📊 How We Calculated This</h2>
                    <div class="breakdown-item">
                        <span>Genre Match (cosine similarity)</span>
                        <span>${compatibility.breakdown.genres}%</span>
                    </div>
                    <div class="breakdown-item">
                        <span>Genre Overlap (weighted)</span>
                        <span>${compatibility.breakdown.weightedGenres}%</span>
                    </div>
                    <div class="breakdown-item">
                        <span>Artist Overlap</span>
                        <span>${compatibility.breakdown.artists}%</span>
                    </div>
                    <div class="breakdown-item">
                        <span>Track Overlap</span>
                        <span>${compatibility.breakdown.tracks}%</span>
                    </div>
                    <div class="breakdown-item">
                        <span>Popularity Match</span>
                        <span>${compatibility.breakdown.popularity}%</span>
                    </div>
                </div>
                
                ${compatibility.sharedGenres.length > 0 ? `
                    <div class="shared">
                        <h2>🎸 Genres You Both Like</h2>
                        <div class="shared-items">
                            ${compatibility.sharedGenres.slice(0, 10).map(g => 
                                `<span class="shared-item">${g}</span>`
                            ).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${compatibility.sharedArtists.length > 0 ? `
                    <div class="shared">
                        <h2>🎤 Artists You Both Love</h2>
                        <div class="shared-items">
                            ${compatibility.sharedArtists.slice(0, 10).map(a => 
                                `<span class="shared-item">${a}</span>`
                            ).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${compatibility.sharedTracks.length > 0 ? `
                    <div class="shared">
                        <h2>🎧 Songs You Both Play</h2>
                        <div class="shared-items">
                            ${compatibility.sharedTracks.slice(0, 5).map(t => 
                                `<span class="shared-item">${t}</span>`
                            ).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <a href="/" class="back-btn">Try Again</a>
            </div>
        </body>
        </html>
    `);
});

// ============================================
// START SERVER
// ============================================
app.listen(PORT, () => {
    console.log(`🎵 Vibe Match running at http://127.0.0.1:${PORT}`);
});