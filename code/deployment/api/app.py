from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from qdrant_client import QdrantClient
import pandas as pd
import random
import os
from dotenv import load_dotenv
# from config import COLLECTION_NAME

load_dotenv()

EMBED_DIM = int(os.getenv('EMBED_DIM', 128))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 5))
QDRANT_URL = os.getenv('QDRANT_URL')
COLLECTION_NAME = 'yambda_50m'
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

app = FastAPI(
    title="Music Recommendation API",
    description="PPO-based music recommendation with metadata",
    version="1.0.0"
)

# Global variables
model = None
qdrant = None
songs_metadata_df = None


# ============ Models ============
class SongFeedback(BaseModel):
    track_id: int
    played_ratio: float
    action: int

class SessionStartRequest(BaseModel):
    user_id: str
    genre_preference: str = None

class HistoryRequest(BaseModel):
    session_id: str
    songs: list[SongFeedback]

class SongResponse(BaseModel):
    track_id: int
    # title: str
    # artist: str
    track_length_seconds: int
    color_hex: str


# ============ Startup ============
@app.on_event("startup")
async def startup_event():
    global model, qdrant, songs_metadata_df

    print("Loading PPO model...")
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "../../../models/ppo_musicrec_final")
    model = PPO.load(model_path)
    print("✓ Model loaded")

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)
    print("✓ Qdrant connected")

    print("Loading songs metadata CSV...")
    # metadata_path = os.path.join(current_dir, "../../../data/songs_metadata.csv")
    metadata_path = os.path.join(current_dir, '../../../data/clustered_tracks_new.csv')
    songs_metadata_df = pd.read_csv(metadata_path)
    print(f"✓ Loaded {len(songs_metadata_df)} songs metadata")


# ============ Helper Functions ============
def get_song_metadata(track_ids: list[int]) -> list[dict]:
    """Fetch metadata for given song IDs from CSV"""
    results = []

    for track_id in track_ids:
        # Look up in dataframe
        song_data = songs_metadata_df[songs_metadata_df['track_id'] == track_id]

        if not song_data.empty:
            row = song_data.iloc[0]
            results.append({
                "track_id": int(row['track_id']),
                # "title": str(row['title']),
                # "artist": str(row['artist']),
                "track_length_seconds": int(row['track_length_seconds']),
                "color_hex": str(row['color_hex'])
            })
        else:
            # Fallback for missing songs
            results.append({
                "track_id": track_id,
                # "title": f"Unknown Song {track_id}",
                # "artist": "Unknown Artist",
                "track_length_seconds": 180,
                "color_hex": "#808080"
            })

    return results


def get_random_popular_songs(n: int) -> list[int]:
    """Get N random songs from Qdrant"""
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count

        max_offset = max(0, total_points - n * 10)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=n,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )

        points = scroll_result[0]
        song_ids = [int(p.id) for p in points]

        print(f"✓ Got {len(song_ids)} starter songs: {song_ids}")
        return song_ids

    except Exception as e:
        print(f"✗ Error getting random songs: {e}")
        # Fallback: return some known IDs from CSV
        return songs_metadata_df['track_id'].sample(min(n, len(songs_metadata_df))).tolist()


def get_emb(track_id: int) -> np.ndarray:
    """Retrieve embedding from Qdrant"""
    try:
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[int(track_id)],
            with_vectors=True
        )
        if points and points[0].vector:
            return np.array(points[0].vector, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding for {track_id}: {e}")

    return np.zeros(EMBED_DIM, dtype=np.float32)


# ============ ENDPOINT 1: Start Session ============
@app.post("/start_session")
def start_session(req: SessionStartRequest):
    """
    Initialize a new session with 5 starter songs WITH metadata.
    """
    # Get song IDs
    starter_song_ids = get_random_popular_songs(BATCH_SIZE)

    # Fetch metadata
    songs_with_metadata = get_song_metadata(starter_song_ids)

    return {
        "session_id": req.user_id,
        "songs": songs_with_metadata,
        "message": "Session started. Listen and provide feedback!"
    }


# ============ ENDPOINT 2: Get Recommendations ============
@app.post("/recommend")
def recommend(req: HistoryRequest):
    """
    Get next recommendations WITH metadata based on user feedback.
    """
    if not req.songs or len(req.songs) == 0:
        return {"error": "No songs in history. Use /start_session first."}

    # Step 1: Build state
    state = []
    for i in range(BATCH_SIZE):
        if i < len(req.songs):
            item = req.songs[i]
            emb = get_emb(item.track_id)
            state.extend(emb)
            state.append(item.played_ratio)
            state.append(item.action)
        else:
            state.extend([0.0] * EMBED_DIM)
            state.append(0.0)
            state.append(0.0)

    state_np = np.array(state, dtype=np.float32)

    # Step 2: PPO predicts delta
    action, _ = model.predict(state_np, deterministic=True)

    # Step 3: Compute pos/neg centroids
    pos_embs, neg_embs, pos_w, neg_w = [], [], [], []

    for i in range(min(BATCH_SIZE, len(req.songs))):
        item = req.songs[i]
        emb = get_emb(item.track_id)

        if item.action > 0 or item.played_ratio > 0.7:
            pos_embs.append(emb)
            pos_w.append(item.played_ratio + item.action)

        if item.action < 0 or item.played_ratio < 0.4:
            neg_embs.append(emb)
            neg_w.append((1 - item.played_ratio) + abs(item.action))

    target_pos = np.average(pos_embs, axis=0, weights=pos_w) if pos_embs else np.zeros(EMBED_DIM)
    target_neg = np.average(neg_embs, axis=0, weights=neg_w) if neg_embs else np.zeros(EMBED_DIM)

    # Step 4: Combine into final target
    final_target = target_pos - 0.5 * target_neg + 0.15 * action
    norm = np.linalg.norm(final_target)
    if norm > 0:
        final_target = final_target / norm

    # Step 5: Search Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=final_target.tolist(),
        limit=BATCH_SIZE * 3,
        with_payload=False
    ).points

    # Step 6: Filter seen songs
    seen_ids = {song.track_id for song in req.songs}
    recommendation_ids = []

    for res in results:
        track_id = int(res.id)

        if track_id not in seen_ids:
            recommendation_ids.append(track_id)

        if len(recommendation_ids) == BATCH_SIZE:
            break

    # Fallback if not enough
    if len(recommendation_ids) < BATCH_SIZE:
        extra_needed = BATCH_SIZE - len(recommendation_ids)
        extra_songs = get_random_popular_songs(extra_needed * 2)
        for song_id in extra_songs:
            if song_id not in seen_ids and song_id not in recommendation_ids:
                recommendation_ids.append(song_id)
            if len(recommendation_ids) == BATCH_SIZE:
                break

    # Step 7: Fetch metadata for recommended songs
    songs_with_metadata = get_song_metadata(recommendation_ids)

    return {
        "songs": songs_with_metadata
    }


# ============ Health Check ============
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "qdrant_connected": qdrant is not None,
        "metadata_loaded": songs_metadata_df is not None,
        "total_songs_metadata": len(songs_metadata_df) if songs_metadata_df is not None else 0
    }


# ============ Debug ============
@app.get("/debug/metadata/{track_id}")
def debug_metadata(track_id: int):
    """Check metadata for a specific song"""
    metadata = get_song_metadata([track_id])
    return metadata[0] if metadata else {"error": "Not found"}
