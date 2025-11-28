from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from qdrant_client import QdrantClient
import random

EMBED_DIM = 128
BATCH_SIZE = 5
COLLECTION_NAME = "yambda_50m"
QDRANT_URL = "https://60266f02-4dd1-4ed8-82c0-801dc928f25d.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Mbw44XBHjl9wK7jLdOh9eh2U48T8JoilRGcv_f1T1Ls"


app = FastAPI()

model = PPO.load("../../../models/ppo_musicrec_final")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ============ Models ============
class SongFeedback(BaseModel):
    item_id: int
    played_ratio: float  # 0.0 - 1.0
    action: int          # 1=like, 0=neutral, -1=skip

class SessionStartRequest(BaseModel):
    user_id: str
    genre_preference: str = 'rock'  # Optional: "rock", "pop", etc.

class HistoryRequest(BaseModel):
    session_id: str
    songs: list[SongFeedback]


# ============ ENDPOINT 1: Start Session ============
@app.post("/start_session")
def start_session(req: SessionStartRequest):
    """
    Initialize a new session with 5 starter songs.
    This runs ONCE at the beginning.
    """

    # Option 1: Random popular songs
    starter_songs = get_random_popular_songs(BATCH_SIZE)

    # Option 2: If user has genre preference
    # if req.genre_preference:
    #     starter_songs = get_songs_by_genre(req.genre_preference, BATCH_SIZE)

    return {
        "session_id": req.user_id,
        "initial_songs": starter_songs,
        "message": "Session started. Listen and provide feedback!"
    }


def get_random_popular_songs(n: int) -> list[int]:
    """
    Get N random songs using point IDs directly.
    """
    try:
        import random

        # Get collection info
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count

        # Random offset to get different songs each time
        max_offset = max(0, total_points - n * 10)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        # Scroll to get points
        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=n,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )

        points = scroll_result[0]

        # Return point IDs as item IDs
        song_ids = [int(p.id) for p in points]

        print(f"✓ Got {len(song_ids)} starter songs: {song_ids}")
        return song_ids

    except Exception as e:
        print(f"✗ Error getting random songs: {e}")
        # Fallback: return some low IDs
        return [26, 43, 50, 71, 81]  # IDs from your sample


# ============ ENDPOINT 2: Get Recommendations ============
@app.post("/recommend")
def recommend(req: HistoryRequest):
    """
    Get next recommendations based on user feedback.
    This runs in a LOOP after every interaction.
    """

    # Ensure we have songs to process
    if not req.songs or len(req.songs) == 0:
        return {"error": "No songs in history. Use /start_session first."}

    # Step 1: Build state
    state = []
    for i in range(BATCH_SIZE):
        if i < len(req.songs):
            item = req.songs[i]
            emb = get_emb(item.item_id)
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
        emb = get_emb(item.item_id)

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
        with_payload=True
    ).points

    # Step 6: Filter seen songs
    seen_ids = {song.item_id for song in req.songs}
    recommendations = []

    for res in results:
        # item_id = res.payload.get('item_id')
        item_id = int(res.id)
        if item_id and item_id not in seen_ids:
            recommendations.append(item_id)
        if len(recommendations) == BATCH_SIZE:
            break

    # Fallback: if not enough recommendations, add random songs
    if len(recommendations) < BATCH_SIZE:
        extra_needed = BATCH_SIZE - len(recommendations)
        extra_songs = get_random_popular_songs(extra_needed * 2)
        for song_id in extra_songs:
            if song_id not in seen_ids and song_id not in recommendations:
                recommendations.append(song_id)
            if len(recommendations) == BATCH_SIZE:
                break

    return {
        "recommendations": recommendations,
        "count": len(recommendations)
    }


def get_emb(item_id: int):
    """Retrieve embedding from Qdrant."""
    try:
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[int(item_id)],
            with_vectors=True
        )
        if points and points[0].vector:
            return np.array(points[0].vector, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding for {item_id}: {e}")

    return np.zeros(EMBED_DIM, dtype=np.float32)


# ============ Health Check ============
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "qdrant_connected": qdrant is not None
    }

@app.get("/debug/qdrant")
def debug_qdrant():
    """Debug endpoint to check Qdrant connection"""
    try:
        # Get collection info
        collection_info = qdrant.get_collection(COLLECTION_NAME)

        # Try to get a few random points
        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]

        return {
            "status": "connected",
            "collection_name": COLLECTION_NAME,
            "total_points": collection_info.points_count,
            "sample_points": [
                {
                    "id": p.id,
                    "payload": p.payload
                } for p in points
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

