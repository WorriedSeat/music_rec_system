import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

class MusicRecEnv(gym.Env):

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api__key: str,
        sessions_path: str,
        collection_name: str,
        batch_size: int,
        embedding_dim: int
    ):
        super().__init__()
        
        # === ДАННЫЕ ===
        self.sessions_df = pd.read_parquet(sessions_path)
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api__key
        )
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        
        # === СПЕЙСЫ ===
        # Action: коррекция в 128D пространстве
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(embedding_dim,), dtype=np.float32
        )
        
        # Observation: batch_size × (128 emb + time + action_score)
        state_size = batch_size * (embedding_dim + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # === Состояние ===
        self.current_session = None #dataframe всей сессии
        self.session_pos = 0 #позици

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_session = self.sessions_df.sample(1, random_state=seed).iloc[0]
        self.session_pos = 0
        
        batch_ids = self.current_session['item_ids'][:self.batch_size]
        times = self.current_session['played_ratio_pct'][:self.batch_size]
        feedbacks = self.current_session['actions'][:self.batch_size]
        
        state = self._build_state(batch_ids, times, feedbacks)
        self.current_state = state
        return state, {}

    def step(self, action: np.ndarray):
        start_idx = self.session_pos + self.batch_size
        if start_idx >= self.current_session['session_length']:
            return self.current_state, 0.0, True, False, {}

        # === Извлекаем pos/neg из текущего батча ===
        pos_embs, neg_embs = [], []
        pos_w, neg_w = [], []
        
        for i in range(self.batch_size):
            emb = self.current_state[i*(self.embedding_dim+2):i*(self.embedding_dim+2)+self.embedding_dim]
            time = self.current_state[i*(self.embedding_dim+2)+self.embedding_dim]
            fb = self.current_state[i*(self.embedding_dim+2)+self.embedding_dim+1]
            
            if fb > 0 or time > 0.7:
                pos_embs.append(emb)
                pos_w.append(time + fb)
            if fb < 0 or time < 0.4:
                neg_embs.append(emb)
                neg_w.append((1 - time) + abs(fb))

        target_pos = np.average(pos_embs, axis=0, weights=pos_w) if pos_embs else np.zeros(self.embedding_dim)
        target_neg = np.average(neg_embs, axis=0, weights=neg_w) if neg_embs else np.zeros(self.embedding_dim)
        
        # === Коррекция от action ===
        final_target = target_pos - 0.5 * target_neg + 0.15 * action
        final_target = final_target / (np.linalg.norm(final_target) + 1e-8)

        # === Поиск в Qdrant ===
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=final_target.tolist(),
            limit=self.batch_size * 10,
            with_payload=True
        )
        
        seen = set(self.current_session['item_ids'][:start_idx + self.batch_size])
        new_batch = []
        for r in results:
            item_id = r.payload.get('item_id')
            if item_id and item_id not in seen and item_id not in new_batch:
                new_batch.append(item_id)
            if len(new_batch) == self.batch_size:
                break
        
        # fallback
        if len(new_batch) < self.batch_size:
            candidates = self.sessions_df['item_ids'].explode().unique()
            for cand in np.random.choice(candidates, size=100, replace=False):
                if cand not in seen and cand not in new_batch:
                    new_batch.append(cand)
                if len(new_batch) == self.batch_size:
                    break

        # === Фидбек из реальной сессии ===
        real_times = np.array(self.current_session['played_ratio_pct'][start_idx:start_idx+self.batch_size])
        real_actions = np.array(self.current_session['actions'][start_idx:start_idx+self.batch_size])
        
        reward = float(np.mean(real_times) + np.mean(real_actions))

        # === Новый state ===
        self.current_state = self._build_state(new_batch, real_times, real_actions)
        self.session_pos = start_idx
        
        done = (self.session_pos + self.batch_size >= self.current_session['session_length'])
        return self.current_state, reward, done, False, {}

    def _get_emb(self, item_id) -> np.ndarray:
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[int(item_id)],
                with_payload=False,
                with_vectors=True
            )
            return np.array(point[0].vector, dtype=np.float32)
        except Exception as e:
            print(f"EXCEPTION WHILE GETTING EMBEDDING: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        

    def _build_state(self, batch_ids, times, feedbacks):
        state = []
        for i in range(self.batch_size):
            emb = self._get_emb(batch_ids[i])
            state.extend(emb)
            state.append(float(times[i]))
            state.append(float(feedbacks[i]))
        return np.array(state, dtype=np.float32)