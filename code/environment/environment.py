import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

class MusicRecEnv(gym.Env):
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api__key: str,  # Исправлено: убрал лишний _
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
            api_key=qdrant_api__key,  # Исправлено
            timeout=120.0
        )
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # === СПЕЙСЫ ===
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(embedding_dim,), dtype=np.float32
        )

        # Observation: batch_size × (embedding_dim + 2)
        state_size = batch_size * (embedding_dim + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

        # === Состояние ===
        self.current_session = None
        self.session_pos = 0
        self.current_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Выбираем случайную сессию
        self.current_session = self.sessions_df.sample(1).iloc[0]
        self.session_pos = 0

        # Получаем начальный батч
        batch_ids = self.current_session['item_ids'][:self.batch_size]
        times = self.current_session['played_ratio_pct'][:self.batch_size]
        feedbacks = self.current_session['actions'][:self.batch_size]

        # Обеспечиваем правильный размер
        if len(batch_ids) < self.batch_size:
            # Дополняем нулями если нужно
            batch_ids = list(batch_ids) + [0] * (self.batch_size - len(batch_ids))
            times = list(times) + [0.0] * (self.batch_size - len(times))
            feedbacks = list(feedbacks) + [0.0] * (self.batch_size - len(feedbacks))

        state = self._build_state(batch_ids, times, feedbacks)
        self.current_state = state
        return state, {}

    def step(self, action: np.ndarray):
        start_idx = self.session_pos + self.batch_size

        # Проверяем окончание сессии
        if start_idx >= len(self.current_session['item_ids']):
            return self.current_state, 0.0, True, False, {}

        # === Извлекаем pos/neg из текущего батча ===
        pos_embs, neg_embs = [], []
        pos_w, neg_w = [], []

        for i in range(self.batch_size):
            idx = i * (self.embedding_dim + 2)
            emb = self.current_state[idx:idx + self.embedding_dim]
            time_val = self.current_state[idx + self.embedding_dim]
            fb_val = self.current_state[idx + self.embedding_dim + 1]

            if fb_val > 0 or time_val > 0.7:
                pos_embs.append(emb)
                pos_w.append(time_val + fb_val)
            if fb_val < 0 or time_val < 0.4:
                neg_embs.append(emb)
                neg_w.append((1 - time_val) + abs(fb_val))

        # Вычисляем целевые эмбеддинги
        target_pos = np.average(pos_embs, axis=0, weights=pos_w) if pos_embs else np.zeros(self.embedding_dim)
        target_neg = np.average(neg_embs, axis=0, weights=neg_w) if neg_embs else np.zeros(self.embedding_dim)

        # Коррекция от action
        final_target = target_pos - 0.5 * target_neg + 0.15 * action
        norm = np.linalg.norm(final_target)
        if norm > 0:
            final_target = final_target / norm

        # === Поиск в Qdrant ===
        try:
            results = self.client.query_points(  # Используем новый метод
                collection_name=self.collection_name,
                query=final_target.tolist(),
                limit=self.batch_size * 10,
                with_payload=True
                ).points
        except Exception as e:
            print(f"Search error: {e}")
            results = []

        # Собираем новые рекомендации
        seen = set(self.current_session['item_ids'][:start_idx])
        new_batch = []

        for point in results:
            item_id = point.payload.get('item_id')
            if item_id and item_id not in seen and item_id not in new_batch:
                new_batch.append(item_id)
            if len(new_batch) == self.batch_size:
                break

        # Fallback если недостаточно рекомендаций
        if len(new_batch) < self.batch_size:
            needed = self.batch_size - len(new_batch)
            all_items = set()
            for session in self.sessions_df['item_ids']:
                all_items.update(session)

            candidates = list(all_items - seen - set(new_batch))
            if len(candidates) > needed:
                new_batch.extend(np.random.choice(candidates, needed, replace=False))
            else:
                new_batch.extend(candidates)
                # Дополняем до batch_size если всё равно не хватает
                while len(new_batch) < self.batch_size:
                    new_batch.append(0)  # placeholder

        # === Получаем реальный фидбек ===
        end_idx = min(start_idx + self.batch_size, len(self.current_session['item_ids']))
        real_times = list(self.current_session['played_ratio_pct'][start_idx:end_idx])
        real_actions = list(self.current_session['actions'][start_idx:end_idx])

        # Дополняем до batch_size если нужно
        if len(real_times) < self.batch_size:
            real_times.extend([0.0] * (self.batch_size - len(real_times)))
            real_actions.extend([0.0] * (self.batch_size - len(real_actions)))

        # Вычисляем reward
        reward = float(np.mean(real_times) + np.mean(real_actions))

        # === Строим новое состояние ===
        self.current_state = self._build_state(new_batch, real_times, real_actions)
        self.session_pos = start_idx

        # Проверяем окончание сессии
        done = (self.session_pos + self.batch_size >= len(self.current_session['item_ids']))
        return self.current_state, reward, done, False, {}

    def _get_emb(self, item_id) -> np.ndarray:
        """Безопасное получение эмбеддинга"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[int(item_id)],
                with_payload=False,
                with_vectors=True
            )
            if points and points[0].vector:
                return np.array(points[0].vector, dtype=np.float32)
        except Exception as e:
            print(f"Embedding error for {item_id}: {e}")
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def _build_state(self, batch_ids, times, feedbacks):
        """Строит состояние гарантированно правильного размера"""
        state = []
        for i in range(self.batch_size):
            # Получаем эмбеддинг
            emb = self._get_emb(batch_ids[i])
            state.extend(emb)

            # Добавляем время и фидбек
            state.append(float(times[i]))
            state.append(float(feedbacks[i]))

        return np.array(state, dtype=np.float32)