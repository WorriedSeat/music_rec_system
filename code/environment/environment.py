import numpy as np
import gymnasium as gym
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class MusicRecEnv(gym.Env):
    def __init__(self, embeddings, all_tracks):
        super().__init__()
        self.embeddings = embeddings
        self.all_tracks = np.array(all_tracks)  # Для быстрого доступа
        self.batch_size = BATCH_SIZE
        self.current_batch = None
        self.current_state = None
        
        # Action: continuous vector для "коррекции" target
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(EMBED_DIM,))
        
        # State: flattened [embed1, time1, feedback1, embed2, ...]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,))
        
        # Для KNN: fit на всех эмбеддингах
        self.all_embs = np.array([self.embeddings[t] for t in self.all_tracks])
        self.nn = NearestNeighbors(n_neighbors=self.batch_size * 2, metric='cosine')  # *2 для кандидатов
        self.nn.fit(self.all_embs)
    
    def reset(self, seed=None):
        # Старт: случайный батч
        self.current_batch = np.random.choice(self.all_tracks, self.batch_size, replace=False)
        # Симулируем начальный фидбек (на старте случайный или нулевой)
        times = np.random.uniform(0.3, 0.7, self.batch_size)  # Placeholder
        feedbacks = np.random.choice([-1, 0, 1], self.batch_size, p=[0.2, 0.4, 0.4])
        self.current_state = self._build_state(self.current_batch, times, feedbacks)
        return self.current_state, {}
    
    def step(self, action):
        # Вычисляем target на основе state (weighted pos/neg)
        pos_embs, neg_embs, pos_weights, neg_weights = self._extract_pos_neg_from_state()
        if len(pos_embs) > 0:
            target_pos = np.average(pos_embs, axis=0, weights=pos_weights)
        else:
            target_pos = np.zeros(EMBED_DIM)
        if len(neg_embs) > 0:
            target_neg = np.average(neg_embs, axis=0, weights=neg_weights)
        else:
            target_neg = np.zeros(EMBED_DIM)
        
        # Action как коррекция: final_target = pos - 0.5*neg + action (нормализовать)
        final_target = target_pos - 0.5 * target_neg + 0.1 * action  # 0.1 для слабого влияния action
        final_target /= np.linalg.norm(final_target) or 1.0  # Нормализуем
        
        # Выбор нового батча: KNN по final_target, фильтр на уникальность и избегание повторов
        distances, indices = self.nn.kneighbors([final_target])
        candidates = self.all_tracks[indices[0]]
        new_batch = []
        for cand in candidates:
            if cand not in self.current_batch and cand not in new_batch:  # Избегай повторов
                new_batch.append(cand)
            if len(new_batch) == self.batch_size:
                break
        
        # Симулируем фидбек для нового батча (позже замени на реальный или из датасета)
        times = np.random.uniform(0.2, 0.8, self.batch_size)  # Placeholder: в реале на основе сходства или симуляции пользователя
        feedbacks = np.random.choice([-1, 0, 1], self.batch_size, p=[0.3, 0.3, 0.4])
        
        # Reward
        likes = np.sum(feedbacks == 1)
        dislikes = np.sum(feedbacks == -1)
        reward = np.mean(times) + (likes * 1.0 - dislikes * 0.5)
        
        # Обновляем state и batch
        self.current_state = self._build_state(new_batch, times, feedbacks)
        self.current_batch = new_batch
        
        done = False  # Или True после N шагов, e.g., if episode_length > 10
        return self.current_state, reward, done, False, {}
    
    def _build_state(self, batch, times, feedbacks):
        state_parts = []
        for i in range(self.batch_size):
            emb = self.embeddings[batch[i]]
            state_parts.extend(emb.tolist())
            state_parts.append(times[i])
            state_parts.append(feedbacks[i])
        return np.array(state_parts)
    
    def _extract_pos_neg_from_state(self):
        pos_embs, neg_embs = [], []
        pos_weights, neg_weights = [], []
        for i in range(self.batch_size):
            start = i * (EMBED_DIM + 2)
            emb = self.current_state[start:start+EMBED_DIM]
            time = self.current_state[start+EMBED_DIM]
            fb = self.current_state[start+EMBED_DIM+1]
            if fb > 0 or time > 0.5:  # Pos if like or high time
                pos_embs.append(emb)
                pos_weights.append(time * (fb + 1))  # Вес = time * (1+fb)
            elif fb < 0 or time < 0.3:  # Neg if dislike or low time
                neg_embs.append(emb)
                neg_weights.append((1 - time) * (-fb + 1))  # Инвертируем для веса
        return pos_embs, neg_embs, pos_weights, neg_weights