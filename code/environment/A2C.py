import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  
import pandas as pd
import torch
import time
import os
import sys
from dotenv import load_dotenv

import numpy as np  
import matplotlib.pyplot as plt  

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from config import COLLECTION_NAME, EMBED_DIM, BATCH_SIZE, DATA_READY_PATH 
from environment import MusicRecEnv  

load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
print(f"Qdrant URL: {QDRANT_URL}")

env = MusicRecEnv(
    qdrant_url=QDRANT_URL,
    qdrant_api__key=QDRANT_API_KEY,
    sessions_path=DATA_READY_PATH,
    collection_name=COLLECTION_NAME,
    batch_size=BATCH_SIZE,
    embedding_dim=EMBED_DIM
)

from tqdm import tqdm

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class RewardLoggerCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        r = float(rewards[0])
        d = bool(dones[0])

        self._current_reward += r

        if d:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

        return True


models = {
    "A2C": A2C("MlpPolicy", env, verbose=0)
}

results = []
training_histories = {}  
total_timesteps = 900 

for name, model in models.items():
    print(f"\n=== Обучаем {name} ===")
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
    reward_callback = RewardLoggerCallback()
    combined_callback = CallbackList([progress_callback, reward_callback])  

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=combined_callback)
    train_time = time.time() - start_time

    training_histories[name] = reward_callback.episode_rewards.copy()  

    torch.save(model.policy.state_dict(), f"{name}.pth")  

    print(f"Тестируем {name}...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    results.append({
        "model": name,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "train_time_sec": train_time
    })

df = pd.DataFrame(results)
df.to_csv("rl_musicrec_results.csv", index=False)

print("\n=== РЕЗУЛЬТАТЫ ===")
print(df)

plt.figure(figsize=(10, 5))
for name, rewards in training_histories.items():
    if len(rewards) == 0:
        continue
    plt.plot(rewards, label=name)

plt.xlabel("Эпизод")
plt.ylabel("Суммарная награда за эпизод")
plt.title("Кривые обучения (reward per episode)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_rewards.png")
plt.close()


model_names = df["model"].values
mean_rewards = df["mean_reward"].values
std_rewards = df["std_reward"].values

x = np.arange(len(model_names))

plt.figure(figsize=(8, 5))
plt.bar(x, mean_rewards, yerr=std_rewards, capsize=5)
plt.xticks(x, model_names)
plt.ylabel("Средняя награда (± std)")
plt.title("Сравнение моделей по средней награде")
plt.tight_layout()
plt.savefig("eval_results.png")
plt.close()

print("Графики сохранены в файлы: training_rewards.png и eval_results.png")
