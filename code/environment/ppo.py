import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np
import time
import os
import sys
from dotenv import load_dotenv

# === Путь к проекту ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from config import COLLECTION_NAME, EMBED_DIM, BATCH_SIZE, DATA_READY_PATH
from environment import MusicRecEnv  # твой класс среды

load_dotenv()

# === ПАРАМЕТРЫ ===
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
print(f"Qdrant URL: {QDRANT_URL}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Batch size: {BATCH_SIZE}, Embedding dim: {EMBED_DIM}")

# === СОЗДАЁМ ДИРЕКТОРИИ ===
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === СОЗДАЁМ СРЕДУ ===
print("\n=== Создаём среду ===")
base_env = MusicRecEnv(
    qdrant_url=QDRANT_URL,
    qdrant_api__key=QDRANT_API_KEY,
    sessions_path=DATA_READY_PATH,
    collection_name=COLLECTION_NAME,
    batch_size=BATCH_SIZE,
    embedding_dim=EMBED_DIM
)

# Оборачиваем в Monitor для логирования
env = Monitor(base_env, filename="logs/training_monitor.csv")

# Отдельная среда для оценки
eval_base_env = MusicRecEnv(
    qdrant_url=QDRANT_URL,
    qdrant_api__key=QDRANT_API_KEY,
    sessions_path=DATA_READY_PATH,
    collection_name=COLLECTION_NAME,
    batch_size=BATCH_SIZE,
    embedding_dim=EMBED_DIM
)
eval_env = Monitor(eval_base_env, filename="logs/eval_monitor.csv")

print("✓ Среда создана успешно")

# === CALLBACK ДЛЯ ПРОГРЕСС-БАРА ===
from tqdm import tqdm

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO", unit="step")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
total_timesteps = 900 # Общее количество шагов обучения

# === СОЗДАНИЕ PPO МОДЕЛИ ===
print("\n=== Создаём PPO модель ===")
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,          # Стандартная learning rate для PPO
    n_steps=2048,                # Количество шагов для сбора перед обновлением
    batch_size=64,               # Размер минибатча для оптимизации
    n_epochs=10,                 # Количество эпох оптимизации
    gamma=0.99,                  # Discount factor
    gae_lambda=0.95,             # GAE параметр для advantage estimation
    clip_range=0.2,              # PPO clipping parameter
    clip_range_vf=None,          # Clipping для value function (None = не используется)
    normalize_advantage=True,    # Нормализация advantage
    ent_coef=0.0,                # Entropy coefficient (0 = нет entropy bonus)
    vf_coef=0.5,                 # Value function coefficient
    max_grad_norm=0.5,           # Gradient clipping
    use_sde=False,               # State Dependent Exploration
    sde_sample_freq=-1,
    target_kl=None,              # Target KL divergence (None = не используется)
    tensorboard_log="./logs/ppo_tensorboard",
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Архитектура сети: 2 слоя по 256 нейронов
    ),
    verbose=1,
    seed=42,
    device='auto'  # Автоматически выбирает GPU если доступен
)

print("✓ PPO модель создана")
print(f"\nПараметры модели:")
print(f"  - Learning rate: {model.learning_rate}")
print(f"  - N steps: {model.n_steps}")
print(f"  - Batch size: {model.batch_size}")
print(f"  - N epochs: {model.n_epochs}")
print(f"  - Gamma: {model.gamma}")
print(f"  - GAE lambda: {model.gae_lambda}")
print(f"  - Clip range: {model.clip_range}")

# === НАСТРОЙКА CALLBACKS ===
progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)

# Сохранение чекпоинтов каждые 10000 шагов
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_musicrec",
    save_replay_buffer=False,
    save_vecnormalize=True
)

# Периодическая оценка модели
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model/",
    log_path="./logs/eval_results/",
    eval_freq=5000,  # Оценка каждые 5000 шагов
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)

# === ОБУЧЕНИЕ ===
print("\n" + "="*60)
print("=== НАЧАЛО ОБУЧЕНИЯ PPO ===")
print("="*60)
print(f"Total timesteps: {total_timesteps}")
print(f"Ожидаемое время: ~{total_timesteps/100:.0f} секунд")
print("="*60 + "\n")

start_time = time.time()

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, checkpoint_callback, eval_callback],
        log_interval=10,  # Логировать каждые 10 обновлений
        tb_log_name="PPO_MusicRec",
        reset_num_timesteps=True,
        progress_bar=False  # Отключаем встроенный прогресс-бар, используем свой
    )

    train_time = time.time() - start_time

    print("\n" + "="*60)
    print("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print("="*60)
    print(f"✓ Время обучения: {train_time:.2f} сек ({train_time/60:.2f} мин)")

except KeyboardInterrupt:
    print("\n\n⚠ Обучение прервано пользователем")
    train_time = time.time() - start_time
    print(f"Время до прерывания: {train_time:.2f} сек")

except Exception as e:
    print(f"\n\n✗ ОШИБКА при обучении: {e}")
    import traceback
    traceback.print_exc()
    train_time = time.time() - start_time

# === СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ ===
print("\n=== Сохранение модели ===")
model.save("models/ppo_musicrec_final")
print("✓ Модель сохранена: models/ppo_musicrec_final.zip")

# === ФИНАЛЬНАЯ ОЦЕНКА ===
print("\n=== Финальная оценка модели ===")
try:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,  # Детерминированная политика для оценки
        render=False,
        return_episode_rewards=False
    )

    print(f"\n📊 Результаты оценки:")
    print(f"  Mean reward: {mean_reward:.4f}")
    print(f"  Std reward:  {std_reward:.4f}")

    # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
    results = {
        "model": "PPO",
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "train_time_sec": train_time,
        "train_time_min": train_time / 60,
        "total_timesteps": total_timesteps,
        "learning_rate": float(model.learning_rate),
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "clip_range": float(model.clip_range),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.DataFrame([results])
    results_file = "results/ppo_musicrec_results.csv"

    # Добавляем к существующим результатам если файл есть
    if os.path.exists(results_file):
        df_old = pd.read_csv(results_file)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(results_file, index=False)
    print(f"\n✓ Результаты сохранены: {results_file}")

    print("\n" + "="*60)
    print("=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print("="*60)
    print(df.tail(1).to_string(index=False))
    print("="*60)

except Exception as e:
    print(f"✗ Ошибка при оценке: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Закрываем среды
    env.close()
    eval_env.close()
    print("\n✓ Среды закрыты")

print("\n🎉 ГОТОВО!")
print(f"\n📁 Файлы:")
print(f"  - Модель: models/ppo_musicrec_final.zip")
print(f"  - Лучшая модель: models/best_model/best_model.zip")
print(f"  - Результаты: results/ppo_musicrec_results.csv")
print(f"  - Логи: logs/")
print(f"\n💡 Для просмотра обучения в TensorBoard:")
print(f"  tensorboard --logdir=./logs/ppo_tensorboard")
