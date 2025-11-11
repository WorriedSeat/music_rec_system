import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from config import COLLECTION_NAME, EMBED_RAW_PATH, EMBED_DIM, DATA_PREP_PATH

def upload_embeddings(collection_name:str, item_ids: set, embed_path:str, embed_dim:int, batch_size:int=5000):
    # Подключение к Qdrant
    load_dotenv()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=180
    )

    # Создаём коллекцию (если нет)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    total_loaded = 0

    # Читаем parquet по батчам
    parquet_file = pq.ParquetFile(embed_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['item_id', 'normalized_embed']):
        df_batch = batch.to_pandas()
        
        # Фильтруем только нужные item_id
        mask = df_batch['item_id'].isin(item_ids)
        df_filtered = df_batch[mask]
        
        if len(df_filtered) == 0:
            continue  # Пропускаем пустые батчи
        
        # Формируем точки для Qdrant
        points = []
        for _, row in df_filtered.iterrows():
            vector = np.array(row['normalized_embed'], dtype=np.float32).tolist()
            points.append(
                PointStruct(
                    id=int(row['item_id']),  # Qdrant требует int или str
                    vector=vector,
                    payload={}  # Можно добавить метаданные позже
                )
            )
        
        # Загружаем в Qdrant
        client.upsert(collection_name=collection_name, points=points)
        total_loaded += len(points)
        print(f"Загружено: {total_loaded:,} / {len(item_ids):,} треков")

    print(f"Готово! Загружено {total_loaded:,} эмбеддингов.")

if __name__ == "__main__":
    #Getting all unique item ids
    df = pd.read_parquet(DATA_PREP_PATH)
    item_ids_50m = pd.Series(df['item_ids'].explode()).unique()
    item_ids_50m = set(item_ids_50m)
    upload_embeddings(COLLECTION_NAME, item_ids_50m, EMBED_RAW_PATH, EMBED_DIM)