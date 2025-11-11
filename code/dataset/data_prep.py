import numpy as np
import pandas as pd
import polars as pl
from config import TIMESTAMP_THRESHOLD, BATCH_SIZE, DROPOUT_BATCH_RATE, DATA_RAW_PATH, DATA_PREP_PATH, EMBED_RAW_PATH

def split_sessions_for_user(row, threshold):
    timestamps = np.array(row['timestamp'], dtype=np.int64)
    if len(timestamps) < 2:
        return [row]  # one session
    
    diffs = np.diff(timestamps)
    split_idx = np.where(diffs > threshold)[0] + 1
    starts = np.concatenate([[0], split_idx])
    ends = np.concatenate([split_idx, [len(timestamps)]])
    
    sessions = []
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        length = end - start

        # Filter the length of the session
        if length < BATCH_SIZE:
            continue
        elif length < 2*BATCH_SIZE:
          if np.random.uniform(0, 1) < DROPOUT_BATCH_RATE:
            continue
        
        sessions.append({
            'uid': row['uid'],
            'session_idx': i,
            'session_length': length,
            'item_ids': row['item_id'][start:end],
            'played_ratio_pct': row['played_ratio_pct'][start:end] if row['played_ratio_pct'] is not None else None,
            'event_type': row['event_type'][start:end]
        })
    return sessions

def clean_item_ids():
    OUTPUT_PATH = 'data/prep/sessions_cleaned.parquet'

    print("Loading embeddings...")
    valid_items = pl.read_parquet(EMBED_RAW_PATH, columns=['item_id'])
    valid_item_ids = valid_items['item_id'].unique().to_list()  # уже уникальные, no extra unique
    print(f"Initial valid item_ids from embeddings: {len(valid_item_ids):,}")

    print("Loading sessions...")
    sessions_pl = pl.read_parquet(DATA_PREP_PATH)
    print(f"Original sessions: {len(sessions_pl):,}")

    # Добавляем пересечение с уникальными item_ids из сессий (ускорение is_in)
    print("Computing intersection with session item_ids...")
    session_item_ids = sessions_pl.select(pl.col('item_ids').explode()).unique().to_series().to_list()
    valid_item_ids = np.intersect1d(session_item_ids, valid_item_ids, assume_unique=True).tolist()
    valid_item_ids_set = set(valid_item_ids)  # для быстрого lookup
    print(f"Optimized valid item_ids (intersection): {len(valid_item_ids):,}")

    # Explode to long format (lazy — no RAM spike yet)
    print("Exploding and filtering...")
    long_pl = sessions_pl.explode(['item_ids', 'played_ratio_pct', 'event_type'])  # timestamps убрал

    # Filter with progress simulation (Polars fast, but add tqdm for estimate)
    # Polars не поддерживает tqdm напрямую, но мы можем симулировать на батчах если нужно
    # Для скорости — делаем в один pass
    long_pl = long_pl.filter(pl.col('item_ids').is_in(valid_item_ids))

    print(f"Filtered rows: {long_pl.height:,}")

    # Group back
    print("Grouping back...")
    grouped = long_pl.group_by(['uid', 'session_idx']).agg([
        pl.col('item_ids').alias('item_ids'),
        pl.col('played_ratio_pct').alias('played_ratio_pct'),
        pl.col('event_type').alias('event_type'),
        pl.len().alias('session_length')
    ])

    # To Pandas or save directly
    cleaned_df = grouped.to_pandas()

    # Statistics (FIXED HERE: col.sum() instead of sum(col))
    total_tracks_before = sessions_pl.select(pl.col('session_length').sum()).item()
    total_tracks_after = cleaned_df['session_length'].sum()
    print(f"Removed tracks: {total_tracks_before - total_tracks_after:,}")

    # Save
    cleaned_df.to_parquet(OUTPUT_PATH)
    print("Done!")


# ----------------- MAIN LOOP -----------------
if __name__ == "__main__":
    print("Loading the entire dataset...")
    df = pd.read_parquet(DATA_RAW_PATH, columns=['uid', 'timestamp', 'item_id', 'played_ratio_pct', 'event_type'])

    all_sessions = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        sessions = split_sessions_for_user(row_dict, TIMESTAMP_THRESHOLD)
        all_sessions.extend(sessions)

    # Create DF and save
    sessions_df = pd.DataFrame(all_sessions)
    sessions_df.to_parquet(DATA_PREP_PATH, engine='pyarrow', index=False, compression='snappy')

    print(f"\nDone! File: {DATA_PREP_PATH}")
    print(f"Number of sessions: {len(sessions_df)}")
    print(sessions_df.head())

    print("Cleaning item_ids...")
    clean_item_ids()