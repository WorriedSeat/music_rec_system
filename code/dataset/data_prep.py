import numpy as np
import pandas as pd
from config import TIMESTAMP_THRESHOLD, BATCH_SIZE, DROPOUT_BATCH_RATE, DATA_RAW_PATH, DATA_PREP_PATH

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