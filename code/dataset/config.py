TIMESTAMP_THRESHOLD = 900          # 75 minutes
BATCH_SIZE = 5                     # 5 songs to recommend (action) 
DROPOUT_BATCH_RATE = 0.6           # rate to dropout small music sessions
EMBED_DIM = 128

# Paths
DATA_RAW_PATH = 'data/raw/multi_event.parquet'
EMBED_RAW_PATH = 'data/raw/embeddings.parquet'
DATA_PREP_PATH = 'data/prep/sessions_threshold_900.parquet'
DATA_CLEANED_PATH = 'data/prep/sessions_cleaned.parquet'

COLLECTION_NAME = "yambda_50m"
