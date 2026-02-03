"""Download 2025 PBP data and merge with existing cache"""
import nflreadpy as nfl
import polars as pl
from pathlib import Path

print("Downloading 2025 PBP data...")
df_2025 = nfl.load_pbp(seasons=[2025])
print(f"Got {len(df_2025)} plays from 2025 season")
print(f"Type: {type(df_2025)}")

# Check if it's already Polars
if isinstance(df_2025, pl.DataFrame):
    df_2025_pl = df_2025
else:
    # Try converting to Polars via Arrow
    import pyarrow as pa
    table = pa.Table.from_pandas(df_2025)
    df_2025_pl = pl.from_arrow(table)
    print("Converted via PyArrow")

print(f"2025 DataFrame: {df_2025_pl.shape}")

# Check for existing cache
cache_path = Path("data/raw/pbp.parquet")

if cache_path.exists():
    print("Loading existing cache...")
    df_existing = pl.read_parquet(cache_path)
    
    # Check existing seasons
    seasons = df_existing.select("season").unique().to_series().to_list()
    print(f"Existing seasons: {min(seasons)} - {max(seasons)}")
    
    if 2025 in seasons:
        print("2025 already in cache, removing old 2025 data...")
        df_existing = df_existing.filter(pl.col("season") != 2025)
    
    # Get column intersection
    common_cols = sorted(set(df_existing.columns) & set(df_2025_pl.columns))
    print(f"Common columns: {len(common_cols)}")
    
    # Combine
    df_combined = pl.concat([
        df_existing.select(common_cols),
        df_2025_pl.select(common_cols)
    ], how="vertical_relaxed")
    print(f"Combined: {len(df_combined)} total plays")
else:
    df_combined = df_2025_pl
    print("No existing cache, using 2025 data only")

# Save
print("Saving to cache...")
df_combined.write_parquet(cache_path)
print(f"✅ Updated cache at {cache_path}")

# Stats
plays_2025 = df_combined.filter(pl.col("season") == 2025)
print(f"\n2025 Season: {len(plays_2025)} plays")
print(f"  SEA: {len(plays_2025.filter(pl.col('posteam') == 'SEA'))} plays")
print(f"  NE:  {len(plays_2025.filter(pl.col('posteam') == 'NE'))} plays")
