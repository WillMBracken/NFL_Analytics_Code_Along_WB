"""
Data Ingestion Module for Super Bowl Analytics

Provides cached wrapper functions for nflreadpy data sources.
Implements the "Extract, Load, Transform" (ELT) pattern with
Parquet checkpointing for efficient reloading during live sessions.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import polars as pl

# nflreadpy is the modern replacement for nfl_data_py
try:
    import nflreadpy as nfl
except ImportError:
    raise ImportError(
        "nflreadpy is required. Install with: pip install nflreadpy"
    )


# Default cache directory (relative to project root, not current working directory)
# Determine project root from this module's location (src/ is one level down from root)
_MODULE_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _MODULE_DIR.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "raw"


def ensure_cache_dir(cache_dir: Path = CACHE_DIR) -> None:
    """Ensure the cache directory exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)


def load_pbp_cached(
    seasons: Union[range, Sequence[int], None] = None,
    cache_path: Optional[str] = None,
    force_refresh: bool = False
) -> pl.LazyFrame:
    """
    Load NFL play-by-play data with intelligent local caching.
    
    This function demonstrates the power of Polars' LazyFrame API:
    - Data is NOT loaded into memory until .collect() is called
    - Enables predicate pushdown for efficient filtering
    - Supports parallel execution across CPU cores
    
    Parameters
    ----------
    seasons : range, list, or None
        Seasons to load. Defaults to 2000-2025 for full coverage.
    cache_path : str, optional
        Path to cache file. Defaults to data/raw/pbp_slim.parquet
    force_refresh : bool
        If True, re-download even if cache exists.
        
    Returns
    -------
    pl.LazyFrame
        Lazy reference to the play-by-play data.
        
    Example
    -------
    >>> pbp = load_pbp_cached(seasons=range(2020, 2025))
    >>> # Nothing loaded yet! Just a query plan.
    >>> qb_plays = pbp.filter(pl.col("passer_id").is_not_null())
    >>> qb_plays.collect()  # NOW it loads and filters
    """
    if seasons is None:
        seasons = range(2000, 2026)
    
    # Use slim parquet by default (29MB vs 293MB - optimized for DataCamp)
    cache_file = Path(cache_path) if cache_path else CACHE_DIR / "pbp_slim.parquet"
    ensure_cache_dir(cache_file.parent)
    
    # Check cache first (unless force refresh)
    if cache_file.exists() and not force_refresh:
        print(f"Loading PBP from cache: {cache_file}")
        return pl.scan_parquet(cache_file)
    
    # Fallback: try full pbp.parquet if slim doesn't exist
    full_cache = CACHE_DIR / "pbp.parquet"
    if full_cache.exists() and not force_refresh:
        print(f"Loading PBP from cache: {full_cache}")
        return pl.scan_parquet(full_cache)
    
    # Download fresh data (only if no cache exists)
    print(f"Downloading PBP data for seasons {list(seasons)}...")
    print("This may take a few minutes on first run.")
    
    # nflreadpy may return Polars or pandas DataFrame depending on version
    df_raw = nfl.load_pbp(seasons=list(seasons))
    
    # Handle both Polars and pandas return types
    if isinstance(df_raw, pl.DataFrame):
        df = df_raw
    else:
        # Assume pandas DataFrame
        df = pl.from_pandas(df_raw)
    
    # Optimize storage: downcast floats and categorize strings
    df = _optimize_pbp_types(df)
    
    # Cache to Parquet for fast future loads
    df.write_parquet(full_cache)
    print(f"Cached to: {full_cache}")
    
    return pl.scan_parquet(full_cache)


def _optimize_pbp_types(df: pl.DataFrame) -> pl.DataFrame:
    """
    Optimize PBP data types for memory efficiency.
    
    Teaching moment: NFL PBP has 300+ columns. Smart type casting
    can reduce memory by 30-50%, crucial for laptops with limited RAM.
    """
    # Identify categorical candidates (low cardinality string columns)
    categorical_cols = [
        "play_type", "pass_location", "run_location", "run_gap",
        "posteam", "defteam", "home_team", "away_team",
        "posteam_type", "game_type", "roof", "surface"
    ]
    
    # Build transformation expressions
    transforms = []
    for col in categorical_cols:
        if col in df.columns:
            transforms.append(pl.col(col).cast(pl.Categorical))
    
    if transforms:
        df = df.with_columns(transforms)
    
    return df


def load_schedules_cached(
    seasons: Union[range, Sequence[int], None] = None,
    cache_path: Optional[str] = None,
    force_refresh: bool = False
) -> pl.LazyFrame:
    """
    Load NFL game schedules with weather and betting data.
    
    Contains crucial metadata:
    - Weather conditions (temp, wind, roof type)
    - Betting lines (spread, total, moneyline)
    - Final scores for result analysis
    
    Parameters
    ----------
    seasons : range, list, or None
        Seasons to load. Defaults to 2015-2024.
    cache_path : str, optional
        Path to cache file.
    force_refresh : bool
        Force re-download if True.
        
    Returns
    -------
    pl.LazyFrame
        Lazy reference to schedule data.
    """
    if seasons is None:
        seasons = range(2015, 2025)
    
    cache_file = Path(cache_path) if cache_path else CACHE_DIR / "schedules.parquet"
    ensure_cache_dir(cache_file.parent)
    
    if cache_file.exists() and not force_refresh:
        print(f"Loading schedules from cache: {cache_file}")
        return pl.scan_parquet(cache_file)
    
    print(f"Downloading schedule data for seasons {list(seasons)}...")
    df_raw = nfl.load_schedules(seasons=list(seasons))
    
    # Handle both Polars and pandas return types
    if isinstance(df_raw, pl.DataFrame):
        df = df_raw
    else:
        df = pl.from_pandas(df_raw)
    
    df.write_parquet(cache_file)
    print(f"Cached to: {cache_file}")
    
    return pl.scan_parquet(cache_file)


def load_players_cached(
    cache_path: Optional[str] = None,
    force_refresh: bool = False
) -> pl.LazyFrame:
    """
    Load player roster data - the "Rosetta Stone" for ID mapping.
    
    Essential for joining PBP data (which uses gsis_id) with
    external sources like Madden ratings (which use names).
    
    Returns
    -------
    pl.LazyFrame
        Player biographical and ID mapping data.
    """
    cache_file = Path(cache_path) if cache_path else CACHE_DIR / "players.parquet"
    ensure_cache_dir(cache_file.parent)
    
    if cache_file.exists() and not force_refresh:
        print(f"Loading players from cache: {cache_file}")
        return pl.scan_parquet(cache_file)
    
    print("Downloading player roster data...")
    df_raw = nfl.load_players()
    
    # Handle both Polars and pandas return types
    if isinstance(df_raw, pl.DataFrame):
        df = df_raw
    else:
        df = pl.from_pandas(df_raw)
    
    df.write_parquet(cache_file)
    print(f"Cached to: {cache_file}")
    
    return pl.scan_parquet(cache_file)


def load_nextgen_stats_cached(
    stat_type: str = "passing",
    seasons: Union[range, Sequence[int], None] = None,
    cache_path: Optional[str] = None,
    force_refresh: bool = False
) -> pl.LazyFrame:
    """
    Load Next Gen Stats tracking data.
    
    NGS provides advanced metrics derived from player tracking:
    - Passing: Time to Throw, Completion Probability, Air Yards
    - Rushing: Time Behind Line, Yards Before Contact
    - Receiving: Average Separation, Cushion, Target Share
    
    Parameters
    ----------
    stat_type : str
        One of: "passing", "rushing", "receiving"
    seasons : range, list, or None
        Seasons to load (NGS available from 2016).
    cache_path : str, optional
        Path to cache file.
    force_refresh : bool
        Force re-download if True.
        
    Returns
    -------
    pl.LazyFrame
        Next Gen Stats data.
    """
    valid_types = ["passing", "rushing", "receiving"]
    if stat_type not in valid_types:
        raise ValueError(f"stat_type must be one of {valid_types}")
    
    if seasons is None:
        seasons = range(2016, 2025)  # NGS starts in 2016
    
    cache_file = Path(cache_path) if cache_path else CACHE_DIR / f"nextgen_{stat_type}.parquet"
    ensure_cache_dir(cache_file.parent)
    
    if cache_file.exists() and not force_refresh:
        print(f"Loading NGS {stat_type} from cache: {cache_file}")
        return pl.scan_parquet(cache_file)
    
    print(f"Downloading Next Gen Stats ({stat_type}) for seasons {list(seasons)}...")
    df_raw = nfl.load_nextgen_stats(seasons=list(seasons), stat_type=stat_type)
    
    # Handle both Polars and pandas return types
    if isinstance(df_raw, pl.DataFrame):
        df = df_raw
    else:
        df = pl.from_pandas(df_raw)
    
    df.write_parquet(cache_file)
    print(f"Cached to: {cache_file}")
    
    return pl.scan_parquet(cache_file)


def load_super_bowl_games(schedules: Optional[pl.LazyFrame] = None) -> pl.LazyFrame:
    """
    Filter schedules to Super Bowl games only.
    
    Convenience function for the Code-Along focus on Super Bowls.
    
    Parameters
    ----------
    schedules : pl.LazyFrame, optional
        Pre-loaded schedules. If None, loads from cache.
        
    Returns
    -------
    pl.LazyFrame
        Super Bowl games only (game_type == "SB").
    """
    if schedules is None:
        schedules = load_schedules_cached()
    
    return schedules.filter(pl.col("game_type") == "SB")


def clear_cache(cache_dir: Path = CACHE_DIR) -> None:
    """
    Clear all cached Parquet files.
    
    Use sparingly - only when you need fresh data or
    to free up disk space.
    """
    import shutil
    
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            f.unlink()
            print(f"Deleted: {f}")
        print("Cache cleared.")
    else:
        print("Cache directory does not exist.")


# Convenience function to load all core data at once
def load_all_core_data(
    seasons: Union[range, Sequence[int], None] = None,
    force_refresh: bool = False
) -> dict[str, pl.LazyFrame]:
    """
    Load all core datasets for Super Bowl analysis.
    
    Returns a dictionary of LazyFrames ready for analysis.
    Great for the initial "data exploration" phase of the Code-Along.
    
    Returns
    -------
    dict[str, pl.LazyFrame]
        Keys: "pbp", "schedules", "players"
    """
    if seasons is None:
        seasons = range(2015, 2025)
    
    return {
        "pbp": load_pbp_cached(seasons, force_refresh=force_refresh),
        "schedules": load_schedules_cached(seasons, force_refresh=force_refresh),
        "players": load_players_cached(force_refresh=force_refresh),
    }

