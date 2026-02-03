"""
Data Cleaning Module for Super Bowl Analytics

Provides utilities for standardizing data across sources:
- Team abbreviation mapping (Madden <-> NFLverse)
- Player name normalization for fuzzy matching
- Environment categorization (Indoor/Outdoor)
- Betting line standardization

These transformations are essential for joining disparate data sources.
"""

import polars as pl
from typing import Optional


# =============================================================================
# TEAM ABBREVIATION MAPPING
# =============================================================================

# Madden uses different team codes than NFLverse in some cases
# This mapping converts Madden abbreviations to NFLverse standard
MADDEN_TO_NFLVERSE_TEAMS = {
    "TAM": "TB",    # Tampa Bay Buccaneers
    "LVR": "LV",    # Las Vegas Raiders  
    "LAR": "LA",    # Los Angeles Rams
    "GNB": "GB",    # Green Bay Packers
    "KAN": "KC",    # Kansas City Chiefs
    "NWE": "NE",    # New England Patriots
    "NOR": "NO",    # New Orleans Saints
    "SFO": "SF",    # San Francisco 49ers
    "OAK": "LV",    # Oakland Raiders (historical)
    "STL": "LA",    # St. Louis Rams (historical)
    "SDG": "LAC",   # San Diego Chargers (historical)
    "JAC": "JAX",   # Jacksonville Jaguars (varies by source)
}

# Reverse mapping for converting NFLverse to Madden format
NFLVERSE_TO_MADDEN_TEAMS = {v: k for k, v in MADDEN_TO_NFLVERSE_TEAMS.items()}


def standardize_team_abbr(
    team_col: pl.Expr,
    source: str = "madden"
) -> pl.Expr:
    """
    Standardize team abbreviations to NFLverse format.
    
    Parameters
    ----------
    team_col : pl.Expr
        Expression referencing the team column.
    source : str
        Source format: "madden" or "nflverse"
        
    Returns
    -------
    pl.Expr
        Standardized team abbreviation.
        
    Example
    -------
    >>> df.with_columns(
    ...     standardize_team_abbr(pl.col("team"), source="madden")
    ...     .alias("team_std")
    ... )
    """
    if source == "madden":
        mapping = MADDEN_TO_NFLVERSE_TEAMS
    else:
        mapping = {}  # NFLverse is already standard
    
    return team_col.replace(mapping, default=team_col)


# =============================================================================
# PLAYER NAME NORMALIZATION
# =============================================================================

# Common suffixes to strip for matching
NAME_SUFFIXES = ["JR.", "JR", "SR.", "SR", "III", "II", "IV", "V"]


def normalize_player_name(name_col: pl.Expr) -> pl.Expr:
    """
    Normalize player names for cross-source matching.
    
    This is CRITICAL for joining Madden data (full names like "Patrick Mahomes")
    with NFLverse data (abbreviated like "P.Mahomes"). Both formats are 
    converted to a consistent key: "FIRSTINITIALLASTNAME" (e.g., "PMAHOMES").
    
    Transformations:
    1. Convert to uppercase
    2. Remove suffixes (Jr., III, II, IV, Sr.)
    3. Remove punctuation (apostrophes, hyphens, periods)
    4. Extract first initial + last name (handles both formats)
    5. Remove all spaces
    
    Parameters
    ----------
    name_col : pl.Expr
        Expression referencing the name column.
        
    Returns
    -------
    pl.Expr
        Normalized name key for matching.
        
    Example
    -------
    >>> df.with_columns(
    ...     normalize_player_name(pl.col("player_name"))
    ...     .alias("name_normalized")
    ... )
    
    Before: "Patrick Mahomes II"  -> After: "PMAHOMES"
    Before: "P.Mahomes"           -> After: "PMAHOMES"
    Before: "Ja'Marr Chase"       -> After: "JCHASE"
    Before: "Travis Kelce "       -> After: "TKELCE"
    """
    return (
        name_col
        .str.to_uppercase()
        # Remove common suffixes (with optional period)
        .str.replace(r"\s+(JR\.?|SR\.?|III|II|IV|V)$", "")
        # Remove apostrophes, hyphens, periods
        .str.replace_all(r"['\-\.]", "")
        # Strip whitespace
        .str.strip_chars()
        # Extract first letter + last word (works for both "PATRICK MAHOMES" and "PMAHOMES")
        .str.replace(r"^(\w)\w*\s+", "$1")  # "PATRICK MAHOMES" -> "PMAHOMES"
        # Remove any remaining spaces (for single-word names like "PMAHOMES")
        .str.replace_all(r"\s+", "")
    )


def extract_first_last_name(name_col: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """
    Split full name into first and last name components.
    
    Useful for partial matching when full names don't align.
    
    Returns
    -------
    tuple[pl.Expr, pl.Expr]
        (first_name_expr, last_name_expr)
        
    Example
    -------
    >>> first, last = extract_first_last_name(pl.col("name"))
    >>> df.with_columns([first.alias("first"), last.alias("last")])
    """
    first_name = name_col.str.extract(r"^(\w+)", group_index=1)
    last_name = name_col.str.extract(r"\s(\w+)$", group_index=1)
    return first_name, last_name


# =============================================================================
# ENVIRONMENT CATEGORIZATION
# =============================================================================

# Roof types that indicate indoor environment
INDOOR_ROOF_TYPES = ["dome", "closed", "retractable closed"]


def categorize_environment(roof_col: pl.Expr) -> pl.Expr:
    """
    Categorize stadium environment as Indoor or Outdoor.
    
    The 'roof' column in schedule data is notoriously messy,
    with values like "outdoors", "Outdoors", "Open", "Closed", "Dome".
    This function standardizes to binary categories.
    
    Parameters
    ----------
    roof_col : pl.Expr
        Expression referencing the roof column.
        
    Returns
    -------
    pl.Expr
        "Indoor" or "Outdoor" categorization.
        
    Example
    -------
    >>> df.with_columns(
    ...     categorize_environment(pl.col("roof")).alias("environment")
    ... )
    """
    return (
        pl.when(
            roof_col.str.to_lowercase().is_in(["dome", "closed"])
        )
        .then(pl.lit("Indoor"))
        .otherwise(pl.lit("Outdoor"))
    )


def clean_weather_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean and standardize weather columns in schedule data.
    
    Handles missing values and adds derived columns.
    
    Parameters
    ----------
    df : pl.LazyFrame
        Schedule data with temp, wind, roof columns.
        
    Returns
    -------
    pl.LazyFrame
        Cleaned weather data.
    """
    return df.with_columns([
        # Categorize environment
        categorize_environment(pl.col("roof")).alias("environment"),
        
        # Fill missing temp for dome games (assume climate controlled)
        pl.when(pl.col("roof").str.to_lowercase() == "dome")
        .then(pl.lit(72.0))  # Standard dome temperature
        .otherwise(pl.col("temp"))
        .alias("temp_adjusted"),
        
        # Fill missing wind for dome games (no wind indoors)
        pl.when(pl.col("roof").str.to_lowercase() == "dome")
        .then(pl.lit(0.0))
        .otherwise(pl.col("wind"))
        .alias("wind_adjusted"),
        
        # Create weather impact score (higher = more adverse)
        # Cold and windy conditions impact passing games
        pl.when(pl.col("roof").str.to_lowercase() == "dome")
        .then(pl.lit(0.0))
        .otherwise(
            # Penalize cold (below 40°F) and wind (above 15 mph)
            pl.max_horizontal(pl.lit(0), pl.lit(40) - pl.col("temp")) / 10 +
            pl.max_horizontal(pl.lit(0), pl.col("wind") - pl.lit(15)) / 5
        )
        .alias("weather_impact_score"),
    ])


# =============================================================================
# BETTING LINE STANDARDIZATION
# =============================================================================

def calculate_cover_margin(
    home_score_col: pl.Expr,
    away_score_col: pl.Expr,
    spread_col: pl.Expr
) -> pl.Expr:
    """
    Calculate the margin by which the favorite covered the spread.
    
    Spread is typically expressed from home team perspective:
    - Negative spread: Home team favored (e.g., -3.5)
    - Positive spread: Away team favored (e.g., +3.5)
    
    Cover margin > 0: Favorite covered
    Cover margin < 0: Underdog covered
    
    Parameters
    ----------
    home_score_col : pl.Expr
        Home team final score.
    away_score_col : pl.Expr
        Away team final score.
    spread_col : pl.Expr
        Point spread (negative = home favored).
        
    Returns
    -------
    pl.Expr
        Cover margin from favorite's perspective.
        
    Example
    -------
    >>> df.with_columns(
    ...     calculate_cover_margin(
    ...         pl.col("home_score"),
    ...         pl.col("away_score"),
    ...         pl.col("spread_line")
    ...     ).alias("cover_margin")
    ... )
    """
    # Actual margin from home team perspective
    actual_margin = home_score_col - away_score_col
    
    # Cover margin: did the favorite cover?
    # If spread is -3.5, home needs to win by 4+ to cover
    return (actual_margin + spread_col).alias("cover_margin")


def classify_ats_result(cover_margin: pl.Expr) -> pl.Expr:
    """
    Classify the against-the-spread (ATS) result.
    
    Returns
    -------
    pl.Expr
        "Favorite Cover", "Underdog Cover", or "Push"
    """
    return (
        pl.when(cover_margin > 0)
        .then(pl.lit("Favorite Cover"))
        .when(cover_margin < 0)
        .then(pl.lit("Underdog Cover"))
        .otherwise(pl.lit("Push"))
    )


def calculate_total_result(
    home_score_col: pl.Expr,
    away_score_col: pl.Expr,
    total_line_col: pl.Expr
) -> pl.Expr:
    """
    Calculate whether the game went Over or Under the total.
    
    Parameters
    ----------
    home_score_col : pl.Expr
        Home team final score.
    away_score_col : pl.Expr
        Away team final score.
    total_line_col : pl.Expr
        Over/Under line.
        
    Returns
    -------
    pl.Expr
        "Over", "Under", or "Push"
    """
    actual_total = home_score_col + away_score_col
    
    return (
        pl.when(actual_total > total_line_col)
        .then(pl.lit("Over"))
        .when(actual_total < total_line_col)
        .then(pl.lit("Under"))
        .otherwise(pl.lit("Push"))
    )


# =============================================================================
# POSITION STANDARDIZATION
# =============================================================================

# Map various position notations to standard abbreviations
POSITION_MAPPING = {
    "QUARTERBACK": "QB",
    "HALFBACK": "RB",
    "FULLBACK": "FB",
    "WIDE RECEIVER": "WR",
    "TIGHT END": "TE",
    "LEFT TACKLE": "OT",
    "RIGHT TACKLE": "OT",
    "LEFT GUARD": "OG",
    "RIGHT GUARD": "OG",
    "CENTER": "C",
    "LEFT END": "DE",
    "RIGHT END": "DE",
    "DEFENSIVE TACKLE": "DT",
    "OUTSIDE LINEBACKER": "OLB",
    "MIDDLE LINEBACKER": "MLB",
    "INSIDE LINEBACKER": "ILB",
    "CORNERBACK": "CB",
    "FREE SAFETY": "FS",
    "STRONG SAFETY": "SS",
    "KICKER": "K",
    "PUNTER": "P",
    "HB": "RB",
    "SS": "S",
    "FS": "S",
    "OLB": "LB",
    "MLB": "LB",
    "ILB": "LB",
    "RE": "DE",
    "LE": "DE",
    "LT": "OT",
    "RT": "OT",
    "LG": "OG",
    "RG": "OG",
}


def standardize_position(pos_col: pl.Expr) -> pl.Expr:
    """
    Standardize position abbreviations across sources.
    
    Madden uses detailed positions (HB, FS, SS) while
    NFLverse often uses broader categories (RB, S).
    
    Parameters
    ----------
    pos_col : pl.Expr
        Expression referencing the position column.
        
    Returns
    -------
    pl.Expr
        Standardized position abbreviation.
    """
    return pos_col.str.to_uppercase().replace(POSITION_MAPPING, default=pos_col)


# =============================================================================
# COMPREHENSIVE CLEANING PIPELINE
# =============================================================================

def clean_madden_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply all cleaning transformations to Madden ratings data.
    
    This is a convenience function that applies:
    1. Name normalization
    2. Team standardization
    3. Position standardization
    
    Parameters
    ----------
    df : pl.LazyFrame
        Raw Madden ratings data.
        
    Returns
    -------
    pl.LazyFrame
        Cleaned data ready for joining with NFLverse.
    """
    return df.with_columns([
        normalize_player_name(pl.col("player_name")).alias("name_normalized"),
        standardize_team_abbr(pl.col("team"), source="madden").alias("team_std"),
        standardize_position(pl.col("position")).alias("position_std"),
    ])


def clean_nfl_data(df: pl.LazyFrame, name_col: str = "name") -> pl.LazyFrame:
    """
    Apply name normalization to NFLverse data for matching.
    
    Parameters
    ----------
    df : pl.LazyFrame
        NFLverse data with player names.
    name_col : str
        Name of the column containing player names.
        
    Returns
    -------
    pl.LazyFrame
        Data with normalized name column added.
    """
    return df.with_columns([
        normalize_player_name(pl.col(name_col)).alias("name_normalized"),
    ])

