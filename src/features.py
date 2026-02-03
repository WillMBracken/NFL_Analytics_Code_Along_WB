"""
Feature Engineering Module for Super Bowl Analytics

Provides Polars expressions for advanced NFL metrics:
- EPA (Expected Points Added) calculations
- Rolling statistics and correlations
- Window functions for group-level context
- Aggregation expressions for team/player summaries

These expressions demonstrate the power of Polars' functional API
and can be composed together for complex analyses.
"""

import polars as pl
from typing import Optional, Union


# =============================================================================
# EPA (EXPECTED POINTS ADDED) EXPRESSIONS
# =============================================================================

def epa_above_avg(group_col: str = "season") -> pl.Expr:
    """
    Calculate EPA relative to the group average.
    
    This uses Polars' window function (.over()) to compute
    group-level statistics without collapsing the DataFrame.
    
    Parameters
    ----------
    group_col : str
        Column to group by (default: "season").
        
    Returns
    -------
    pl.Expr
        EPA minus the group mean.
        
    Example
    -------
    >>> df.with_columns(epa_above_avg("season"))
    
    Teaching moment: .over() is like SQL's OVER (PARTITION BY).
    It computes the mean within each group but preserves all rows.
    """
    return (
        pl.col("epa") - pl.col("epa").mean().over(group_col)
    ).alias("epa_above_avg")


def epa_percentile(group_col: str = "season", quantile: float = 0.5) -> pl.Expr:
    """
    Calculate EPA percentile rank within a group.
    
    Parameters
    ----------
    group_col : str
        Column to group by.
    quantile : float
        Percentile threshold (0-1).
        
    Returns
    -------
    pl.Expr
        Boolean indicating if EPA is above the percentile.
    """
    return (
        pl.col("epa") > pl.col("epa").quantile(quantile).over(group_col)
    ).alias(f"epa_above_p{int(quantile*100)}")


def cumulative_epa() -> pl.Expr:
    """
    Calculate cumulative EPA over the course of a game/season.
    
    Useful for visualizing momentum shifts.
    """
    return pl.col("epa").cum_sum().alias("cumulative_epa")


def epa_per_play(
    total_epa_col: str = "total_epa",
    plays_col: str = "plays"
) -> pl.Expr:
    """
    Calculate EPA per play from aggregated totals.
    """
    return (pl.col(total_epa_col) / pl.col(plays_col)).alias("epa_per_play")


# =============================================================================
# ROLLING STATISTICS
# =============================================================================

def rolling_epa(window_size: int = 50) -> pl.Expr:
    """
    Calculate rolling average EPA over recent plays.
    
    Parameters
    ----------
    window_size : int
        Number of plays to include in the window.
        
    Returns
    -------
    pl.Expr
        Rolling mean EPA.
        
    Example
    -------
    >>> qb_plays.with_columns(rolling_epa(100))
    
    This shows a QB's "hot streak" or "cold spell" over recent plays.
    """
    return (
        pl.col("epa")
        .rolling_mean(window_size=window_size, min_periods=10)
        .alias(f"rolling_epa_{window_size}")
    )


def rolling_success_rate(window_size: int = 50) -> pl.Expr:
    """
    Calculate rolling success rate (EPA > 0) over recent plays.
    
    Success rate is a more stable metric than raw EPA.
    """
    return (
        (pl.col("epa") > 0)
        .cast(pl.Float64)
        .rolling_mean(window_size=window_size, min_periods=10)
        .alias(f"rolling_success_rate_{window_size}")
    )


def rolling_pass_rate(window_size: int = 50) -> pl.Expr:
    """
    Calculate rolling pass rate over recent plays.
    
    Useful for analyzing team tendencies and game scripts.
    """
    return (
        (pl.col("play_type") == "pass")
        .cast(pl.Float64)
        .rolling_mean(window_size=window_size, min_periods=10)
        .alias(f"rolling_pass_rate_{window_size}")
    )


def rolling_yards_per_play(window_size: int = 50) -> pl.Expr:
    """
    Calculate rolling yards per play.
    """
    return (
        pl.col("yards_gained")
        .rolling_mean(window_size=window_size, min_periods=10)
        .alias(f"rolling_ypp_{window_size}")
    )


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def pass_rate_wp_correlation(window_size: int = 100) -> pl.Expr:
    """
    Calculate rolling correlation between pass rate and win probability.
    
    This reveals whether teams pass more when ahead (garbage time)
    or behind (desperation), and how this relates to win probability.
    
    Parameters
    ----------
    window_size : int
        Window for rolling correlation.
        
    Returns
    -------
    pl.Expr
        Rolling correlation coefficient.
        
    Note
    ----
    Polars' rolling_corr is MUCH faster than a manual calculation.
    This is a key teaching point in the course.
    """
    # Create binary pass indicator
    is_pass = (pl.col("play_type") == "pass").cast(pl.Float64)
    
    return (
        pl.rolling_corr(
            is_pass,
            pl.col("wp"),
            window_size=window_size,
            min_periods=20
        )
        .alias("pass_wp_correlation")
    )


# =============================================================================
# WINDOW FUNCTIONS FOR GROUP CONTEXT
# =============================================================================

def rank_within_group(
    value_col: str,
    group_col: str,
    descending: bool = True
) -> pl.Expr:
    """
    Rank values within groups.
    
    Parameters
    ----------
    value_col : str
        Column to rank by.
    group_col : str
        Column to group by.
    descending : bool
        If True, higher values get lower ranks (rank 1 = best).
        
    Returns
    -------
    pl.Expr
        Rank within group.
        
    Example
    -------
    >>> # Rank QBs by EPA within each season
    >>> df.with_columns(rank_within_group("total_epa", "season"))
    """
    return (
        pl.col(value_col)
        .rank(method="ordinal", descending=descending)
        .over(group_col)
        .alias(f"{value_col}_rank")
    )


def pct_of_group_total(
    value_col: str,
    group_col: str
) -> pl.Expr:
    """
    Calculate value as percentage of group total.
    
    Example: QB's EPA as % of team's total EPA.
    """
    return (
        pl.col(value_col) / pl.col(value_col).sum().over(group_col) * 100
    ).alias(f"{value_col}_pct_of_total")


def lag_value(
    value_col: str,
    periods: int = 1,
    group_col: Optional[str] = None
) -> pl.Expr:
    """
    Get lagged values, optionally within groups.
    
    Useful for comparing to previous season/game.
    
    Parameters
    ----------
    value_col : str
        Column to lag.
    periods : int
        Number of periods to lag.
    group_col : str, optional
        If provided, lag within groups.
        
    Returns
    -------
    pl.Expr
        Lagged values.
    """
    expr = pl.col(value_col).shift(periods)
    
    if group_col:
        expr = expr.over(group_col)
    
    return expr.alias(f"{value_col}_lag{periods}")


def delta_from_previous(
    value_col: str,
    group_col: Optional[str] = None
) -> pl.Expr:
    """
    Calculate change from previous value.
    
    Example: Season-over-season EPA change.
    """
    current = pl.col(value_col)
    previous = lag_value(value_col, 1, group_col)
    
    return (current - previous).alias(f"{value_col}_delta")


# =============================================================================
# AGGREGATION EXPRESSIONS
# =============================================================================

def qb_season_stats() -> list[pl.Expr]:
    """
    Standard QB season aggregation expressions.
    
    Returns a list of expressions to use with group_by().agg().
    
    Example
    -------
    >>> pbp.group_by(["passer_id", "season"]).agg(qb_season_stats())
    """
    return [
        # Volume metrics
        pl.len().alias("dropbacks"),
        pl.col("pass_attempt").sum().alias("attempts"),
        pl.col("complete_pass").sum().alias("completions"),
        pl.col("passing_yards").sum().alias("passing_yards"),
        pl.col("pass_touchdown").sum().alias("touchdowns"),
        pl.col("interception").sum().alias("interceptions"),
        
        # Efficiency metrics
        pl.col("epa").sum().alias("total_epa"),
        pl.col("epa").mean().alias("epa_per_dropback"),
        (pl.col("epa") > 0).mean().alias("success_rate"),
        
        # Advanced metrics
        pl.col("cpoe").mean().alias("avg_cpoe"),
        pl.col("air_yards").mean().alias("avg_air_yards"),
        pl.col("yards_after_catch").mean().alias("avg_yac"),
        
        # Context
        pl.col("posteam").first().alias("team"),
        pl.col("name").first().alias("player_name"),
    ]


def team_game_stats() -> list[pl.Expr]:
    """
    Team game-level aggregation expressions.
    
    Returns
    -------
    list[pl.Expr]
        Expressions for group_by([game_id, posteam]).agg()
    """
    return [
        # Overall
        pl.len().alias("plays"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("epa").mean().alias("epa_per_play"),
        (pl.col("epa") > 0).mean().alias("success_rate"),
        
        # Passing
        (pl.col("play_type") == "pass").sum().alias("pass_plays"),
        pl.col("passing_yards").sum().alias("passing_yards"),
        pl.col("pass_touchdown").sum().alias("pass_tds"),
        pl.col("interception").sum().alias("interceptions"),
        
        # Rushing  
        (pl.col("play_type") == "run").sum().alias("rush_plays"),
        pl.col("rushing_yards").sum().alias("rushing_yards"),
        pl.col("rush_touchdown").sum().alias("rush_tds"),
        
        # Situational
        pl.col("third_down_converted").sum().alias("third_down_conv"),
        pl.col("third_down_failed").sum().alias("third_down_fail"),
    ]


def defensive_stats() -> list[pl.Expr]:
    """
    Defensive team aggregation expressions.
    
    Note: Uses defteam column to aggregate plays allowed.
    """
    return [
        pl.len().alias("plays_allowed"),
        pl.col("epa").sum().alias("epa_allowed"),
        pl.col("epa").mean().alias("epa_per_play_allowed"),
        (pl.col("epa") > 0).mean().alias("opp_success_rate"),
        pl.col("sack").sum().alias("sacks"),
        pl.col("interception").sum().alias("turnovers_forced"),
    ]


# =============================================================================
# SUPER BOWL SPECIFIC FEATURES
# =============================================================================

def super_bowl_features() -> list[pl.Expr]:
    """
    Features specifically useful for Super Bowl analysis.
    """
    return [
        # Game totals
        (pl.col("home_score") + pl.col("away_score")).alias("total_points"),
        
        # Margin of victory
        (pl.col("home_score") - pl.col("away_score")).abs().alias("margin"),
        
        # Score digit for Super Bowl Squares
        (pl.col("home_score") % 10).alias("home_digit"),
        (pl.col("away_score") % 10).alias("away_digit"),
        
        # Close game indicator
        ((pl.col("home_score") - pl.col("away_score")).abs() <= 7)
        .alias("one_score_game"),
        
        # Blowout indicator
        ((pl.col("home_score") - pl.col("away_score")).abs() >= 14)
        .alias("blowout"),
    ]


def madden_curse_features() -> list[pl.Expr]:
    """
    Features for analyzing the "Madden Curse".
    
    Compares current season performance to next season.
    """
    return [
        # Performance delta
        (pl.col("epa_next_season") - pl.col("epa_current_season"))
        .alias("epa_change"),
        
        # Games missed
        (pl.col("games_current") - pl.col("games_next"))
        .alias("games_lost"),
        
        # Rating change
        (pl.col("madden_ovr_next") - pl.col("madden_ovr_current"))
        .alias("rating_change"),
    ]


# =============================================================================
# COMPOSITE FEATURE BUILDER
# =============================================================================

def build_qb_features(
    pbp: pl.LazyFrame,
    seasons: Optional[list[int]] = None
) -> pl.LazyFrame:
    """
    Build comprehensive QB feature set from play-by-play data.
    
    This is a convenience function that:
    1. Filters to passing plays
    2. Groups by QB and season
    3. Calculates standard metrics
    4. Adds rankings and percentiles
    
    Parameters
    ----------
    pbp : pl.LazyFrame
        Play-by-play data.
    seasons : list[int], optional
        Seasons to include.
        
    Returns
    -------
    pl.LazyFrame
        QB season-level features.
    """
    # Filter to passing plays
    qb_plays = pbp.filter(
        pl.col("play_type") == "pass",
        pl.col("passer_id").is_not_null()
    )
    
    if seasons:
        qb_plays = qb_plays.filter(pl.col("season").is_in(seasons))
    
    # Aggregate to QB-season level
    qb_stats = qb_plays.group_by(["passer_id", "season"]).agg(qb_season_stats())
    
    # Add rankings within season
    qb_stats = qb_stats.with_columns([
        rank_within_group("total_epa", "season"),
        rank_within_group("epa_per_dropback", "season"),
        rank_within_group("success_rate", "season"),
    ])
    
    # Add season-over-season delta
    qb_stats = qb_stats.sort(["passer_id", "season"]).with_columns([
        delta_from_previous("total_epa", "passer_id"),
        delta_from_previous("epa_per_dropback", "passer_id"),
    ])
    
    return qb_stats


def build_team_features(
    pbp: pl.LazyFrame,
    side: str = "offense"
) -> pl.LazyFrame:
    """
    Build team-level features from play-by-play data.
    
    Parameters
    ----------
    pbp : pl.LazyFrame
        Play-by-play data.
    side : str
        "offense" or "defense"
        
    Returns
    -------
    pl.LazyFrame
        Team season-level features.
    """
    team_col = "posteam" if side == "offense" else "defteam"
    
    team_stats = (
        pbp
        .filter(pl.col("play_type").is_in(["pass", "run"]))
        .group_by([team_col, "season"])
        .agg(team_game_stats() if side == "offense" else defensive_stats())
    )
    
    return team_stats.with_columns([
        rank_within_group("total_epa" if side == "offense" else "epa_allowed", "season"),
    ])


# =============================================================================
# ENHANCED SUPER BOWL PREDICTION FEATURES
# =============================================================================

def team_offensive_features() -> list[pl.Expr]:
    """
    Comprehensive offensive team features for prediction model.
    
    Returns aggregation expressions for offensive metrics.
    """
    return [
        # Overall EPA
        pl.col("epa").sum().alias("off_epa"),
        pl.col("epa").mean().alias("off_epa_per_play"),
        (pl.col("epa") > 0).mean().alias("off_success_rate"),
        
        # Pass-specific EPA
        pl.col("epa").filter(pl.col("play_type") == "pass").sum().alias("pass_epa"),
        pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("pass_epa_per_play"),
        
        # Rush-specific EPA
        pl.col("epa").filter(pl.col("play_type") == "run").sum().alias("rush_epa"),
        pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("rush_epa_per_play"),
        
        # Explosive play rate (EPA > 1.5)
        (pl.col("epa") > 1.5).mean().alias("explosive_rate"),
        
        # Red zone efficiency (inside opponent 20)
        pl.col("epa").filter(pl.col("yardline_100") <= 20).sum().alias("redzone_epa"),
        pl.col("epa").filter(pl.col("yardline_100") <= 20).mean().alias("redzone_epa_per_play"),
        
        # Third down efficiency
        pl.col("epa").filter(pl.col("down") == 3).sum().alias("third_down_epa"),
        pl.col("epa").filter(pl.col("down") == 3).mean().alias("third_down_epa_per_play"),
        
        # Turnovers
        pl.col("interception").sum().alias("interceptions_thrown"),
        pl.col("fumble_lost").sum().alias("fumbles_lost"),
    ]


def team_defensive_features() -> list[pl.Expr]:
    """
    Comprehensive defensive team features for prediction model.
    
    Returns aggregation expressions for defensive metrics.
    """
    return [
        # Overall EPA allowed
        pl.col("epa").sum().alias("def_epa_allowed"),
        pl.col("epa").mean().alias("def_epa_per_play"),
        (pl.col("epa") > 0).mean().alias("opp_success_rate"),
        
        # Pressure/Pass rush
        pl.col("sack").sum().alias("sacks"),
        pl.col("qb_hit").sum().alias("qb_hits"),
        
        # Turnovers forced
        pl.col("interception").sum().alias("interceptions"),
        pl.col("fumble_forced").sum().alias("fumbles_forced"),
        
        # Red zone defense
        pl.col("epa").filter(pl.col("yardline_100") <= 20).sum().alias("redzone_epa_allowed"),
        
        # Third down defense
        pl.col("epa").filter(pl.col("down") == 3).sum().alias("third_down_epa_allowed"),
    ]


def build_team_season_features(pbp: pl.LazyFrame) -> pl.DataFrame:
    """
    Build comprehensive team-season features from play-by-play data.
    
    This function creates all offensive and defensive features needed
    for the Super Bowl prediction model.
    
    Parameters
    ----------
    pbp : pl.LazyFrame
        Play-by-play data.
        
    Returns
    -------
    pl.DataFrame
        Team-season level features.
    """
    # Filter to regular season plays
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"]),
        pl.col("season_type") == "REG"
    )
    
    # Offensive features (using posteam)
    offense = (
        plays
        .group_by(["posteam", "season"])
        .agg(team_offensive_features())
        .collect()
    )
    
    # Defensive features (using defteam)
    defense = (
        plays
        .group_by(["defteam", "season"])
        .agg(team_defensive_features())
        .collect()
    )
    
    # Join offense and defense
    team_features = offense.join(
        defense,
        left_on=["posteam", "season"],
        right_on=["defteam", "season"],
        how="left"
    )
    
    # Calculate derived metrics
    team_features = team_features.with_columns([
        # Turnover differential
        (
            (pl.col("interceptions") + pl.col("fumbles_forced")) -
            (pl.col("interceptions_thrown") + pl.col("fumbles_lost"))
        ).alias("turnover_diff"),
        
        # Pressure rate (sacks + hits per passing play)
        # Estimate passing plays as ~60% of total
        ((pl.col("sacks") + pl.col("qb_hits")) / (pl.len() * 0.6)).alias("pressure_rate"),
    ])
    
    return team_features


def build_playoff_features(pbp: pl.LazyFrame, schedules: pl.LazyFrame) -> pl.DataFrame:
    """
    Build playoff-specific features from play-by-play data.
    
    Playoff performance can differ significantly from regular season.
    
    Parameters
    ----------
    pbp : pl.LazyFrame
        Play-by-play data.
    schedules : pl.LazyFrame
        Schedule data (to identify playoff games).
        
    Returns
    -------
    pl.DataFrame
        Playoff EPA features by team-season.
    """
    # Filter to playoff plays only
    playoff_plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"]),
        pl.col("season_type") == "POST"
    )
    
    # Aggregate playoff EPA
    playoff_features = (
        playoff_plays
        .group_by(["posteam", "season"])
        .agg([
            pl.col("epa").sum().alias("playoff_off_epa"),
            pl.col("epa").mean().alias("playoff_off_epa_per_play"),
            pl.len().alias("playoff_plays"),
        ])
        .collect()
    )
    
    return playoff_features


def build_madden_features(madden_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract team-level features from Madden ratings data.
    
    Parameters
    ----------
    madden_df : pl.DataFrame
        Madden player ratings DataFrame.
        
    Returns
    -------
    pl.DataFrame
        Team-level Madden features by season.
    """
    # Cast season to int if needed
    if madden_df["season"].dtype == pl.Utf8:
        madden_df = madden_df.with_columns(pl.col("season").cast(pl.Int32))
    
    # QB overall rating
    qb_ratings = (
        madden_df
        .filter(pl.col("position") == "QB")
        .group_by(["team", "season"])
        .agg(pl.col("overall").max().alias("qb_ovr"))
    )
    
    # Team average and max ratings
    team_ratings = (
        madden_df
        .filter(pl.col("team").is_not_null())
        .group_by(["team", "season"])
        .agg([
            pl.col("overall").mean().alias("avg_starter_ovr"),
            pl.col("overall").max().alias("max_ovr"),
            pl.col("overall").min().alias("min_ovr"),
        ])
    )
    
    # Join QB ratings to team ratings
    madden_features = team_ratings.join(
        qb_ratings,
        on=["team", "season"],
        how="left"
    )
    
    return madden_features


def spread_to_win_prob(spread: float) -> float:
    """
    Convert point spread to implied win probability.
    
    Uses the approximation: win_prob = 0.5 - (spread / 14)
    This is a simplified model; actual conversion is more complex.
    
    Parameters
    ----------
    spread : float
        Point spread (negative means favored).
        
    Returns
    -------
    float
        Implied win probability (0-1).
    """
    # Clamp to reasonable range
    prob = 0.5 - (spread / 14)
    return max(0.05, min(0.95, prob))


def build_super_bowl_features(
    pbp: pl.LazyFrame,
    schedules: pl.LazyFrame,
    madden_df: pl.DataFrame,
    sb_games: pl.DataFrame
) -> pl.DataFrame:
    """
    Build the complete feature matrix for Super Bowl prediction.
    
    This function orchestrates all feature extraction and creates
    differential features for home vs away team comparisons.
    
    Parameters
    ----------
    pbp : pl.LazyFrame
        Play-by-play data.
    schedules : pl.LazyFrame
        Schedule data with betting lines.
    madden_df : pl.DataFrame
        Madden player ratings.
    sb_games : pl.DataFrame
        Super Bowl games with teams and outcomes.
        
    Returns
    -------
    pl.DataFrame
        Complete feature matrix for modeling.
    """
    # Build base features
    team_features = build_team_season_features(pbp)
    playoff_features = build_playoff_features(pbp, schedules)
    madden_features = build_madden_features(madden_df)
    
    # Cast team columns to string for joining
    team_features = team_features.with_columns(pl.col("posteam").cast(pl.Utf8))
    playoff_features = playoff_features.with_columns(pl.col("posteam").cast(pl.Utf8))
    
    # Join playoff features
    team_features = team_features.join(
        playoff_features,
        on=["posteam", "season"],
        how="left"
    )
    
    # Join Madden features
    team_features = team_features.join(
        madden_features,
        left_on=["posteam", "season"],
        right_on=["team", "season"],
        how="left"
    )
    
    # Now create matchup-level features for each Super Bowl
    # Join home team features
    sb_features = sb_games.join(
        team_features.rename(lambda c: f"home_{c}" if c not in ["posteam", "season"] else c),
        left_on=["home_team", "season"],
        right_on=["posteam", "season"],
        how="left"
    )
    
    # Join away team features
    sb_features = sb_features.join(
        team_features.rename(lambda c: f"away_{c}" if c not in ["posteam", "season"] else c),
        left_on=["away_team", "season"],
        right_on=["posteam", "season"],
        how="left"
    )
    
    # Calculate differential features (home - away)
    diff_features = [
        # EPA differentials
        (pl.col("home_off_epa") - pl.col("away_off_epa")).alias("off_epa_diff"),
        (pl.col("home_def_epa_allowed") - pl.col("away_def_epa_allowed")).alias("def_epa_diff"),
        (pl.col("home_pass_epa") - pl.col("away_pass_epa")).alias("pass_epa_diff"),
        (pl.col("home_rush_epa") - pl.col("away_rush_epa")).alias("rush_epa_diff"),
        
        # Success and explosive rate differentials
        (pl.col("home_off_success_rate") - pl.col("away_off_success_rate")).alias("success_rate_diff"),
        (pl.col("home_explosive_rate") - pl.col("away_explosive_rate")).alias("explosive_rate_diff"),
        
        # Situational differentials
        (pl.col("home_redzone_epa") - pl.col("away_redzone_epa")).alias("redzone_epa_diff"),
        (pl.col("home_third_down_epa") - pl.col("away_third_down_epa")).alias("third_down_epa_diff"),
        
        # Turnover differential
        (pl.col("home_turnover_diff") - pl.col("away_turnover_diff")).alias("turnover_diff"),
        
        # Playoff EPA differential (if available)
        (
            pl.col("home_playoff_off_epa").fill_null(0) - 
            pl.col("away_playoff_off_epa").fill_null(0)
        ).alias("playoff_epa_diff"),
        
        # Madden differentials
        (pl.col("home_qb_ovr").fill_null(80) - pl.col("away_qb_ovr").fill_null(80)).alias("qb_ovr_diff"),
        (pl.col("home_avg_starter_ovr").fill_null(80) - pl.col("away_avg_starter_ovr").fill_null(80)).alias("avg_ovr_diff"),
        (pl.col("home_max_ovr").fill_null(90) - pl.col("away_max_ovr").fill_null(90)).alias("max_ovr_diff"),
        
        # Market features
        pl.col("spread_line").fill_null(0).alias("spread_line"),
        pl.col("total_line").fill_null(47).alias("total_line"),
        
        # Implied win probability from spread
        (0.5 - pl.col("spread_line").fill_null(0) / 14).clip(0.05, 0.95).alias("implied_win_prob"),
        
        # Target variable
        (pl.col("home_team") == pl.col("winner")).cast(pl.Int32).alias("home_win"),
    ]
    
    sb_features = sb_features.with_columns(diff_features)
    
    return sb_features


def get_feature_columns() -> list[str]:
    """
    Return the list of feature columns for the prediction model.
    
    Returns
    -------
    list[str]
        Feature column names in order.
    """
    return [
        # EPA differentials
        "off_epa_diff",
        "def_epa_diff",
        "pass_epa_diff",
        "rush_epa_diff",
        
        # Rate differentials
        "success_rate_diff",
        "explosive_rate_diff",
        
        # Situational
        "redzone_epa_diff",
        "third_down_epa_diff",
        
        # Turnovers
        "turnover_diff",
        
        # Playoffs
        "playoff_epa_diff",
        
        # Madden ratings
        "qb_ovr_diff",
        "avg_ovr_diff",
        "max_ovr_diff",
        
        # Market
        "spread_line",
        "total_line",
        "implied_win_prob",
    ]

