"""
Visualization Module for Super Bowl Analytics

Provides Plotly chart factory functions for consistent,
portfolio-ready visualizations throughout the course.

Design Philosophy:
- Dark theme for modern, professional look
- Consistent color palette across all charts
- Interactive features (hover, zoom) enabled by default
- Export-ready for portfolio artifacts
"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Union
import numpy as np


# =============================================================================
# THEME CONFIGURATION
# =============================================================================

# Custom color palette inspired by NFL broadcast graphics
NFL_COLORS = {
    "primary": "#013369",      # NFL Blue
    "secondary": "#D50A0A",    # NFL Red
    "accent": "#FFB612",       # Gold
    "success": "#00D084",      # Green
    "warning": "#FF6B35",      # Orange
    "background": "#0D1117",   # Dark background
    "surface": "#161B22",      # Card background
    "text": "#E6EDF3",         # Light text
    "muted": "#7D8590",        # Muted text
}

# Sequential color scale for heatmaps
HEATMAP_COLORSCALE = [
    [0.0, "#0D1117"],
    [0.25, "#013369"],
    [0.5, "#4A90D9"],
    [0.75, "#FFB612"],
    [1.0, "#D50A0A"],
]

# Diverging color scale (for above/below average)
DIVERGING_COLORSCALE = [
    [0.0, "#D50A0A"],
    [0.5, "#E6EDF3"],
    [1.0, "#00D084"],
]


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Apply consistent dark theme to any Plotly figure.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=NFL_COLORS["background"],
        plot_bgcolor=NFL_COLORS["surface"],
        font=dict(
            family="JetBrains Mono, Fira Code, monospace",
            color=NFL_COLORS["text"],
            size=12
        ),
        title=dict(
            font=dict(size=20, color=NFL_COLORS["text"]),
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor=NFL_COLORS["muted"],
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor="rgba(125, 133, 144, 0.2)",
        zerolinecolor=NFL_COLORS["muted"],
        tickfont=dict(color=NFL_COLORS["text"])
    )
    fig.update_yaxes(
        gridcolor="rgba(125, 133, 144, 0.2)",
        zerolinecolor=NFL_COLORS["muted"],
        tickfont=dict(color=NFL_COLORS["text"])
    )
    
    return fig


# =============================================================================
# LINE CHARTS
# =============================================================================

def plot_epa_evolution(
    df: Union[pl.DataFrame, pl.LazyFrame],
    x_col: str = "season",
    y_col: str = "epa_per_play",
    group_col: Optional[str] = None,
    title: str = "EPA Evolution Over Seasons",
    show_trend: bool = True
) -> go.Figure:
    """
    Create a line chart showing EPA evolution over time.
    
    Perfect for the "Era of the Quarterback" module.
    
    Parameters
    ----------
    df : DataFrame or LazyFrame
        Data with season and EPA columns.
    x_col : str
        Column for x-axis (typically season/week).
    y_col : str
        Column for y-axis (EPA metric).
    group_col : str, optional
        Column to create separate lines (e.g., team, player).
    title : str
        Chart title.
    show_trend : bool
        Add trend line if True.
        
    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Convert to pandas for Plotly Express
    pdf = df.to_pandas()
    
    if group_col:
        fig = px.line(
            pdf, x=x_col, y=y_col, color=group_col,
            title=title,
            markers=True,
            color_discrete_sequence=[
                NFL_COLORS["primary"],
                NFL_COLORS["secondary"],
                NFL_COLORS["accent"],
                NFL_COLORS["success"],
            ]
        )
    else:
        fig = px.line(
            pdf, x=x_col, y=y_col,
            title=title,
            markers=True,
        )
        fig.update_traces(
            line=dict(color=NFL_COLORS["accent"], width=3),
            marker=dict(size=8, color=NFL_COLORS["primary"])
        )
    
    # Add trend line
    if show_trend and not group_col:
        x_numeric = pdf[x_col].values
        y_values = pdf[y_col].values
        
        # Simple linear regression
        mask = ~np.isnan(y_values)
        if mask.sum() > 2:
            z = np.polyfit(x_numeric[mask], y_values[mask], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=x_numeric,
                y=p(x_numeric),
                mode="lines",
                name="Trend",
                line=dict(dash="dash", color=NFL_COLORS["muted"], width=2)
            ))
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        hovermode="x unified"
    )
    
    return fig


def plot_rolling_metric(
    df: Union[pl.DataFrame, pl.LazyFrame],
    x_col: str = "play_number",
    y_col: str = "rolling_epa",
    player_col: str = "player_name",
    title: str = "Rolling EPA Performance"
) -> go.Figure:
    """
    Create a multi-player rolling metric comparison.
    
    Shows "hot streaks" and "cold spells" over time.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.to_pandas()
    
    fig = px.line(
        pdf, x=x_col, y=y_col, color=player_col,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig = apply_dark_theme(fig)
    
    # Add zero reference line
    fig.add_hline(
        y=0, line_dash="dash",
        line_color=NFL_COLORS["muted"],
        annotation_text="League Average"
    )
    
    return fig


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def plot_madden_correlation(
    df: Union[pl.DataFrame, pl.LazyFrame],
    x_col: str = "overall",
    y_col: str = "epa_per_play",
    size_col: Optional[str] = None,
    color_col: Optional[str] = None,
    hover_name: str = "player_name",
    title: str = "Madden Rating vs Actual Performance"
) -> go.Figure:
    """
    Create scatter plot comparing Madden ratings to actual performance.
    
    Core visualization for "The Madden Oracle" module.
    
    Parameters
    ----------
    df : DataFrame or LazyFrame
        Joined Madden + performance data.
    x_col : str
        Madden rating column.
    y_col : str
        Performance metric column.
    size_col : str, optional
        Column for bubble size (e.g., dropbacks).
    color_col : str, optional
        Column for color grouping.
    hover_name : str
        Column for hover labels.
    title : str
        Chart title.
        
    Returns
    -------
    go.Figure
        Scatter plot with optional regression line.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.to_pandas()
    
    fig = px.scatter(
        pdf, x=x_col, y=y_col,
        size=size_col,
        color=color_col,
        hover_name=hover_name,
        title=title,
        trendline="ols",  # Ordinary Least Squares regression
        color_discrete_sequence=[NFL_COLORS["primary"], NFL_COLORS["secondary"]],
    )
    
    # Style the main scatter points
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color=NFL_COLORS["text"]),
            opacity=0.8
        ),
        selector=dict(mode="markers")
    )
    
    # Style the trendline
    fig.update_traces(
        line=dict(color=NFL_COLORS["accent"], width=3, dash="solid"),
        selector=dict(mode="lines")
    )
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        xaxis_title="Madden Overall Rating",
        yaxis_title="EPA per Play",
    )
    
    return fig


def plot_outliers(
    df: Union[pl.DataFrame, pl.LazyFrame],
    x_col: str,
    y_col: str,
    label_col: str = "player_name",
    title: str = "Performance Outliers",
    n_labels: int = 10
) -> go.Figure:
    """
    Scatter plot highlighting outliers with annotations.
    
    Automatically labels the most extreme points.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.to_pandas()
    
    fig = px.scatter(
        pdf, x=x_col, y=y_col,
        hover_name=label_col,
        title=title,
        trendline="ols"
    )
    
    # Calculate residuals to find outliers
    x_vals = pdf[x_col].values
    y_vals = pdf[y_col].values
    
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    if mask.sum() > 2:
        z = np.polyfit(x_vals[mask], y_vals[mask], 1)
        p = np.poly1d(z)
        residuals = np.abs(y_vals - p(x_vals))
        
        # Get top outliers
        top_idx = np.argsort(residuals)[-n_labels:]
        
        # Add annotations for outliers
        for idx in top_idx:
            if mask[idx]:
                fig.add_annotation(
                    x=x_vals[idx],
                    y=y_vals[idx],
                    text=pdf[label_col].iloc[idx],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor=NFL_COLORS["accent"],
                    font=dict(size=10, color=NFL_COLORS["text"]),
                    bgcolor=NFL_COLORS["surface"],
                    bordercolor=NFL_COLORS["accent"],
                    borderwidth=1,
                    borderpad=4
                )
    
    fig = apply_dark_theme(fig)
    
    return fig


# =============================================================================
# BOX PLOTS / DISTRIBUTIONS
# =============================================================================

def plot_winner_vs_loser(
    df: Union[pl.DataFrame, pl.LazyFrame],
    metric_col: str = "defensive_epa",
    group_col: str = "result",
    title: str = "Super Bowl Winners vs Losers"
) -> go.Figure:
    """
    Box plot comparing distributions between winners and losers.
    
    Core visualization for "Defense Wins Championships?" module.
    
    Parameters
    ----------
    df : DataFrame or LazyFrame
        Super Bowl team data with result and metric columns.
    metric_col : str
        Metric to compare.
    group_col : str
        Column with "Winner"/"Loser" values.
    title : str
        Chart title.
        
    Returns
    -------
    go.Figure
        Box plot with statistical comparison.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.to_pandas()
    
    fig = go.Figure()
    
    for i, group in enumerate(["Winner", "Loser"]):
        group_data = pdf[pdf[group_col] == group][metric_col]
        
        color = NFL_COLORS["success"] if group == "Winner" else NFL_COLORS["secondary"]
        
        fig.add_trace(go.Box(
            y=group_data,
            name=group,
            marker_color=color,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            marker=dict(
                size=8,
                opacity=0.6,
                line=dict(width=1, color=NFL_COLORS["text"])
            ),
            line=dict(width=2)
        ))
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        title=title,
        yaxis_title=metric_col.replace("_", " ").title(),
        showlegend=False,
    )
    
    # Add median annotation
    for i, group in enumerate(["Winner", "Loser"]):
        group_data = pdf[pdf[group_col] == group][metric_col]
        median = group_data.median()
        fig.add_annotation(
            x=i,
            y=median,
            text=f"Median: {median:.3f}",
            showarrow=False,
            yshift=30,
            font=dict(size=11, color=NFL_COLORS["accent"])
        )
    
    return fig


def plot_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
    value_col: str,
    group_col: Optional[str] = None,
    title: str = "Distribution",
    nbins: int = 30
) -> go.Figure:
    """
    Histogram with optional group comparison.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.to_pandas()
    
    if group_col:
        fig = px.histogram(
            pdf, x=value_col, color=group_col,
            title=title,
            nbins=nbins,
            barmode="overlay",
            opacity=0.7,
            color_discrete_sequence=[NFL_COLORS["primary"], NFL_COLORS["secondary"]]
        )
    else:
        fig = px.histogram(
            pdf, x=value_col,
            title=title,
            nbins=nbins,
            color_discrete_sequence=[NFL_COLORS["primary"]]
        )
    
    fig = apply_dark_theme(fig)
    
    return fig


# =============================================================================
# HEATMAPS
# =============================================================================

def plot_super_bowl_squares(
    df: Union[pl.DataFrame, pl.LazyFrame],
    home_digit_col: str = "home_digit",
    away_digit_col: str = "away_digit",
    title: str = "Super Bowl Squares Probability Matrix"
) -> go.Figure:
    """
    Create a heatmap of score digit probabilities.
    
    THE crowd-pleaser visualization for "Super Bowl Squares" module.
    Shows which numbers are most valuable in office pools.
    
    Parameters
    ----------
    df : DataFrame or LazyFrame
        Score data with digit columns.
    home_digit_col : str
        Column with home team final digit (0-9).
    away_digit_col : str
        Column with away team final digit (0-9).
    title : str
        Chart title.
        
    Returns
    -------
    go.Figure
        Interactive heatmap.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Create pivot table of digit combinations
    pivot = (
        df
        .group_by([home_digit_col, away_digit_col])
        .agg(pl.len().alias("count"))
        .collect() if isinstance(df, pl.LazyFrame) else 
        df.group_by([home_digit_col, away_digit_col]).agg(pl.len().alias("count"))
    )
    
    # Convert to probability matrix
    total = pivot["count"].sum()
    
    # Create 10x10 matrix
    matrix = np.zeros((10, 10))
    
    pdf = pivot.to_pandas()
    for _, row in pdf.iterrows():
        home_d = int(row[home_digit_col])
        away_d = int(row[away_digit_col])
        matrix[away_d, home_d] = row["count"] / total * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(10)),
        y=list(range(10)),
        colorscale=HEATMAP_COLORSCALE,
        text=np.round(matrix, 1),
        texttemplate="%{text}%",
        textfont=dict(size=12, color=NFL_COLORS["text"]),
        hovertemplate="Home: %{x}<br>Away: %{y}<br>Probability: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title="Probability %",
            titleside="right",
            ticksuffix="%"
        )
    ))
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        title=title,
        xaxis_title="Home Team Final Digit",
        yaxis_title="Away Team Final Digit",
        xaxis=dict(dtick=1, tickmode="linear"),
        yaxis=dict(dtick=1, tickmode="linear", autorange="reversed"),
    )
    
    return fig


def plot_correlation_matrix(
    df: Union[pl.DataFrame, pl.LazyFrame],
    columns: list[str],
    title: str = "Feature Correlation Matrix"
) -> go.Figure:
    """
    Create a correlation heatmap for selected features.
    
    Useful for feature selection before modeling.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    pdf = df.select(columns).to_pandas()
    corr_matrix = pdf.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45),
    )
    
    return fig


# =============================================================================
# GAUGE / INDICATOR CHARTS
# =============================================================================

def plot_prediction_gauge(
    probability: float,
    team_name: str,
    title: str = "Super Bowl Win Probability"
) -> go.Figure:
    """
    Create a gauge chart showing win probability prediction.
    
    Perfect for "The Prediction Machine" module finale.
    
    Parameters
    ----------
    probability : float
        Win probability (0-1).
    team_name : str
        Team name for the prediction.
    title : str
        Chart title.
        
    Returns
    -------
    go.Figure
        Gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=48)),
        title=dict(
            text=f"{team_name}<br><span style='font-size:0.6em'>{title}</span>",
            font=dict(size=24)
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=2,
                tickcolor=NFL_COLORS["text"],
                ticksuffix="%"
            ),
            bar=dict(color=NFL_COLORS["accent"], thickness=0.75),
            bgcolor=NFL_COLORS["surface"],
            borderwidth=2,
            bordercolor=NFL_COLORS["muted"],
            steps=[
                dict(range=[0, 40], color=NFL_COLORS["secondary"]),
                dict(range=[40, 60], color=NFL_COLORS["muted"]),
                dict(range=[60, 100], color=NFL_COLORS["success"]),
            ],
            threshold=dict(
                line=dict(color=NFL_COLORS["text"], width=4),
                thickness=0.8,
                value=50
            )
        )
    ))
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        height=400,
    )
    
    return fig


def plot_matchup_comparison(
    team1: str,
    team2: str,
    team1_prob: float,
    team1_stats: dict,
    team2_stats: dict,
    title: str = "Super Bowl Matchup Prediction"
) -> go.Figure:
    """
    Create a side-by-side matchup comparison.
    
    Shows two teams' key stats and predicted win probabilities.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=[team1, team2]
    )
    
    # Team 1 gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=team1_prob * 100,
        number=dict(suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=NFL_COLORS["primary"]),
            bgcolor=NFL_COLORS["surface"],
        )
    ), row=1, col=1)
    
    # Team 2 gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=(1 - team1_prob) * 100,
        number=dict(suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=NFL_COLORS["secondary"]),
            bgcolor=NFL_COLORS["surface"],
        )
    ), row=1, col=2)
    
    fig = apply_dark_theme(fig)
    
    fig.update_layout(
        title=title,
        height=400,
    )
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(
    fig: go.Figure,
    filename: str,
    format: str = "html",
    width: int = 1200,
    height: int = 800
) -> None:
    """
    Save a Plotly figure to file.
    
    Parameters
    ----------
    fig : go.Figure
        Figure to save.
    filename : str
        Output filename (without extension).
    format : str
        Output format: "html", "png", "svg", "pdf"
    width : int
        Image width (for raster formats).
    height : int
        Image height (for raster formats).
    """
    if format == "html":
        fig.write_html(f"{filename}.html", include_plotlyjs="cdn")
    else:
        fig.write_image(
            f"{filename}.{format}",
            width=width,
            height=height,
            scale=2
        )


def show_figure(fig: go.Figure) -> None:
    """
    Display figure in notebook with consistent styling.
    """
    fig.show(config={
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })

