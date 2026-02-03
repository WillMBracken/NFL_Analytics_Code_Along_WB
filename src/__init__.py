"""
Super Bowl Analytics with Polars - Source Module

This package provides reusable utilities for NFL data analysis:
- ingestion: Cached data loading from nflreadpy
- cleaning: Name normalization and team mapping
- features: EPA calculations and rolling statistics  
- viz: Plotly chart factory functions
"""

from . import ingestion
from . import cleaning
from . import features
from . import viz

__version__ = "1.0.0"
__all__ = ["ingestion", "cleaning", "features", "viz"]

