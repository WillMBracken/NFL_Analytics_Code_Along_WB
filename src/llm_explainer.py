"""
LLM Explainer Module for Super Bowl Analytics

Provides natural language explanations for model predictions using
either Ollama (local LLM server) or Hugging Face Transformers.

Both options are free and run locally without API costs.
"""

from typing import Optional, Dict, Any
import warnings


# =============================================================================
# FEATURE DEFINITIONS (for LLM context)
# =============================================================================

FEATURE_DEFINITIONS = {
    # EPA differentials
    "off_epa_diff": {
        "name": "Offensive Efficiency",
        "description": "Overall offensive performance difference (higher = better offense)",
        "positive_meaning": "Home team has a more efficient offense",
        "negative_meaning": "Away team has a more efficient offense"
    },
    "def_epa_diff": {
        "name": "Defensive Efficiency",
        "description": "Points allowed per play difference (lower/negative = better defense)",
        "positive_meaning": "Home team allows more points (worse defense)",
        "negative_meaning": "Home team allows fewer points (better defense)"
    },
    "pass_epa_diff": {
        "name": "Passing Attack",
        "description": "Passing game efficiency difference",
        "positive_meaning": "Home team has a stronger passing game",
        "negative_meaning": "Away team has a stronger passing game"
    },
    "rush_epa_diff": {
        "name": "Running Game",
        "description": "Rushing efficiency difference",
        "positive_meaning": "Home team has a more effective running game",
        "negative_meaning": "Away team has a more effective running game"
    },
    
    # Rate differentials
    "success_rate_diff": {
        "name": "Consistency",
        "description": "Percentage of plays that gain positive yards/value",
        "positive_meaning": "Home team is more consistent on every play",
        "negative_meaning": "Away team is more consistent on every play"
    },
    "explosive_rate_diff": {
        "name": "Big Play Ability",
        "description": "Rate of explosive/chunk plays",
        "positive_meaning": "Home team generates more big plays",
        "negative_meaning": "Away team generates more big plays"
    },
    
    # Situational
    "redzone_epa_diff": {
        "name": "Red Zone Scoring",
        "description": "Efficiency when close to the end zone (inside 20 yards)",
        "positive_meaning": "Home team is better at finishing drives with touchdowns",
        "negative_meaning": "Away team is better at finishing drives with touchdowns"
    },
    "third_down_epa_diff": {
        "name": "Third Down Conversions",
        "description": "Ability to convert crucial third down plays",
        "positive_meaning": "Home team is better at keeping drives alive",
        "negative_meaning": "Away team is better at keeping drives alive"
    },
    
    # Turnovers
    "turnover_diff": {
        "name": "Ball Security",
        "description": "Net turnovers (interceptions + fumbles forced - turnovers committed)",
        "positive_meaning": "Home team wins the turnover battle",
        "negative_meaning": "Away team wins the turnover battle"
    },
    
    # Playoffs
    "playoff_epa_diff": {
        "name": "Playoff Experience",
        "description": "Performance in playoff games this season",
        "positive_meaning": "Home team performed better in the playoffs",
        "negative_meaning": "Away team performed better in the playoffs"
    },
    
    # Madden ratings
    "qb_ovr_diff": {
        "name": "Quarterback Rating",
        "description": "Madden video game QB rating (subjective expert assessment)",
        "positive_meaning": "Home team has a higher-rated quarterback",
        "negative_meaning": "Away team has a higher-rated quarterback"
    },
    "avg_ovr_diff": {
        "name": "Team Roster Strength",
        "description": "Average player rating across the roster",
        "positive_meaning": "Home team has a stronger overall roster",
        "negative_meaning": "Away team has a stronger overall roster"
    },
    "max_ovr_diff": {
        "name": "Star Power",
        "description": "Rating of the best player on each team",
        "positive_meaning": "Home team has a higher-rated star player",
        "negative_meaning": "Away team has a higher-rated star player"
    },
    
    # Market
    "spread_line": {
        "name": "Vegas Spread",
        "description": "Point spread set by oddsmakers (negative = home favored)",
        "positive_meaning": "Away team is favored by Vegas",
        "negative_meaning": "Home team is favored by Vegas"
    },
    "total_line": {
        "name": "Expected Total Points",
        "description": "Over/under total points (higher = expected high-scoring game)",
        "positive_meaning": "Game expected to be high-scoring",
        "negative_meaning": "Game expected to be low-scoring"
    },
    "implied_win_prob": {
        "name": "Vegas Implied Probability",
        "description": "Win probability derived from betting lines",
        "positive_meaning": "Betting markets favor home team",
        "negative_meaning": "Betting markets favor away team"
    },
}


def get_feature_explanation(feature_name: str, value: float) -> str:
    """
    Get a human-readable explanation of a feature's meaning given its value.
    
    Parameters
    ----------
    feature_name : str
        The feature column name.
    value : float
        The feature value (differential).
        
    Returns
    -------
    str
        Human-readable explanation.
    """
    if feature_name not in FEATURE_DEFINITIONS:
        # Generic fallback
        direction = "favors home team" if value > 0 else "favors away team"
        readable_name = feature_name.replace("_diff", "").replace("_", " ").title()
        return f"{readable_name}: {direction}"
    
    defn = FEATURE_DEFINITIONS[feature_name]
    if value > 0:
        return f"{defn['name']}: {defn['positive_meaning']}"
    else:
        return f"{defn['name']}: {defn['negative_meaning']}"


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def format_contributions(
    contributions: Dict[str, float], 
    top_n: int = 5,
    include_definitions: bool = True
) -> str:
    """
    Format feature contributions as a readable list with optional definitions.
    
    Parameters
    ----------
    contributions : dict
        Feature name -> contribution value (percentage points).
    top_n : int
        Number of top features to include.
    include_definitions : bool
        If True, include feature definitions for LLM context.
        
    Returns
    -------
    str
        Formatted contribution list.
    """
    # Sort by absolute contribution
    sorted_contribs = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]
    
    lines = []
    for feat, value in sorted_contribs:
        if include_definitions and feat in FEATURE_DEFINITIONS:
            defn = FEATURE_DEFINITIONS[feat]
            explanation = get_feature_explanation(feat, value)
            lines.append(f"- {defn['name']} ({value:+.1f} pp): {explanation.split(': ', 1)[1] if ': ' in explanation else explanation}")
        else:
            # Fallback to simple formatting
            readable_name = feat.replace("_diff", "").replace("_", " ").title()
            direction = "favors home" if value > 0 else "favors away"
            lines.append(f"- {readable_name}: {value:+.1f} pp ({direction})")
    
    return "\n".join(lines)


def build_explanation_prompt(ctx: Dict[str, Any]) -> str:
    """
    Build the prompt for generating natural language explanations.
    
    Parameters
    ----------
    ctx : dict
        Prediction context containing:
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - predicted_winner: Predicted winning team
        - confidence: Win probability (0-1)
        - contributions: Dict of feature contributions
        - features: Dict of raw feature values (optional)
        
    Returns
    -------
    str
        Formatted prompt for the LLM.
    """
    contributions_text = format_contributions(ctx.get("contributions", {}), include_definitions=True)
    
    # Build feature glossary for context
    glossary_items = []
    for feat in list(ctx.get("contributions", {}).keys())[:5]:
        if feat in FEATURE_DEFINITIONS:
            defn = FEATURE_DEFINITIONS[feat]
            glossary_items.append(f"- {defn['name']}: {defn['description']}")
    glossary_text = "\n".join(glossary_items) if glossary_items else ""
    
    prompt = f"""You are an NFL analyst explaining a Super Bowl prediction to a casual fan.

MATCHUP: {ctx['home_team']} vs {ctx['away_team']}
PREDICTED WINNER: {ctx['predicted_winner']} ({ctx['confidence']:.0%} confidence)

FEATURE GLOSSARY (what each metric measures):
{glossary_text}

KEY FACTORS (contribution to prediction - positive means favors home team):
{contributions_text}

Write 2-3 simple sentences explaining why {ctx['predicted_winner']} is predicted to win.
Guidelines:
- Use everyday language a casual fan would understand
- Translate metrics like "Offensive Efficiency" to "more effective offense" or "better at moving the ball"
- Mention specific team names ({ctx['home_team']} and {ctx['away_team']}) rather than "home/away"
- Focus on the most impactful 2-3 factors
- Don't mention numbers, percentages, or technical terms"""

    return prompt


def build_comparison_prompt(ctx: Dict[str, Any]) -> str:
    """
    Build a prompt comparing the two teams in detail.
    
    Parameters
    ----------
    ctx : dict
        Prediction context with feature values.
        
    Returns
    -------
    str
        Formatted comparison prompt.
    """
    features = ctx.get("features", {})
    
    # Format feature values
    feature_lines = []
    for feat, value in features.items():
        readable_name = feat.replace("_diff", "").replace("_", " ").title()
        if "diff" in feat.lower() or feat.endswith("_diff"):
            direction = f"{ctx['home_team']} advantage" if value > 0 else f"{ctx['away_team']} advantage"
            feature_lines.append(f"- {readable_name}: {value:+.1f} ({direction})")
        else:
            feature_lines.append(f"- {readable_name}: {value:.1f}")
    
    features_text = "\n".join(feature_lines)
    
    prompt = f"""You are an NFL analyst providing a detailed breakdown of a Super Bowl matchup.

MATCHUP: {ctx['home_team']} vs {ctx['away_team']}
PREDICTION: {ctx['predicted_winner']} wins ({ctx['confidence']:.0%} confidence)

TEAM COMPARISON:
{features_text}

Provide a 3-4 sentence analysis breaking down:
1. Each team's strengths and weaknesses
2. The key matchup that will decide the game
3. Why the predicted winner has the edge

Use casual, engaging language that a sports fan would enjoy reading."""

    return prompt


# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================

def check_ollama_available() -> bool:
    """
    Check if Ollama is installed and running.
    
    Returns
    -------
    bool
        True if Ollama is available.
    """
    try:
        import ollama
        # Try to list models to check if server is running
        ollama.list()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def list_ollama_models() -> list:
    """
    List available Ollama models.
    
    Returns
    -------
    list
        List of model names.
    """
    try:
        import ollama
        models = ollama.list()
        return [m.get("name", m.get("model", "unknown")) for m in models.get("models", [])]
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return []


def explain_with_ollama(
    ctx: Dict[str, Any],
    model: str = "llama3.2",
    detailed: bool = False
) -> str:
    """
    Generate natural language explanation using Ollama.
    
    Ollama is a local LLM server that provides fast inference.
    Install from: https://ollama.ai
    
    Parameters
    ----------
    ctx : dict
        Prediction context.
    model : str
        Ollama model name (e.g., "llama3.2", "mistral", "phi3").
    detailed : bool
        If True, use the detailed comparison prompt.
        
    Returns
    -------
    str
        Generated explanation.
        
    Raises
    ------
    ImportError
        If ollama package is not installed.
    ConnectionError
        If Ollama server is not running.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "Ollama package not installed. Install with: pip install ollama\n"
            "Also install Ollama app from: https://ollama.ai"
        )
    
    # Build prompt
    if detailed:
        prompt = build_comparison_prompt(ctx)
    else:
        prompt = build_explanation_prompt(ctx)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 200}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        if "connection refused" in str(e).lower():
            raise ConnectionError(
                "Ollama server not running. Start with: ollama serve\n"
                f"Then pull a model: ollama pull {model}"
            )
        raise


# =============================================================================
# HUGGING FACE TRANSFORMERS INTEGRATION
# =============================================================================

def check_transformers_available() -> bool:
    """
    Check if Transformers library is available.
    
    Returns
    -------
    bool
        True if transformers is installed.
    """
    try:
        import transformers
        return True
    except ImportError:
        return False


# Cache the pipeline to avoid reloading
_transformers_pipeline = None
_transformers_model_name = None


def explain_with_transformers(
    ctx: Dict[str, Any],
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    detailed: bool = False,
    max_new_tokens: int = 150
) -> str:
    """
    Generate natural language explanation using Hugging Face Transformers.
    
    This runs entirely in Python without external services.
    First run will download the model (~2GB for TinyLlama).
    
    Parameters
    ----------
    ctx : dict
        Prediction context.
    model : str
        HuggingFace model name. Recommended small models:
        - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params, fast)
        - "microsoft/phi-2" (2.7B params, higher quality)
        - "Qwen/Qwen2-0.5B-Instruct" (0.5B params, fastest)
    detailed : bool
        If True, use the detailed comparison prompt.
    max_new_tokens : int
        Maximum tokens to generate.
        
    Returns
    -------
    str
        Generated explanation.
        
    Raises
    ------
    ImportError
        If transformers package is not installed.
    """
    global _transformers_pipeline, _transformers_model_name
    
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Transformers package not installed. Install with:\n"
            "pip install transformers accelerate"
        )
    
    # Build prompt
    if detailed:
        prompt = build_comparison_prompt(ctx)
    else:
        prompt = build_explanation_prompt(ctx)
    
    # Initialize or reuse pipeline
    if _transformers_pipeline is None or _transformers_model_name != model:
        print(f"Loading model: {model}")
        print("This may take a minute on first run (downloading model)...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _transformers_pipeline = pipeline(
                "text-generation",
                model=model,
                device_map="auto",
                torch_dtype="auto"
            )
        _transformers_model_name = model
        print("Model loaded!")
    
    # Generate response
    result = _transformers_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=_transformers_pipeline.tokenizer.eos_token_id
    )
    
    # Extract generated text (remove the prompt)
    generated = result[0]["generated_text"]
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    
    return generated


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def explain_prediction(
    ctx: Dict[str, Any],
    backend: str = "auto",
    model: Optional[str] = None,
    detailed: bool = False
) -> str:
    """
    Generate natural language explanation for a prediction.
    
    Automatically selects the best available backend (Ollama preferred).
    
    Parameters
    ----------
    ctx : dict
        Prediction context containing:
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - predicted_winner: Predicted winning team
        - confidence: Win probability (0-1)
        - contributions: Dict of feature contributions
        - features: Dict of raw feature values (optional)
    backend : str
        "ollama", "transformers", or "auto" (tries ollama first).
    model : str, optional
        Model name override. If None, uses defaults.
    detailed : bool
        If True, generates more detailed comparison.
        
    Returns
    -------
    str
        Natural language explanation.
        
    Examples
    --------
    >>> ctx = {
    ...     "home_team": "KC",
    ...     "away_team": "SF", 
    ...     "predicted_winner": "KC",
    ...     "confidence": 0.62,
    ...     "contributions": {
    ...         "off_epa_diff": 8.5,
    ...         "qb_ovr_diff": 5.2,
    ...         "def_epa_diff": -3.1
    ...     }
    ... }
    >>> print(explain_prediction(ctx))
    "The model favors Kansas City because they have a stronger offense
    and Patrick Mahomes gives them an edge at quarterback..."
    """
    # Auto-select backend
    if backend == "auto":
        if check_ollama_available():
            backend = "ollama"
        elif check_transformers_available():
            backend = "transformers"
        else:
            return _generate_template_explanation(ctx)
    
    # Generate explanation
    try:
        if backend == "ollama":
            model = model or "llama3.2"
            return explain_with_ollama(ctx, model=model, detailed=detailed)
        elif backend == "transformers":
            model = model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            return explain_with_transformers(ctx, model=model, detailed=detailed)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return _generate_template_explanation(ctx)


def _generate_template_explanation(ctx: Dict[str, Any]) -> str:
    """
    Generate a high-quality template-based explanation when LLM is unavailable.
    
    Uses NFL-specific language and nuanced phrasing for professional-quality explanations.
    """
    winner = ctx["predicted_winner"]
    loser = ctx["away_team"] if winner == ctx["home_team"] else ctx["home_team"]
    home_team = ctx["home_team"]
    away_team = ctx["away_team"]
    confidence = ctx["confidence"]
    
    # NFL team name expansions for more natural language
    TEAM_NAMES = {
        "KC": "Kansas City", "PHI": "Philadelphia", "SF": "San Francisco",
        "TB": "Tampa Bay", "LAR": "the Rams", "CIN": "Cincinnati",
        "NE": "New England", "ATL": "Atlanta", "DEN": "Denver",
        "CAR": "Carolina", "SEA": "Seattle", "GB": "Green Bay",
        "NYG": "the Giants", "BAL": "Baltimore", "NO": "New Orleans",
        "PIT": "Pittsburgh", "ARI": "Arizona", "IND": "Indianapolis",
        "CHI": "Chicago", "DAL": "Dallas", "BUF": "Buffalo", "MIA": "Miami",
        "MIN": "Minnesota", "DET": "Detroit", "LV": "Las Vegas",
        "LAC": "the Chargers", "NYJ": "the Jets", "CLE": "Cleveland", 
        "HOU": "Houston", "TEN": "Tennessee", "JAX": "Jacksonville", 
        "WAS": "Washington"
    }
    
    winner_name = TEAM_NAMES.get(winner, winner)
    loser_name = TEAM_NAMES.get(loser, loser)
    
    # Map each feature directly to NFL-appropriate phrases
    FEATURE_TO_PHRASE = {
        "off_epa_diff": ("moved the ball more efficiently on offense", "offensive efficiency"),
        "def_epa_diff": ("limited opposing offenses more effectively", "defensive performance"),
        "pass_epa_diff": ("dominated through the air with superior passing", "passing attack"),
        "rush_epa_diff": ("established the ground game with a stronger rushing attack", "running game"),
        "success_rate_diff": ("executed more consistently on every snap", "play-by-play consistency"),
        "explosive_rate_diff": ("created more explosive chunk plays", "big-play ability"),
        "redzone_epa_diff": ("converted red zone trips into touchdowns more efficiently", "red zone scoring"),
        "third_down_epa_diff": ("sustained drives by converting on third down", "third down execution"),
        "turnover_diff": ("won the turnover battle with better ball security", "turnover margin"),
        "playoff_epa_diff": ("elevated their play during the postseason", "playoff performance"),
        "qb_ovr_diff": ("has the edge at quarterback, the game's most important position", "quarterback advantage"),
        "avg_ovr_diff": ("fields a deeper and more talented roster overall", "roster depth"),
        "max_ovr_diff": ("boasts the best individual playmaker on the field", "star power"),
        "spread_line": ("has the confidence of oddsmakers", "Vegas backing"),
        "total_line": ("projects for a high-scoring affair", "game pace"),
        "implied_win_prob": ("is favored by betting markets", "market sentiment"),
    }
    
    # Find top contributors
    contributions = ctx.get("contributions", {})
    sorted_contribs = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:4]
    
    # Collect advantages that favor the winner
    winner_advantages = []
    
    for feat, value in sorted_contribs:
        if feat not in FEATURE_TO_PHRASE:
            continue
            
        phrase, short_name = FEATURE_TO_PHRASE[feat]
        
        # Determine which team this factor favors
        # For defensive metrics, lower (negative) is better for home team
        if "def" in feat:
            favors_home = value < 0
        else:
            favors_home = value > 0
        
        # Check if this factor favors the predicted winner
        if (favors_home and winner == home_team) or (not favors_home and winner == away_team):
            winner_advantages.append((phrase, short_name, abs(value)))
    
    # Build the explanation
    if len(winner_advantages) >= 2:
        primary = winner_advantages[0][0]
        secondary = winner_advantages[1][0]
        
        if confidence >= 0.65:
            return (
                f"{winner_name} is projected to defeat {loser_name} ({confidence:.0%} confidence). "
                f"The model heavily favors {winner_name} because they {primary}. "
                f"Additionally, they {secondary}. "
                f"These combined advantages make {winner_name} a clear favorite heading into the Super Bowl."
            )
        elif confidence >= 0.55:
            return (
                f"{winner_name} holds a meaningful edge over {loser_name} ({confidence:.0%} confidence). "
                f"{winner_name} {primary}, giving them an advantage in this matchup. "
                f"They also {secondary}, which tips the scales in their favor. "
                f"Expect a competitive game, but {winner_name} has the edge."
            )
        else:
            return (
                f"In what projects to be a closely contested Super Bowl, {winner_name} has a slight edge over {loser_name} ({confidence:.0%} confidence). "
                f"While both teams are evenly matched, {winner_name} {primary}. "
                f"Small margins in {winner_advantages[1][1]} also favor {winner_name}. "
                f"This one could go either way."
            )
    elif len(winner_advantages) == 1:
        primary = winner_advantages[0][0]
        return (
            f"{winner_name} is given a narrow edge over {loser_name} ({confidence:.0%} confidence). "
            f"The key differentiator is that {winner_name} {primary}. "
            f"With both teams otherwise evenly matched, this factor could prove decisive."
        )
    else:
        return (
            f"The model projects {winner_name} to edge {loser_name} in a toss-up matchup ({confidence:.0%} confidence). "
            f"Both teams are remarkably evenly matched across key metrics. "
            f"The margin is razor-thin, and either team could hoist the Lombardi Trophy."
        )


def get_available_backends() -> Dict[str, bool]:
    """
    Check which LLM backends are available.
    
    Returns
    -------
    dict
        Backend name -> availability status.
    """
    return {
        "ollama": check_ollama_available(),
        "transformers": check_transformers_available(),
        "template": True  # Always available
    }


# =============================================================================
# BATCH EXPLANATION
# =============================================================================

def explain_all_predictions(
    predictions: list,
    backend: str = "auto",
    model: Optional[str] = None,
    show_progress: bool = True
) -> list:
    """
    Generate explanations for multiple predictions.
    
    Parameters
    ----------
    predictions : list
        List of prediction context dictionaries.
    backend : str
        "ollama", "transformers", or "auto".
    model : str, optional
        Model name override.
    show_progress : bool
        If True, print progress.
        
    Returns
    -------
    list
        List of explanation strings.
    """
    explanations = []
    total = len(predictions)
    
    for i, ctx in enumerate(predictions):
        if show_progress:
            print(f"Generating explanation {i+1}/{total}...", end="\r")
        
        explanation = explain_prediction(ctx, backend=backend, model=model)
        explanations.append(explanation)
    
    if show_progress:
        print(f"Generated {total} explanations.          ")
    
    return explanations

