import pandas as pd
from typing import List, Tuple, Dict, Any

OUTFIELD_FEATURES = [
    'age', 'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot', 'skill_moves',
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
    'mentality_penalties', 'mentality_composure', 'defending_marking_awareness',
    'defending_standing_tackle', 'defending_sliding_tackle'
]

GOALKEEPER_FEATURES = [
    'age', 'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot',
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
    'goalkeeping_positioning', 'goalkeeping_reflexes', 'movement_reactions',
    'mentality_composure'
]

TARGET_COL = 'overall'
GROUP_COL = 'player_id'

def get_outfield_features() -> List[str]:
    """Return the list of features for outfield players."""
    return OUTFIELD_FEATURES.copy()

def get_goalkeeper_features() -> List[str]:
    """Return the list of features for goalkeepers."""
    return GOALKEEPER_FEATURES.copy()

def select_features(df: pd.DataFrame, is_goalkeeper: bool = False) -> pd.DataFrame:
    """
    Select the relevant features and target column from the dataframe.
    Also retains the player_id for grouping during splits.

    Args:
        df: Input dataframe.
        is_goalkeeper: If True, selects goalkeeper features, otherwise outfield features.

    Returns:
        pd.DataFrame: Dataframe with selected columns.
    """
    features = GOALKEEPER_FEATURES if is_goalkeeper else OUTFIELD_FEATURES

    # Check if target and group column are in the dataframe
    cols_to_keep = features + [TARGET_COL, GROUP_COL]

    # We might have club_position to separate GK and outfield in the full dataset,
    # but the pipeline might pass pre-filtered df.
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in the dataset: {missing_cols}")

    return df[cols_to_keep].copy()

def split_gk_and_outfield(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into goalkeepers and outfield players based on 'club_position' or 'goalkeeping' stats availability.
    Assuming players with 'club_position' == 'GK' are goalkeepers, but we can also use 'goalkeeping_speed' missingness
    or just check 'club_position'. If club_position is not available, check for 'team_position' == 'GK' or just fallback.
    Actually, let's use 'club_position' == 'GK' or 'nation_position' == 'GK', or if 'player_traits' mentions GK.
    A safer way if 'club_position' is missing is to check 'goalkeeping_diving' > some threshold compared to outfield stats,
    but for FIFA datasets, usually 'club_position' == 'SUB'/'RES' makes it hard.
    Wait, in FIFA datasets, 'player_positions' often has 'GK' as the first position.
    """
    if 'player_positions' in df.columns:
        # players can have multiple positions like 'CB, CDM'. Goalkeepers are usually just 'GK'
        is_gk = df['player_positions'].str.startswith('GK', na=False)
    elif 'club_position' in df.columns:
        is_gk = df['club_position'] == 'GK'
    else:
        raise ValueError("Cannot determine player positions without 'player_positions' or 'club_position' column.")

    gk_df = df[is_gk].copy()
    outfield_df = df[~is_gk].copy()

    return gk_df, outfield_df
