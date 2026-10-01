# Modeling Strategy (V1 Finalized)

This document outlines the finalized modeling strategy for the FIFA Overall Rating Prediction System based on approved project decisions.

## 1. Target Definition & Model Architecture
- **Target Variable**: Predict the **existing** `overall` rating based on current attributes.
- **Model Choice**: A PyTorch feedforward neural network designed for regression.
- **Goalkeeper Strategy**: **Separate models** will be built and trained for Goalkeepers (GKs) and Outfield players, as they possess mutually exclusive critical skill sets.

## 2. Feature Selection

For V1, the input scope strictly prioritizes core player attributes. Correlation with `overall` does not automatically classify a feature as a leak. Features are grouped into the following categories:

### Legitimate Predictive Features (Included in V1)
These represent a player's underlying physical, technical, mental, and defensive abilities.
- **Outfield Players**: `age`, `height_cm`, `weight_kg`, `preferred_foot`, `weak_foot`, `skill_moves`, `attacking_crossing`, `attacking_finishing`, `attacking_heading_accuracy`, `attacking_short_passing`, `attacking_volleys`, `skill_dribbling`, `skill_curve`, `skill_fk_accuracy`, `skill_long_passing`, `skill_ball_control`, `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_reactions`, `movement_balance`, `power_shot_power`, `power_jumping`, `power_stamina`, `power_strength`, `power_long_shots`, `mentality_aggression`, `mentality_interceptions`, `mentality_positioning`, `mentality_vision`, `mentality_penalties`, `mentality_composure`, `defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle`.
- **Goalkeepers**: `age`, `height_cm`, `weight_kg`, `preferred_foot`, `weak_foot`, `goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_kicking`, `goalkeeping_positioning`, `goalkeeping_reflexes`, `movement_reactions`, `mentality_composure`.

### Features Excluded (Outside Project Scope)
Metadata, IDs, categorical strings, and sparsely populated columns that fall outside the intended scope of an attribute-based prediction.
- **Identifiers**: `player_id`, `player_url`, `short_name`, `long_name`, `dob`, `real_face`.
- **Club/National Data**: `league_id`, `league_name`, `league_level`, `club_team_id`, `club_name`, `club_position`, `club_jersey_number`, `nationality_id`, `nationality_name`, `nation_team_id`, `nation_position`, `nation_jersey_number`.
- **High Missingness (>50%)**: `player_tags`, `player_traits`, `club_loaned_from`.

### Features Derived from or Closely Related to Target (Optional/Excluded for V1)
These features are typically assigned post-facto or derived algorithmically from the overall rating in the real world. They are excluded for V1 unless explicit justification for their inclusion is provided in future iterations.
- `potential`: Heavily reflects a player's current overall baseline.
- `international_reputation`: Often acts as a subjective post-calculation boost to the overall rating.
- `value_eur`, `wage_eur`, `release_clause_eur`: Explicitly calculated using the player's age, potential, and `overall` rating.

### Direct Target Leakage (Excluded)
- **Aggregate Stats** (`pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`): These are strict mathematical formulas derived directly from the underlying attributes. Including them alongside the underlying stats causes extreme multicollinearity. Because V1 focuses on granular attributes, these aggregates are excluded.

## 3. Data Splitting & Longitudinal Records

- **Longitudinal Handling**: All historical records for the players will be kept to maximize the dataset size.
- **Splitting Strategy**: A grouped data split approach (`GroupShuffleSplit`) will be utilized on `player_id`. This guarantees that records belonging to the same player cannot appear across the training, validation, and test sets.
- **Split Ratios**: 70% Training / 15% Validation / 15% Test.
- **Reproducibility & Verification**: All splits will use a fixed random seed. The data loading pipeline must verify and report the number of unique players present in each split to mathematically prove no data leakage occurred between sets.

## 4. Evaluation Metrics

- **Primary Metric**: **Mean Absolute Error (MAE)**. This provides an interpretable measure (e.g., "the model is off by X overall points on average").
- **Secondary Metrics**:
  - **Root Mean Squared Error (RMSE)**: To penalize larger prediction errors.
  - **R² Score**: To measure variance explained and ensure the neural network outperforms simple baseline models.

## 5. Unresolved Decisions
- None. The V1 modeling strategy is fully finalized and ready for the implementation of preprocessing and data loaders.
