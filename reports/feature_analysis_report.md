# Feature Analysis and Data Quality Audit

## 1. Dataset Overview

- **Dataset Path**: `data/raw/male_players (legacy).csv`
- **Total Rows**: 161,583
- **Total Columns**: 110 (63 numerical, 47 categorical)
- **Unique Players**: 49,699 (meaning there are multiple historical records/versions per player).
- **Duplicate Rows**: 0

The dataset contains multiple years of data for the same players (tracked by `fifa_version` and `player_id`).

## 2. Target Distribution (`overall`)

- **Data Type**: `int64`
- **Missing Values**: 0 (0.0%)
- **Minimum Value**: 40
- **Maximum Value**: 94
- **Mean**: 65.70
- **Median**: 66.0
- **Standard Deviation**: 7.04

The `overall` rating approximates a normal distribution centered around 66. Most players fall within the 60 to 72 range, with a sharp drop-off for elite ratings (85+).

## 3. Missing Value Summary

### No Missing Values (0%)
- Identifiers and metadata: `player_id`, `player_url`, `fifa_version`, `short_name`, `long_name`.
- Target: `overall`, `potential`.
- Bio: `age`, `height_cm`, `weight_kg`, `dob`.
- Technical/Physical attributes: `attacking_crossing`, `movement_acceleration`, `power_stamina`, `skill_dribbling`, etc.
- Goalkeeping basics: `goalkeeping_diving`, `goalkeeping_handling`, etc.

### Less than 10% Missing
- **~1.1% Missing**: `value_eur`, `wage_eur`, `league_id`, `club_name`, `club_position`, `club_jersey_number`.
- **~1.7% Missing**: `league_level`.
- **~7.0% Missing**: `club_joined_date`.

### 10–50% Missing
- **~11.1% Missing**: Aggregate stats (`pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`). These are typically missing for Goalkeepers.
- **~20.3% Missing**: `mentality_composure`.
- **~35.9% Missing**: `release_clause_eur`.

### More than 50% Missing
- **~54.7% Missing**: `player_traits`.
- **~88.9% Missing**: `goalkeeping_speed` (only present for Goalkeepers).
- **~92.2% Missing**: `player_tags`.
- **~94.1% Missing**: `club_loaned_from`.
- **~94.2% Missing**: National team data (`nation_team_id`, `nation_position`, `nation_jersey_number`).

*Note: Columns with >50% missing values generally require dropping or specific imputation strategies (e.g., binary flags).*

## 4. Candidate Feature Groups

### Physical Attributes
`height_cm`, `weight_kg`, `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_balance`, `power_jumping`, `power_stamina`, `power_strength`.

### Technical Attributes
`attacking_crossing`, `attacking_finishing`, `attacking_heading_accuracy`, `attacking_short_passing`, `attacking_volleys`, `skill_dribbling`, `skill_curve`, `skill_fk_accuracy`, `skill_long_passing`, `skill_ball_control`, `power_shot_power`, `power_long_shots`.

### Defensive Attributes
`defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle`, `mentality_interceptions`, `mentality_aggression`.

### Goalkeeping Attributes
`goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_kicking`, `goalkeeping_positioning`, `goalkeeping_reflexes` (and possibly `goalkeeping_speed` if heavily imputed).

### Player Information
`age`, `preferred_foot`, `weak_foot`, `skill_moves`, `international_reputation`, `work_rate`, `body_type`.

### Contract and Market Information (Use with Caution)
`value_eur`, `wage_eur`, `release_clause_eur`, `club_name`, `league_level`.

## 5. Potential Target Leakage Risks

Several features could directly encode or reveal the `overall` rating:

1. **`potential`**: Highly correlated with `overall`. A player's current overall heavily influences their potential baseline.
2. **`value_eur`, `wage_eur`, `release_clause_eur`**: Financial valuations in FIFA are algorithmically derived from the player's `overall` rating, age, and potential. Using them to predict `overall` is a classic case of target leakage.
3. **`international_reputation`**: Often acts as a post-facto modifier to the overall rating in the FIFA algorithm.
4. **`movement_reactions` & `mentality_composure`**: In FIFA's internal logic, these attributes are heavily correlated with the `overall` rating. Sometimes they are artificially inflated/deflated to reach a desired target OVR.
5. **Aggregate Stats (`pace`, `shooting`, etc.)**: These are direct summaries of underlying attributes. If the model uses underlying attributes, keeping these might be redundant, but they are not strictly leakage. However, the calculation of the final OVR is a weighted sum of specific underlying attributes depending on the player's position.

## 6. Recommended Decisions (Requires Project Owner Approval)

1. **Data Leakage Resolution**: Request approval to **drop** `value_eur`, `wage_eur`, `release_clause_eur`, and `potential` from the input feature set to prevent target leakage.
2. **Goalkeeper Handling**: Because aggregate stats (`pace`, `shooting`, etc.) are missing for ~11% of players (Goalkeepers), we need approval on whether to:
   - Impute them with zeros.
   - Train a separate model for goalkeepers versus outfield players.
3. **Longitudinal Data**: Since players have multiple records across `fifa_version`s, request approval on whether to treat each row independently or if a specific train/test split strategy based on year/player_id is required to avoid data leakage across time.
4. **Dropping High-Missingness Columns**: Request approval to drop columns with >50% missing values (e.g., national team data, `club_loaned_from`) unless we extract boolean flags (e.g., `is_on_loan`, `has_national_team`).
