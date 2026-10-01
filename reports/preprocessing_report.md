# Preprocessing Report

## 1. Data Splitting
- **Grouping Strategy**: We used `GroupShuffleSplit` on the `player_id` column to ensure that no player has records appearing in more than one split, preventing data leakage across time.
- **Split Ratios**: The dataset is split into Training (70%), Validation (15%), and Test (15%) sets.
- **Target Separation**: The target variable `overall` is strictly separated from the input features and is not scaled or transformed.

## 2. Feature Selection
- Features were selected based on the approved guidelines in `modeling_strategy.md`.
- **Outfield Features**: Includes physical, technical, mental, and defensive attributes (e.g., `age`, `height_cm`, `weight_kg`, `preferred_foot`, `skill_dribbling`, `movement_acceleration`, etc.).
- **Goalkeeper Features**: Includes physical and goalkeeping-specific attributes (e.g., `age`, `height_cm`, `weight_kg`, `goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_reflexes`, etc.).
- Identifiers, metadata, and high-missingness columns were explicitly excluded.

## 3. Missing-Value Handling
- **Target Variable**: Any records with a missing `overall` rating are immediately dropped before splitting. The number of dropped records is reported during processing.
- **Imputation**:
  - **Numerical Features**: Imputed using `SimpleImputer` with the `median` strategy.
  - **Categorical Features**: Imputed using `SimpleImputer` with the `most_frequent` strategy.
- **Leakage Prevention**: All imputers are fitted strictly on the Training split. The fitted imputers are then used to transform the Validation and Test splits.

## 4. Feature Scaling and Encoding
- **Encoding**: The `preferred_foot` categorical variable is mapped to a binary format (Right: 1, Left: 0).
- **Scaling**: All selected features (which are purely numerical after encoding) are standardized using `StandardScaler`.
- **Leakage Prevention**: The `StandardScaler` is fitted only on the Training split and applied to the Validation and Test splits. The final data type is strictly float32 to ensure compatibility with PyTorch tensors.

## 5. Output and Testing
- The output from the preprocessing pipeline comprises dictionaries containing the processed `train`, `val`, and `test` data structures as Pandas DataFrames/Series.
- Extensive unit testing has been implemented and successfully executed, verifying reproducibility, prevention of data leakage, and separation of outfield/goalkeeper models.
