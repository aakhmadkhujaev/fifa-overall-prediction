import base64
import io
from typing import Tuple, Dict, Any, Mapping

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
PREPROCESSOR_SCHEMA_VERSION = 1

class Preprocessor:
    def __init__(self, is_goalkeeper: bool = False):
        self.is_goalkeeper = is_goalkeeper
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = ['preferred_foot']
        self.feature_names = []
        self.target_col = 'overall'
        self.group_col = 'player_id'

    def _split_data(self, df: pd.DataFrame, group_col: str, test_size: float, val_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe into train, val, and test sets using GroupShuffleSplit.
        """
        # First split into train_val and test
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))

        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Then split train_val into train and val
        val_ratio = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))

        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()

        return train_df, val_df, test_df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode preferred_foot to binary. 1 for Right, 0 for Left.
        """
        if 'preferred_foot' in df.columns:
            allowed_values = {'Right', 'Left'}
            unexpected_values = [
                value
                for value in df['preferred_foot'].dropna().unique()
                if value not in allowed_values
            ]
            if unexpected_values:
                raise ValueError(
                    "Unexpected preferred_foot values: "
                    f"{unexpected_values}. Expected only 'Right' or 'Left'."
                )
            df['preferred_foot'] = df['preferred_foot'].map({'Right': 1, 'Left': 0})
        return df

    def process(self, df: pd.DataFrame, target_col: str = 'overall', group_col: str = 'player_id') -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, Any]]:
        """
        Process the dataframe: split, impute, encode, scale.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty; at least one row is required.")

        # 1. Drop rows with missing target
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        rows_dropped = initial_rows - len(df)

        if df.empty:
            raise ValueError("No rows remain after dropping missing target values.")

        # 2. Split data
        train_df, val_df, test_df = self._split_data(df, group_col, test_size=0.15, val_size=0.15)

        split_frames = {'train': train_df, 'val': val_df, 'test': test_df}
        empty_splits = [name for name, split in split_frames.items() if split.empty]
        if empty_splits:
            raise ValueError(f"Preprocessing produced empty split(s): {empty_splits}.")

        # Determine numeric and categorical features
        features = [c for c in df.columns if c not in [target_col, group_col]]
        self.feature_names = list(features)
        self.target_col = target_col
        self.group_col = group_col
        self.numeric_features = [f for f in features if f not in self.categorical_features]

        # Separate X and y
        X_train, y_train = train_df[features].copy(), train_df[target_col].copy()
        X_val, y_val = val_df[features].copy(), val_df[target_col].copy()
        X_test, y_test = test_df[features].copy(), test_df[target_col].copy()

        # Keep groups for reporting
        groups_train = train_df[group_col]
        groups_val = val_df[group_col]
        groups_test = test_df[group_col]

        # 3. Imputation
        if self.numeric_features:
            X_train[self.numeric_features] = np.asarray(
                self.num_imputer.fit_transform(X_train[self.numeric_features])
            )
            X_val[self.numeric_features] = np.asarray(
                self.num_imputer.transform(X_val[self.numeric_features])
            )
            X_test[self.numeric_features] = np.asarray(
                self.num_imputer.transform(X_test[self.numeric_features])
            )

        if self.categorical_features and all(c in X_train.columns for c in self.categorical_features):
            X_train[self.categorical_features] = np.asarray(
                self.cat_imputer.fit_transform(X_train[self.categorical_features])
            )
            X_val[self.categorical_features] = np.asarray(
                self.cat_imputer.transform(X_val[self.categorical_features])
            )
            X_test[self.categorical_features] = np.asarray(
                self.cat_imputer.transform(X_test[self.categorical_features])
            )

        # 4. Encoding
        X_train = self._encode_categorical(X_train)
        X_val = self._encode_categorical(X_val)
        X_test = self._encode_categorical(X_test)

        # update numerical features list as categorical is now numeric
        all_numeric = self.numeric_features + self.categorical_features

        # 5. Scaling
        X_train[all_numeric] = self.scaler.fit_transform(X_train[all_numeric])
        X_val[all_numeric] = self.scaler.transform(X_val[all_numeric])
        X_test[all_numeric] = self.scaler.transform(X_test[all_numeric])

        # Ensure all columns are numeric floats
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

        for split_name, features_frame, target_series in [
            ('train', X_train, y_train),
            ('val', X_val, y_val),
            ('test', X_test, y_test),
        ]:
            if not np.isfinite(features_frame.to_numpy()).all():
                raise ValueError(
                    f"Preprocessed {split_name} features contain NaN or infinite values."
                )
            if not np.isfinite(target_series.to_numpy()).all():
                raise ValueError(
                    f"Preprocessed {split_name} targets contain NaN or infinite values."
                )

        X_dict = {'train': X_train, 'val': X_val, 'test': X_test}
        y_dict = {'train': y_train, 'val': y_val, 'test': y_test}

        stats = {
            'rows_dropped_missing_target': rows_dropped,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_players': groups_train.nunique(),
            'val_players': groups_val.nunique(),
            'test_players': groups_test.nunique(),
            'features': list(X_train.columns)
        }

        return X_dict, y_dict, stats

    def get_state(self) -> Dict[str, Any]:
        """Return fitted preprocessing state using only torch-serializable values."""
        if not self.feature_names:
            raise ValueError("Preprocessor must be fitted with process() before state export.")

        return {
            'schema_version': PREPROCESSOR_SCHEMA_VERSION,
            'is_goalkeeper': self.is_goalkeeper,
            'feature_names': list(self.feature_names),
            'numeric_features': list(self.numeric_features),
            'categorical_features': list(self.categorical_features),
            'target_col': self.target_col,
            'group_col': self.group_col,
            'transformers': self._serialize_transformers(),
        }

    def _serialize_transformers(self) -> str:
        buffer = io.BytesIO()
        joblib.dump(
            {
                'numeric_imputer': self.num_imputer,
                'categorical_imputer': self.cat_imputer,
                'scaler': self.scaler,
            },
            buffer,
        )
        return base64.b64encode(buffer.getvalue()).decode('ascii')

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> 'Preprocessor':
        """Restore a preprocessor from state saved in a model checkpoint."""
        if state.get('schema_version') != PREPROCESSOR_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preprocessing schema version: {state.get('schema_version')}"
            )

        preprocessor = cls(is_goalkeeper=bool(state['is_goalkeeper']))
        preprocessor.feature_names = list(state['feature_names'])
        preprocessor.numeric_features = list(state['numeric_features'])
        preprocessor.categorical_features = list(state['categorical_features'])
        preprocessor.target_col = str(state['target_col'])
        preprocessor.group_col = str(state['group_col'])

        transformer_bytes = base64.b64decode(state['transformers'].encode('ascii'))
        transformers = joblib.load(io.BytesIO(transformer_bytes))
        preprocessor.num_imputer = transformers['numeric_imputer']
        preprocessor.cat_imputer = transformers['categorical_imputer']
        preprocessor.scaler = transformers['scaler']
        return preprocessor

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform compatible raw features using the fitted checkpoint state."""
        if not self.feature_names:
            raise ValueError("Preprocessor state is not fitted.")

        missing_features = [feature for feature in self.feature_names if feature not in df.columns]
        extra_features = [feature for feature in df.columns if feature not in self.feature_names]
        if missing_features or extra_features:
            raise ValueError(
                "Incompatible feature schema. "
                f"Missing: {missing_features}; extra: {extra_features}."
            )

        transformed = df.loc[:, self.feature_names].copy()
        if self.numeric_features:
            transformed[self.numeric_features] = np.asarray(
                self.num_imputer.transform(transformed[self.numeric_features])
            )
        if self.categorical_features:
            transformed[self.categorical_features] = np.asarray(
                self.cat_imputer.transform(transformed[self.categorical_features])
            )
        transformed = self._encode_categorical(transformed)
        all_numeric = self.numeric_features + self.categorical_features
        transformed[all_numeric] = self.scaler.transform(transformed[all_numeric])
        transformed = transformed.astype(np.float32)
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError("Transformed features contain NaN or infinite values.")
        return transformed
