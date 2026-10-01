import pandas as pd
import json

def analyze_data():
    df = pd.read_csv('data/raw/male_players (legacy).csv', low_memory=False)

    # 1. Target Analysis
    overall = df['overall']
    target_analysis = {
        'dtype': str(overall.dtype),
        'missing': int(overall.isna().sum()),
        'min': float(overall.min()),
        'max': float(overall.max()),
        'mean': float(overall.mean()),
        'median': float(overall.median()),
        'std': float(overall.std()),
        'distribution': {str(k): v for k, v in overall.value_counts(bins=10).sort_index().to_dict().items()}
    }

    # 2. Missing Value Analysis
    missing_analysis = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = float(missing_count / len(df) * 100)
        missing_analysis.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })

    # 5. Dataset Integrity
    integrity = {
        'duplicate_rows': int(df.duplicated().sum()),
        'unique_player_ids': int(df['player_id'].nunique()) if 'player_id' in df.columns else None,
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=['number']).columns),
        'categorical_cols': len(df.select_dtypes(exclude=['number']).columns),
    }

    results = {
        'target_analysis': target_analysis,
        'missing_analysis': missing_analysis,
        'integrity': integrity,
        'columns': list(df.columns)
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)

if __name__ == '__main__':
    analyze_data()
