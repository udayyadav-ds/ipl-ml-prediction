import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Standardize team names
    team_mapping = {'Kings XI Punjab': 'Punjab Kings', 'Delhi Daredevils': 'Delhi Capitals', 
                    'Rising Pune Supergiants': 'Rising Pune Supergiant', 'Deccan Chargers': 'Sunrisers Hyderabad'}
    df['team1'] = df['team1'].replace(team_mapping)
    df['team2'] = df['team2'].replace(team_mapping)
    df['winner'] = df['winner'].replace(team_mapping)
    df = df.dropna(subset=['winner'])

    # --- FEATURE ENGINEERING ---
    # 1. Team Win Ratio (Historical)
    all_teams = sorted(list(set(df['team1'].unique()) | set(df['team2'].unique())))
    win_ratios = {}
    for team in all_teams:
        total_matches = len(df[(df['team1'] == team) | (df['team2'] == team)])
        total_wins = len(df[df['winner'] == team])
        win_ratios[team] = total_wins / total_matches if total_matches > 0 else 0
    
    df['team1_win_ratio'] = df['team1'].map(win_ratios)
    df['team2_win_ratio'] = df['team2'].map(win_ratios)

    # 2. Venue Average Win Margin (Indicates if it's a high-scoring ground)
    venue_stats = df.groupby('venue')['win_by_runs'].mean().to_dict()
    df['venue_avg_runs'] = df['venue'].map(venue_stats)

    # Encoding
    le = LabelEncoder()
    le.fit(all_teams)
    df['team1_encoded'] = le.transform(df['team1'])
    df['team2_encoded'] = le.transform(df['team2'])
    df['winner_encoded'] = le.transform(df['winner'])
    
    le_venue = LabelEncoder()
    df['venue_encoded'] = le_venue.fit_transform(df['venue'])
    
    df.to_csv('processed_matches.csv', index=False)
    print("Advanced Preprocessing complete!")
    return df, le, le_venue
