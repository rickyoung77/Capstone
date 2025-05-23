import pandas as pd
import numpy as np
from geopy.distance import geodesic
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2


# Load the dataset
file_path = 'regular_season_totals_2010_2024.csv'  # Replace with your file path
nba_data = pd.read_csv(file_path)

# Arena coordinates
arena_coords = {
    'ATL': (33.748995, -84.387982), 'BOS': (42.360083, -71.05888),
    'BKN': (40.678178, -73.944158), 'CHA': (35.227087, -80.843127),
    'CHI': (41.878114, -87.629798), 'CLE': (41.49932, -81.694361),
    'DAL': (32.776664, -96.796988), 'DEN': (39.739236, -104.990251),
    'DET': (42.331427, -83.045754), 'GSW': (37.77493, -122.419416),
    'HOU': (29.760427, -95.369803), 'IND': (39.768403, -86.158068),
    'LAC': (34.052235, -118.243683), 'LAL': (34.052235, -118.243683),
    'MEM': (35.149534, -90.04898), 'MIA': (25.76168, -80.19179),
    'MIL': (43.038902, -87.906474), 'MIN': (44.977753, -93.265011),
    'NOP': (29.951065, -90.071533), 'NYK': (40.712776, -74.005974),
    'OKC': (35.46756, -97.516428), 'ORL': (28.538336, -81.379234),
    'PHI': (39.952583, -75.165222), 'PHX': (33.448377, -112.074037),
    'POR': (45.515232, -122.678385), 'SAC': (38.581572, -121.4944),
    'SAS': (29.424122, -98.493628), 'TOR': (43.653225, -79.383186),
    'UTA': (40.760779, -111.891047), 'WAS': (38.907192, -77.036871),
    'NOH': (29.951065, -90.071533), 'NJN': (40.732025, -74.174626),
    'SEA': (47.606209, -122.332069), 'VAN': (49.282729, -123.120738)
}

# Helper functions
def get_coordinates(row, team_column):
    team = row[team_column]
    return arena_coords.get(team, (None, None))  # Return (None, None) if not found

def calculate_distance(row):
    home_coords = row['Home_Coords']
    away_coords = row['Away_Coords']
    if None not in home_coords and None not in away_coords:
        # Use haversine function
        return haversine(home_coords[0], home_coords[1], away_coords[0], away_coords[1])
    return None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# Process home/away teams and calculate distances
nba_data['Home/Away'] = nba_data['MATCHUP'].apply(lambda x: 'Away' if '@' in x else 'Home')
nba_data['Home_Team'] = nba_data['MATCHUP'].apply(lambda x: x.split('vs. ')[0] if 'vs.' in x else x.split(' @ ')[0])
nba_data['Away_Team'] = nba_data['MATCHUP'].apply(lambda x: x.split('vs. ')[-1] if 'vs.' in x else x.split(' @ ')[-1])
nba_data['Home_Coords'] = nba_data.apply(lambda row: get_coordinates(row, 'Home_Team'), axis=1)
nba_data['Away_Coords'] = nba_data.apply(lambda row: get_coordinates(row, 'Away_Team'), axis=1)
nba_data['Travel_Distance'] = nba_data.apply(calculate_distance, axis=1)

# Clean data: remove rows with missing travel distances
nba_data_cleaned = nba_data.dropna(subset=['Travel_Distance'])





# Aggregate team stats
team_stats = nba_data_cleaned.groupby('TEAM_ABBREVIATION').agg(
    Total_Travel_Distance=('Travel_Distance', 'sum'),
    Total_Games=('WL', 'count'),
    Wins=('WL', lambda x: (x == 'W').sum())
).reset_index()
team_stats['Winning_Percentage'] = team_stats['Wins'] / team_stats['Total_Games']

# Explicitly convert columns to NumPy arrays
try:
    X = np.array(team_stats['Total_Travel_Distance']).reshape(-1, 1)
    y = np.array(team_stats['Winning_Percentage'])

    # Debug the data to confirm it's numerical and correctly shaped
    print("X shape:", X.shape, "y shape:", y.shape)
    print("First 5 rows of X:", X[:5])
    print("First 5 rows of y:", y[:5])

    # Fit the Linear Regression model
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Actual Data', alpha=0.7)
    plt.plot(X, y_pred, color='red', label=f'Regression Line (Slope={reg.coef_[0]:.5f})')
    plt.title('Effect of Total Travel Distance on Winning Percentage')
    plt.xlabel('Total Travel Distance (miles)')
    plt.ylabel('Winning Percentage')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print regression results
    print(f"Regression Slope: {reg.coef_[0]:.5f}")
    print(f"Intercept: {reg.intercept_:.5f}")
    print(f"R-squared: {reg.score(X, y):.5f}")

except Exception as e:
    print("An error occurred:", e)

