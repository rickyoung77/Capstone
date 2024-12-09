from nba_api.stats.endpoints import leaguegamelog
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Team region mapping: North (1) or South (0)
team_regions = {
    'ATL': 0, 'BOS': 1, 'BKN': 1, 'CHA': 0, 'CHI': 1,
    'CLE': 1, 'DAL': 0, 'DEN': 1, 'DET': 1, 'GSW': 1,
    'HOU': 0, 'IND': 1, 'LAC': 0, 'LAL': 0, 'MEM': 0,
    'MIA': 0, 'MIL': 1, 'MIN': 1, 'NOP': 0, 'NYK': 1,
    'OKC': 0, 'ORL': 0, 'PHI': 1, 'PHX': 0, 'POR': 1,
    'SAC': 1, 'SAS': 0, 'TOR': 1, 'UTA': 1, 'WAS': 1
}

# Function to fetch game data for a specific season
def fetch_game_data(season):
    gamelog = leaguegamelog.LeagueGameLog(season=season)
    games = gamelog.get_data_frames()[0]
    games['Season'] = season  # Add season column
    return games[['TEAM_ABBREVIATION', 'GAME_DATE', 'Season', 'MATCHUP', 'FGA', 'FGM', 'FG3M', 'TOV', 'OREB', 'DREB', 'FTA', 'PTS']]

# Function to fetch data for multiple seasons
def fetch_multiple_seasons(seasons):
    all_data = []
    for season in seasons:
        print(f"Fetching data for season {season}...")
        season_data = fetch_game_data(season)
        all_data.append(season_data)
    return pd.concat(all_data, ignore_index=True)


'''
def filter_away_games_outside_region(data):
    # Add a binary variable for Away games using .loc
    data.loc[:, 'Away'] = data['MATCHUP'].apply(lambda x: 1 if '@' in x else 0)

    # Filter for Away games
    away_games = data[data['Away'] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Determine if the game is outside the team's region
    away_games.loc[:, 'Outside_Region'] = away_games.apply(
        lambda x: 1 if (x['Region'] != team_regions[x['TEAM_ABBREVIATION']]) else 0,
        axis=1
    )

    # Filter for games outside the team's region
    return away_games[away_games['Outside_Region'] == 1]
'''


# Function to calculate SoP and include Region
def prepare_data(data):
    # Calculate Field Goal Efficiency (FGE)
    data['FGE'] = data['FGM'] / data['FGA']

    # Turnover Rate (TOV%)
    data['TOV%'] = data['TOV'] / (data['FGA'] + data['TOV'] + 0.44 * data['FTA'])

    # Offensive Rebounding Percentage (OREB%)
    data['OREB%'] = data['OREB'] / (data['OREB'] + data['DREB'])

    # Free Throw Rate (FTR)
    data['FTR'] = data['FTA'] / data['FGA']

    # Calculate Strength of Gameplay (SoP)
    data['SoP'] = (
        0.4 * data['FGE'] -
        0.25 * data['TOV%'] +
        0.2 * data['OREB%'] +
        0.15 * data['FTR']
    )

    # Add Region (North = 1, South = 0)
    data['Region'] = data['TEAM_ABBREVIATION'].map(team_regions)

    # Drop rows with missing data
    data = data.dropna()

    return data

# Function to perform regression
def perform_regression(data):
    # Dependent variable: SoP
    Y = data['SoP']  # Use SoP directly without pre-defined weights

    # Independent variables
    X = data['Region']
    X = sm.add_constant(X)

    # Fit regression model
    model = sm.OLS(Y, X).fit()

    print(model.summary())



# Regression on filtered dataset
def perform_regression(filtered_data):
    Y = filtered_data['SoP']  # Dependent variable
    X = filtered_data[['Region']]  # Independent variable
    X = sm.add_constant(X)

    # Fit regression model
    model = sm.OLS(Y, X).fit()
    print(model.summary())


# Main workflow
if __name__ == "__main__":
    # Specify the seasons you want to analyze
    seasons = ['2010-11', '2011-12', '2012-33', '2013-14', '2014-15', '2015-16', '2016-17', '2018-19', '2020-21']

    # Fetch data for multiple seasons
    game_data = fetch_multiple_seasons(seasons)

    # Prepare the data
    processed_data = prepare_data(game_data)
    filtered_data = filter_away_games_outside_region(processed_data)



    # Perform regression analysis
    perform_regression(processed_data)
    #perform_regression(filtered_data)


#Visuals

#Regional Comparison
#To compare gameplay SoP across regions.
plt.figure(figsize=(8, 6))
sns.boxplot(x='Region', y='SoP', data=processed_data)
plt.title('Strength of Gameplay (SoP) by Region')
plt.xlabel('Region (0 = South, 1 = North)')
plt.ylabel('Strength of Gameplay (SoP)')
plt.show()

#Regression Analysis

#Visualize the relationship between SoP and Region
sns.lmplot(x='Region', y='SoP', data=processed_data, ci=95, height=6, aspect=1.5)
plt.title('Regression Analysis: SoP vs. Region')
plt.xlabel('Region (0 = South, 1 = North)')
plt.ylabel('Strength of Gameplay (SoP)')
plt.show()

#Analyze how SoP vary over seasons
seasonal_data = processed_data.groupby('Season')['SoP'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Season', y='SoP', data=seasonal_data, marker='o')
plt.title('Average Strength of Gameplay (SoP) Over Seasons')
plt.xlabel('Season')
plt.ylabel('Average SoP')
plt.xticks(rotation=45)
plt.show()

#Correlation between different metrics to understand relationships.
plt.figure(figsize=(10, 8))
correlation_matrix = processed_data[['FGE', 'TOV%', 'OREB%', 'FTR', 'SoP']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Gameplay Metrics')
plt.show()

#comparing team performance
# Calculate average metrics by team
team_avg_metrics = processed_data.groupby('TEAM_ABBREVIATION')[['SoP', 'FGE', 'TOV%', 'OREB%']].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='TEAM_ABBREVIATION', y='SoP', data=team_avg_metrics, color='green')
plt.title('Average Strength of Gameplay (SoP) by Team')
plt.xlabel('Team')
plt.ylabel('SoP')
plt.xticks(rotation=45)
plt.show()

#using elbow method to determine the optimal number of clusters to apply to kmeans 
def elbow_method(data):
    features = data[['FGE', 'TOV%', 'OREB%', 'FTR', 'SoP', 'Region']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    inertia = []
    for k in range(1, 11):  
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
#plotting elbow curve 
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K (with Region)')
    plt.show()
elbow_method(processed_data)

# functiong for performing k-means clustering between SoP and FGE across multiple seasons 
def perform_kmeans(data, n_clusters):
    features = data[['FGE', 'TOV%', 'OREB%', 'FTR', 'SoP', 'Region']]
    
    # standardize how data is weighed using standard scaler 
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # printing cluster centers
    print("Cluster Centers (scaled):")
    print(kmeans.cluster_centers_)
    
    return data, kmeans

# Visualize clustering 
def visualize_clusters_with_region(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['FGE'], data['SoP'], c=data['Cluster'], cmap='plasma', alpha=0.7)
    plt.xlabel('FGE')
    plt.ylabel('Strength of Play (SoP)')
    plt.title('K-Means Clustering by GamePlay Features')
    plt.colorbar(label='Cluster')
    plt.show()


if __name__ == "__main__":
    seasons = ['2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2018-19', '2020-21']
    game_data = fetch_multiple_seasons(seasons)
    processed_data = prepare_data(game_data)
    # Expriment clustering between 3-4 clusters 
    clustered_data, kmeans_model = perform_kmeans(processed_data, n_clusters=4)
    visualize_clusters_with_region(clustered_data)

# add comment
sns.pairplot(
    processed_data[['SoP', 'FGE', 'TOV%', 'OREB%', 'FTR', 'Region']],
    hue='Region',
    diag_kind='kde',
    palette='coolwarm'
)
plt.suptitle('Pair Plot of Gameplay Metrics by Region', y=1.02)
plt.show()
