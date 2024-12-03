from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import statsmodels.api as sm

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
