"""
Mateo Villalobos

CSE 163

"""
import pandas as pd
import pprint  # Used so I can read the dictionaries easier


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a pandas dataframe.
    """
    df = pd.read_csv(file_path)
    return df


def get_starting_qbs(df, teams):
    starting_qbs = {'CLE': {}, 'WAS': {}, 'NYG': {}, 'NYJ': {}}

    for team in teams:
        for season in range(5):
            year = season + 2021
            game = (df[(df['season'] == year) & (df['week'] == 1) &
                    ((df['home_team'] == team) | (df['away_team'] == team))])
            if game['home_team'].iloc[0] == team:
                qb_name = game['home_qb_name'].iloc[0]
            else:
                qb_name = game['away_qb_name'].iloc[0]
            starting_qbs[team][year] = qb_name

    return starting_qbs


def get_qb_games(df, teams, starting_qbs):
    qb_games = []

    for team in teams:
        for season in range(5):
            year = season + 2021
            season_starter = starting_qbs[team][year]
            for week in range(1, 19):
                filtered_df = df[['season', 'week', 'home_team', 'away_team',
                                  'home_qb_name', 'away_qb_name']]
                game = (filtered_df[(filtered_df['season'] == year) &
                        (filtered_df['week'] == week) &
                        ((filtered_df['home_team'] == team) |
                        (filtered_df['away_team'] == team))])
                if not game.empty:
                    if game['home_team'].iloc[0] == team:
                        qb_name = game['home_qb_name'].iloc[0]
                    elif game['away_team'].iloc[0] == team:
                        qb_name = game['away_qb_name'].iloc[0]
                    qb_games.append({
                        'team': team,
                        'season': year,
                        'week': week,
                        'qb': qb_name,
                        'backup_qb': qb_name != season_starter
                    })

    return qb_games


df = read_csv_file('data/games.csv')
teams = ['CLE', 'WAS', 'NYG', 'NYJ']
starters = get_starting_qbs(df, teams)
backups = get_qb_games(df, teams, starters)
pprint.pprint(backups)
