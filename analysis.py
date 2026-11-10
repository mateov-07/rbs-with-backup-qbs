"""
Mateo Villalobos

CSE 163

"""
import pandas as pd
# import pprint  # Used so I can read the dictionaries easier


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a pandas dataframe.
    """
    if 'weekly-player-stats' in file_path:
        list_columns = ['fg_made_list', 'fg_missed_list', 'fg_blocked_list']
        # Done to supress mixed types terminal warning
        return pd.read_csv(file_path, dtype={col: str for col in list_columns})

    return pd.read_csv(file_path)


def get_starting_qbs(df, teams):
    starting_qbs = {"CLE": {}, "WAS": {}, "NYG": {}, "NYJ": {}}

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

    return pd.DataFrame(qb_games)


def get_starter_rbs(teams):
    starting_rbs = []

    for season in range(5):
        year = season + 2021
        df = read_csv_file('data/weekly-player-stats/' +
                           f'stats_player_week_{year}.csv')
        filtered_df = df[['player_id', 'player_display_name', 'position',
                          'season', 'week', 'team', 'carries']]
        for team in teams:
            rb_games = (filtered_df[(filtered_df['position'] == 'RB') &
                        (filtered_df['week'] <= 18) &
                        (filtered_df['team'] == team) &
                        (filtered_df['carries'] > 0)])
            player_groups = rb_games.groupby(['player_id',
                                             'player_display_name'],
                                             as_index=False)
            total_carries = player_groups['carries'].sum()
            top_row = total_carries['carries'].idxmax()
            starting_rbs.append({
                'player_id': total_carries.loc[top_row, 'player_id'],
                'team': team,
                'season': year,
                'rb': total_carries.loc[top_row, 'player_display_name']
            })

    return pd.DataFrame(starting_rbs)


def get_starter_rb_games(starter_rbs):
    seasons = []

    for season in range(5):
        year = season + 2021
        df = read_csv_file('data/weekly-player-stats/' +
                           f'stats_player_week_{year}.csv')
        seasons.append(df)
    weekly = pd.concat(seasons, ignore_index=True)
    mask = weekly[(weekly['week'] <= 18) & (weekly['carries'] > 0)]
    return mask.merge(starter_rbs, on=['season', 'team', 'player_id'])


def get_combined_dataframe(df, teams):
    starter_qbs = get_starting_qbs(df, teams)
    qb_games = get_qb_games(df, teams, starter_qbs)
    starter_rbs = get_starter_rbs(teams)
    starter_rb_games = get_starter_rb_games(starter_rbs)
    combined = qb_games.merge(starter_rb_games, on=['season', 'team', 'team'])
    #combined.to_csv('test.csv', index=False)


def research_question_one(df):
    print("hi")
    


df = read_csv_file('data/games.csv')
teams = ['CLE', 'WAS', 'NYG', 'NYJ']
combined_df = get_combined_dataframe(df, teams)
research_question_one(combined_df)