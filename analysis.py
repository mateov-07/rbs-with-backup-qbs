"""
Mateo Villalobos

CSE 163

"""
import pandas as pd
from scipy import stats  # Used to run t-tests for research questions
RB_COLUMNS = ['player_id', 'season', 'week', 'team', 'opponent_team',
              'game_id', 'carries', 'rushing_yards', 'rushing_tds',
              'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
              'targets', 'receptions', 'receiving_yards', 'receiving_tds',
              'receiving_epa', 'target_share', 'fantasy_points',
              'fantasy_points_ppr']


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a pandas dataframe.
    """
    if 'weekly-player-stats' in file_path:
        list_columns = ['fg_made_list', 'fg_missed_list', 'fg_blocked_list']
        # Done to supress mixed types terminal warning
        return pd.read_csv(file_path, dtype={col: str for col in list_columns})

    return pd.read_csv(file_path)


def load_weekly_stats():
    seasons = []

    for season in range(5):
        year = season + 2021
        df = read_csv_file('data/weekly-player-stats/' +
                           f'stats_player_week_{year}.csv')
        seasons.append(df)

    mask = pd.concat(seasons, ignore_index=True)
    return mask[mask['week'] <= 18]


def get_starting_qbs(df, teams):
    starting_qbs = {}

    for team in teams:
        starting_qbs[team] = {}
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
    filtered_df = df[['season', 'week', 'home_team', 'away_team',
                      'home_qb_name', 'away_qb_name']]

    for team in teams:
        for season in range(5):
            year = season + 2021
            season_starter = starting_qbs[team][year]
            for week in range(1, 19):
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


def get_starter_rbs(rb_weekly_stats, teams):
    starting_rbs = []

    for season in range(5):
        year = season + 2021
        for team in teams:
            rb_games = rb_weekly_stats[(rb_weekly_stats['season'] == year) &
                                       (rb_weekly_stats['team'] == team)]
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


def get_starter_rb_games(rb_weekly_stats, starter_rbs):
    return rb_weekly_stats[RB_COLUMNS].merge(starter_rbs,
                                       on=['season', 'team', 'player_id'])


def get_combined_dataframe(df, teams):
    weekly = load_weekly_stats()
    rb_weekly_stats = weekly[(weekly['position'] == 'RB') & 
                             (weekly['carries'] > 0)]

    starter_qbs = get_starting_qbs(df, teams)
    qb_games = get_qb_games(df, teams, starter_qbs)
    starter_rbs = get_starter_rbs(rb_weekly_stats, teams)
    starter_rb_games = get_starter_rb_games(rb_weekly_stats, starter_rbs)

    combined = qb_games.merge(starter_rb_games, on=['season', 'week', 'team'])
    # combined.to_csv('test.csv', index=False)
    return combined


def research_question_one(df):
    df['epa_per_rush'] = df['rushing_epa'] / df['carries']
    starters = df[(~df['backup_qb']) & (df['carries'] >= 6)]
    backups = df[(df['backup_qb']) & (df['carries'] >= 6)]
    starter_epa = starters.groupby(['team', 'season'],
                                   as_index=False)['epa_per_rush'].mean()
    backup_epa = backups.groupby(['team', 'season'],
                                 as_index=False)['epa_per_rush'].mean()
    paired = starter_epa.merge(backup_epa, on=['team', 'season'],
                            suffixes=('_starter', '_backup'))
    result = stats.ttest_rel(paired['epa_per_rush_starter'],
                             paired['epa_per_rush_backup'])
    print(result.statistic, result.pvalue)
    print(len(paired))
    # starter_epa.to_csv('starters.csv')
    # backup_epa.to_csv('backups.csv')


df = read_csv_file('data/games.csv')
teams = ['CLE', 'NYG', 'NYJ', 'WAS']
#teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
#         'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 
#         'LV', 'LAC', 'LA', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 
#         'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
combined_df = get_combined_dataframe(df, teams)
research_question_one(combined_df)
