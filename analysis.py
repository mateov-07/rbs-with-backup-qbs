"""
Mateo Villalobos

CSE 163

File that analyzes how a backup quarterback affects his team's starting
running back. Combines dataframe schedule and weekly player stats for the
2021-2025 regular season. Labels each team's QB as either the week 1 starter
or the season backup. Compares RB rushing efficiency, workload, and fantasy
output through run defense metrics and general comparisons.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # Used to run t-tests for research questions
RB_COLUMNS = ['player_id', 'season', 'week', 'team', 'opponent_team',
              'game_id', 'carries', 'rushing_yards', 'rushing_tds',
              'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
              'targets', 'receptions', 'receiving_yards', 'receiving_tds',
              'receiving_epa', 'target_share', 'fantasy_points',
              'fantasy_points_ppr']
TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
         'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
         'LV', 'LAC', 'LA', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
         'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas dataframe.
    """
    if 'weekly-player-stats' in file_path:
        list_columns = ['fg_made_list', 'fg_missed_list', 'fg_blocked_list']
        # Done to supress mixed types terminal warning
        return pd.read_csv(file_path, dtype={col: str for col in list_columns})

    return pd.read_csv(file_path)


def load_weekly_stats() -> pd.DataFrame:
    """
    Reads the weekly player stats files and combines them into one dataframe.
    Returns a dataframe with only the regular season games and the 2021-2025
    season range.
    """
    seasons = []

    for season in range(5):
        year = season + 2021
        df = read_csv_file('data/weekly-player-stats/' +
                           f'stats_player_week_{year}.csv')
        seasons.append(df)

    mask = pd.concat(seasons, ignore_index=True)
    return mask[mask['week'] <= 18]


def get_starting_qbs(df: pd.DataFrame,
                     teams: list[str]) -> dict[str, dict[int, str]]:
    """
    Takes a schedules dataframe and a list of teams and returns a dictionary
    mapping each team to another dictionary of each season to the name of the
    QB who started that team's week 1 game.
    """
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


def get_qb_games(df: pd.DataFrame, teams: list[str],
                 starting_qbs: dict[str, dict[int, str]]) -> pd.DataFrame:
    """
    Takes a schedules dataframe, a list of teams and a dictionary of week 1
    starters and returns a dataframe that lists the QB games and whether they
    were the week 1 starter or not (starter/backup definition).
    """
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


def get_starter_rbs(rb_weekly_stats: pd.DataFrame,
                    teams: list[str]) -> pd.DataFrame:
    """
    Takes weekly stats for RBs and a list of teams and returns a dataframe
    with each team's RB with the most carries per season.
    """
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


def get_starter_rb_games(rb_weekly_stats: pd.DataFrame,
                         starter_rbs: pd.DataFrame) -> pd.DataFrame:
    """
    Takes weekly stats for RBs and a dataframe of starting RBs, and returns
    the season, team and player ID of every game a starting RB played for
    his team that season.
    """
    return rb_weekly_stats[RB_COLUMNS].merge(starter_rbs,
                                             on=['season', 'team',
                                                 'player_id'])


def get_combined_dataframe(df: pd.DataFrame,
                           teams: list[str]) -> pd.DataFrame:
    """
    Takes a schedules dataframe and a list of teams, and returns a dataframe
    with every game the starting RBs played, along with information about
    QBs, RBs and opponent run defenses.
    """
    weekly = load_weekly_stats()
    rb_weekly_stats = weekly[(weekly['position'] == 'RB') &
                             (weekly['carries'] > 0)]

    starter_qbs = get_starting_qbs(df, teams)
    qb_games = get_qb_games(df, teams, starter_qbs)
    starter_rbs = get_starter_rbs(rb_weekly_stats, teams)
    starter_rb_games = get_starter_rb_games(rb_weekly_stats, starter_rbs)

    rushing = weekly[weekly['position'] == 'RB']
    defenses = rushing.groupby(['season', 'opponent_team'],
                               as_index=False)

    run_defense = defenses[['rushing_epa', 'carries']].sum()
    epa_allowed = run_defense['rushing_epa']
    carries_allowed = run_defense['carries']
    run_defense['opp_epa_allowed'] = epa_allowed / carries_allowed
    run_defense = run_defense[['season', 'opponent_team', 'opp_epa_allowed']]

    team_groups = weekly.groupby(['season', 'week', 'team'], as_index=False)
    team_totals = team_groups[['carries', 'receptions']].sum()

    combined = qb_games.merge(starter_rb_games, on=['season', 'week', 'team'])
    combined = combined.merge(team_totals, on=['season', 'week', 'team'],
                              suffixes=('', '_team'))
    combined = combined.merge(run_defense, on=['season', 'opponent_team'])

    return combined


def research_question_one(df: pd.DataFrame) -> None:
    """
    Compares the starting RB's average EPA per rush in starter and backup
    QB games paired by team and season. Prints the means of both roles and
    the paired t-test result, and graphs a scatter plot of averages with
    the general mean marked.
    """
    df['epa_per_rush'] = df['rushing_epa'] / df['carries']
    starters = df[(~df['backup_qb']) & (df['carries'] >= 6)]
    backups = df[(df['backup_qb']) & (df['carries'] >= 6)]
    starter_epa = starters.groupby(['team', 'season'],
                                   as_index=False)['epa_per_rush'].mean()
    backup_epa = backups.groupby(['team', 'season'],
                                 as_index=False)['epa_per_rush'].mean()

    paired = starter_epa.merge(backup_epa, on=['team', 'season'],
                               suffixes=('_starter', '_backup'))
    ttest_result = stats.ttest_rel(paired['epa_per_rush_starter'],
                                   paired['epa_per_rush_backup'])
    starter_epa_mean = paired['epa_per_rush_starter'].mean()
    backup_epa_mean = paired['epa_per_rush_backup'].mean()

    print('Starter mean:', starter_epa_mean)
    print('Backup mean:', backup_epa_mean)
    print('t =', ttest_result.statistic, 'p =', ttest_result.pvalue)

    sns.relplot(data=paired, x='epa_per_rush_starter',
                y='epa_per_rush_backup', hue='team')
    plt.title('Starter QBs vs Backup QBs: RB EPA per rush')
    plt.xlabel('RB EPA per rush with starting QB')
    plt.ylabel('RB EPA per rush with backup QB')
    # Used to mark the averages of RB EPAs via a scatterplot annotation
    plt.scatter(starter_epa_mean, backup_epa_mean, color='black')
    plt.annotate('Average', (starter_epa_mean, backup_epa_mean),
                 xytext=(-20, 8), textcoords='offset points')
    plt.savefig('rq1.png', bbox_inches='tight')


def research_question_two(df: pd.DataFrame) -> None:
    """
    Compares the starting RB's carries per game, team touch percentage,
    and PPR fantasy points in starter and backup QB games, paired by
    team and season. Prints a paired t-test result for each measure and
    graphs a box plot of each measure by QB role.
    """
    df['touches'] = df['carries'] + df['receptions']
    df['team_touches'] = df['carries_team'] + df['receptions_team']
    df['touch_percentage'] = df['touches'] / df['team_touches'] * 100

    metrics = ['carries', 'touch_percentage', 'fantasy_points_ppr']
    labels = {'carries': 'Carries per game',
              'touch_percentage': 'Team touch percentage',
              'fantasy_points_ppr': 'Fantasy PPR points per game'}

    groups = df.groupby(['team', 'season', 'backup_qb'],
                        as_index=False)[metrics].mean()
    starter_avgs = groups[~groups['backup_qb']]
    backup_avgs = groups[groups['backup_qb']]
    paired = starter_avgs.merge(backup_avgs, on=['team', 'season'],
                                suffixes=('_starter', '_backup'))
    groups['qb_role'] = 'Starter QB'
    groups.loc[groups['backup_qb'], 'qb_role'] = 'Backup QB'

    for metric in metrics:
        ttest_result = stats.ttest_rel(paired[metric + '_starter'],
                                       paired[metric + '_backup'])

        print(labels[metric], 't =', ttest_result.statistic,
              'p =', ttest_result.pvalue)

        sns.catplot(data=groups, x='qb_role', y=metric, kind='box')
        plt.title(f'Starting RB {labels[metric]} by QB Role')
        plt.xlabel('QB Role')
        plt.ylabel(labels[metric])
        plt.savefig(f'rq2_{metric}.png', bbox_inches='tight')


def research_question_three(df: pd.DataFrame) -> None:
    """
    Compares the starting RB's raw and opponent-adjusted EPA per rush
    in starter and backup QB games, and whether the two faced differently-
    strengthed run defenses using paired t-tests. Graphs a scatterplot of
    EPA per rush against opponent run defenses with a best-fit line for
    each QB role.
    """
    df['epa_per_rush'] = df['rushing_epa'] / df['carries']
    df['adjusted_epa'] = df['epa_per_rush'] - df['opp_epa_allowed']
    min_carries = df[df['carries'] >= 6].copy()

    metrics = ['epa_per_rush', 'adjusted_epa', 'opp_epa_allowed']
    labels = {'epa_per_rush': 'Raw EPA per rush',
              'adjusted_epa': 'Opponent-adjusted EPA per rush',
              'opp_epa_allowed': 'Opponent EPA allowed per rush'}

    groups = min_carries.groupby(['team', 'season', 'backup_qb'],
                                 as_index=False)[metrics].mean()
    starter_avgs = groups[~groups['backup_qb']]
    backup_avgs = groups[groups['backup_qb']]
    paired = starter_avgs.merge(backup_avgs, on=['team', 'season'],
                                suffixes=('_starter', '_backup'))

    for metric in metrics:
        ttest_result = stats.ttest_rel(paired[metric + '_starter'],
                                       paired[metric + '_backup'])

        print(labels[metric], 't =', ttest_result.statistic,
              'p =', ttest_result.pvalue)

    min_carries['qb_role'] = 'Starter QB'
    min_carries.loc[min_carries['backup_qb'], 'qb_role'] = 'Backup QB'

    sns.lmplot(data=min_carries, x='opp_epa_allowed', y='epa_per_rush',
               hue='qb_role')
    plt.title('Starting RB EPA per rush vs. opponent run defense')
    plt.xlabel('Opponent EPA allowed per rush')
    plt.ylabel('Starting RB EPA per rush')
    plt.savefig('rq3.png', bbox_inches='tight')


def main():
    df = read_csv_file('data/games.csv')
    combined_df = get_combined_dataframe(df, TEAMS)
    print("Research question 1:")
    research_question_one(combined_df)
    print()
    print("Research question 2:")
    research_question_two(combined_df)
    print()
    print("Research question 3:")
    research_question_three(combined_df)


if __name__ == '__main__':
    main()
