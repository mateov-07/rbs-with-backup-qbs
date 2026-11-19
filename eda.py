"""
Mateo Villalobos
CSE 163

Exploratory data analysis for analysis.py. All of the information here
is used within the EDA Results section of the writeup.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # Used to run t-tests for research questions

import analysis

SUMMARIES = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
RQ_METRICS = {'epa_per_rush': 'RQ1: EPA per rush',
              'carries': 'RQ2: Carries per game',
              'touch_percentage': 'RQ2: Team touch percentage',
              'fantasy_points_ppr': 'RQ2: PPR fantasy points',
              'adjusted_epa': 'RQ3: Opponent-adjusted EPA per rush',
              'opp_epa_allowed': 'RQ3: Opponent EPA allowed per rush'}
MIN_CARRY_METRICS = ['epa_per_rush', 'adjusted_epa', 'opp_epa_allowed']
WEEKLY_VARIABLES = ['carries', 'rushing_epa', 'receptions']
COMBINED_VARIABLES = ['carries', 'rushing_yards', 'rushing_epa',
                      'epa_per_rush', 'receptions', 'touch_percentage',
                      'fantasy_points_ppr', 'opp_epa_allowed', 'adjusted_epa',
                      'carries_team', 'receptions_team']


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    counts = counts[counts > 0]
    percents = (counts / len(df) * 100).round(1)

    return pd.DataFrame({'missing': counts,
                         'percent':
                         percents}).sort_values('missing')


def seven_number_summary(df: pd.DataFrame,
                         columns: list[str]) -> pd.DataFrame:
    return df[columns].describe().T[SUMMARIES].round(3)


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['epa_per_rush'] = df['rushing_epa'] / df['carries']
    touches = df['carries'] + df['receptions']
    team_touches = df['carries_team'] + df['receptions_team']
    df['touch_percentage'] = touches / team_touches * 100
    df['adjusted_epa'] = df['epa_per_rush'] - df['opp_epa_allowed']
    df['qb_role'] = 'Starter QB'
    df.loc[df['backup_qb'], 'qb_role'] = 'Backup QB'

    return df


def get_paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    differences = []

    for metric in RQ_METRICS:
        games = df
        if metric in MIN_CARRY_METRICS:
            games = df[df['carries'] >= 6]
        groups = games.groupby(['team', 'season', 'backup_qb'],
                               as_index=False)[metric].mean()

        paired = groups[~groups['backup_qb']].merge(
            groups[groups['backup_qb']], on=['team', 'season'],
            suffixes=('_starter', '_backup'))
        paired['metric'] = metric
        paired['difference'] = (paired[metric + '_starter'] -
                                paired[metric + '_backup'])
        differences.append(paired[['team', 'season', 'metric',
                                   'difference']])

    return pd.concat(differences, ignore_index=True)


def weekly_eda(weekly: pd.DataFrame, starter_rbs: pd.DataFrame) -> None:
    print('Weekly stats shape (rows, columns):', weekly.shape)
    print()

    # Turns it into dictionary so it can be printed cleanly
    print('Rows per season:', weekly.groupby('season').size().to_dict())
    print()

    print('Columns with missing data:')
    print(missing_summary(weekly))
    print()

    no_id = weekly[weekly['player_id'].isna()]
    print('Rows without a player_id:', len(no_id),
          '| their total carries:', no_id['carries'].sum())
    print()

    rushers = weekly[weekly['carries'] > 0]
    print('Rows with carries but no rushing_epa:',
          rushers['rushing_epa'].isna().sum())
    print()

    print('Seven-number summaries (all players):')
    print(seven_number_summary(weekly, WEEKLY_VARIABLES))
    print()

    print('Position counts:')
    # Converts to string type so it can be printed, used a lot for eda.py
    print(weekly['position'].value_counts(dropna=False).to_string())
    print()

    rb_weekly_stats = weekly[(weekly['position'] == 'RB') &
                             (weekly['carries'] > 0)].copy()
    starters = rb_weekly_stats.merge(starter_rbs[['player_id', 'team',
                                                  'season']],
                                     on=['player_id', 'team', 'season'],
                                     how='left', indicator=True)
    starters['rb_type'] = 'Other RBs'
    starters.loc[starters['_merge'] == 'both', 'rb_type'] = 'Starting RB'
    print('RB rows with at least one carry:', len(starters))
    print(starters['rb_type'].value_counts().to_string())
    print()

    plt.figure(figsize=(7, 4))
    # histplot used to create histogram for readability
    ax = sns.histplot(data=starters, x='carries', hue='rb_type',
                      binwidth=1, multiple='stack',
                      hue_order=['Other RBs', 'Starting RB'],
                      palette={'Starting RB': '#2a78d6',
                               'Other RBs': '#9a9893'})
    # plt.ylim sets y-axis range to 0-500
    plt.text(6.4, plt.ylim()[1] * 0.92, '6-carry minimum', fontsize=9)
    # move_legend moves legend to given position without title
    sns.move_legend(ax, 'upper right', title=None)
    plt.title('RB carries per game, 2021-2025 regular seasons')
    plt.xlabel('Carries in the game')
    plt.ylabel('Number of RB games')
    plt.savefig('eda-imgs/eda_rb_carries.png', bbox_inches='tight', dpi=200)
    # Closes plt so I can make a completely new graph later in weekly_eda
    plt.close()

    rushing = weekly[weekly['position'] == 'RB']
    defenses = rushing.groupby(['season', 'opponent_team'],
                               as_index=False)

    run_defense = defenses[['rushing_epa', 'carries']].sum()
    epa_allowed = run_defense['rushing_epa']
    carries_allowed = run_defense['carries']
    run_defense['opp_epa_allowed'] = epa_allowed / carries_allowed
    print('Run defense ratings (team-seasons):', len(run_defense))
    print(seven_number_summary(run_defense, ['opp_epa_allowed']))
    print()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=run_defense, x='season', y='opp_epa_allowed',
                color='#cde2fb')
    plt.title('Opponent run defense ratings by season')
    plt.xlabel('Season')
    plt.ylabel('RB rushing EPA allowed per carry')
    plt.savefig('eda-imgs/eda_run_defense.png', bbox_inches='tight', dpi=200)


def combined_eda(combined: pd.DataFrame) -> None:
    print('Combined shape (rows, columns):', combined.shape)
    print()

    print('Columns with missing data:')
    print(missing_summary(combined))
    print()

    print('Seven-number summaries:')
    print(seven_number_summary(combined, COMBINED_VARIABLES))
    print()

    print('backup_qb counts:')
    print(combined['backup_qb'].value_counts().to_string())
    print()

    print('season counts:')
    print(combined['season'].value_counts().sort_index().to_string())
    print()

    min_carries = combined[combined['carries'] >= 6].copy()
    print()

    print('Games with 6+ carries:', len(min_carries))
    print(min_carries['qb_role'].value_counts().to_string())

    plt.figure(figsize=(7, 4))
    ax = sns.countplot(data=combined, x='season', hue='qb_role',
                       hue_order=['Starter QB', 'Backup QB'],
                       palette={'Starter QB': '#2a78d6',
                                'Backup QB': '#eb6834'})
    plt.ylim(0, 500)
    plt.title('Starting RB games by season and QB role')
    plt.xlabel('Season')
    plt.ylabel('Number of starting RB games')
    sns.move_legend(ax, 'upper center', ncol=2, title='QB role')
    plt.savefig('eda-imgs/eda_games_by_role.png', bbox_inches='tight', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    ax = sns.histplot(data=min_carries, x='epa_per_rush', hue='qb_role',
                      hue_order=['Starter QB', 'Backup QB'],
                      palette={'Starter QB': '#2a78d6',
                               'Backup QB': '#eb6834'},
                      stat='density',
                      common_norm=False, element='step', binwidth=0.1)
    sns.move_legend(ax, 'upper right', title='QB role')
    plt.title('Starting RB EPA per rush by QB role (6+ carries)')
    plt.xlabel('EPA per rush in the game')
    plt.ylabel('Density')
    plt.savefig('eda-imgs/eda_epa_by_role.png', bbox_inches='tight', dpi=200)


def check_assumptions(combined: pd.DataFrame) -> None:
    differences = get_paired_differences(combined)
    pairs = differences[['team', 'season']].drop_duplicates()
    print('Team-seasons with both QB roles:', len(pairs))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (metric, label) in zip(axes.flat, RQ_METRICS.items()):
        diff = differences[differences['metric'] == metric]['difference']
        print(label, '| n =', len(diff),
              '| skew =', round(stats.skew(diff), 2))
        sns.histplot(diff, ax=ax, color='#2a78d6', bins=15)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Starter minus backup')
        ax.set_ylabel('Team-seasons')
    # Main title above all six graphs
    fig.suptitle('Paired differences (starter QB games minus backup QB games)')
    # Makes labels be able to fit and become readable in the small graphs
    fig.tight_layout()
    fig.savefig('eda-imgs/eda_paired_differences.png',
                bbox_inches='tight', dpi=200)


def main():
    df = analysis.read_csv_file('data/games.csv')
    schedules = df[(df['season'] >= 2021) & (df['season'] <= 2025) &
                   (df['week'] <= 18)]

    print('Schedules shape, 2021-2025, Weeks 1-18:', schedules.shape)
    print('Missing QB names:',
          schedules[['home_qb_name', 'away_qb_name']].isna().sum().sum())
    print()

    weekly = analysis.load_weekly_stats()
    rb_weekly_stats = weekly[(weekly['position'] == 'RB') &
                             (weekly['carries'] > 0)]
    starter_rbs = analysis.get_starter_rbs(rb_weekly_stats, analysis.TEAMS)
    print('EDA: stacked weekly player stats')
    weekly_eda(weekly, starter_rbs)
    print()

    combined_df = add_columns(
        analysis.get_combined_dataframe(df, analysis.TEAMS))
    print('EDA: schedules joined with weekly stats')
    combined_eda(combined_df)
    print()

    print('Paired t-test assumption checks')
    check_assumptions(combined_df)


if __name__ == '__main__':
    main()
