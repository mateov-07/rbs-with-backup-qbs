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


def main():
    df = analysis.read_csv_file('data/games.csv')
    schedules = df[(df['season'] >= 2021) & (df['season'] <= 2025) &
                   (df['week'] <= 18)]
    print('Schedules shape, 2021-2025, Weeks 1-18:', schedules.shape)
    print('Missing QB names:',
          schedules[['home_qb_name', 'away_qb_name']].isna().sum().sum())


if __name__ == '__main__':
    main()
