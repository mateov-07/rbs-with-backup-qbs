"""
Mateo Villalobos

CSE 163

Test file for analysis.py and eda.py to ensure that each function
returns the correct answer and targets edge cases.
"""
import math
import pandas as pd

import analysis
import eda

YEARS = [2021, 2022, 2023, 2024, 2025]
TEST_TEAMS = ['TM1', 'TM2']


def test_load_weekly_stats() -> None:
    """
    Tests load_weekly_stats within analysis.py.
    """
    weekly = analysis.load_weekly_stats()

    assert sorted(weekly['season'].unique().tolist()) == YEARS
    assert weekly['week'].min() == 1
    assert weekly['week'].max() == 18


def test_get_starting_qbs() -> None:
    """
    Tests get_starting_qbs within analysis.py.
    """
    schedule = analysis.read_csv_file('data/test-data/test_schedule.csv')
    starters = analysis.get_starting_qbs(schedule, TEST_TEAMS)

    assert starters['TM1'][2021] == 'Team 1 Starter'
    assert starters['TM1'][2022] == 'Team 1 New'
    assert starters['TM2'][2021] == 'Team 2 Starter'
    assert starters['TM2'][2025] == 'Team 2 Starter'
    assert sorted(starters['TM1'].keys()) == YEARS


def test_get_qb_games() -> None:
    """
    Tests get_qb_games within analysis.py.
    """
    schedule = analysis.read_csv_file('data/test-data/test_schedule.csv')
    starters = analysis.get_starting_qbs(schedule, TEST_TEAMS)
    qb_games = analysis.get_qb_games(schedule, TEST_TEAMS, starters)

    assert len(qb_games) == 14

    backups = qb_games[qb_games['backup_qb']]
    assert len(backups) == 2

    tm1_backup = backups[backups['team'] == 'TM1'].iloc[0]
    assert tm1_backup['qb'] == 'Team 1 Backup'
    assert tm1_backup['season'] == 2021 and tm1_backup['week'] == 2

    tm2_backup = backups[backups['team'] == 'TM2'].iloc[0]
    assert tm2_backup['qb'] == 'Team 2 Backup' and tm2_backup['week'] == 3

    tm1_2022 = qb_games[(qb_games['team'] == 'TM1') &
                        (qb_games['season'] == 2022)].iloc[0]
    assert tm1_2022['qb'] == 'Team 1 New'
    assert not tm1_2022['backup_qb']


def test_get_starter_rbs() -> None:
    """
    Tests get_starter_rbs within analysis.py.
    """
    rb_stats = analysis.read_csv_file('data/test-data/test_rb_stats.csv')
    starter_rbs = analysis.get_starter_rbs(rb_stats, TEST_TEAMS)

    assert len(starter_rbs) == 10

    tm1_2021 = starter_rbs[(starter_rbs['team'] == 'TM1') &
                           (starter_rbs['season'] == 2021)].iloc[0]
    assert tm1_2021['player_id'] == 'tm1_rb1'
    assert tm1_2021['rb'] == 'Team 1 RB1'


def test_get_starter_rb_games() -> None:
    """
    Tests get_starter_rb_games within analysis.py.
    """
    rb_stats = analysis.read_csv_file('data/test-data/test_rb_stats.csv')
    starter_rbs = analysis.get_starter_rbs(rb_stats, TEST_TEAMS)
    games = analysis.get_starter_rb_games(rb_stats, starter_rbs)

    tm1_2021 = games[(games['team'] == 'TM1') & (games['season'] == 2021)]
    assert sorted(tm1_2021['week'].tolist()) == [1, 3]
    assert set(tm1_2021['player_id']) == {'tm1_rb1'}
    assert 'tm1_rb2' not in set(games['player_id'])
    assert len(games) == 11


def test_get_combined_dataframe(combined: pd.DataFrame) -> None:
    """
    Tests get_combined_dataframe within analysis.py.
    """
    sf = combined[(combined['team'] == 'SF') & (combined['season'] == 2022)]

    assert set(sf['rb']) == {'Christian McCaffrey'}
    assert sf['week'].min() >= 7
    assert not combined.duplicated(['team', 'season', 'week']).any()
    assert (combined['carries'] >= 1).all()
    assert (combined['carries'] <= combined['carries_team']).all()
    assert (combined.groupby(['team', 'season'])['player_id'].nunique()
            == 1).all()
    assert (combined.groupby(['season', 'opponent_team'])['opp_epa_allowed']
            .nunique() == 1).all()


def test_missing_summary() -> None:
    """
    Tests missing_summary within eda.py.
    """
    df = pd.DataFrame({'a': [1, 2, 3, 4],
                       'b': [1, None, 3, 4],
                       'c': [None, None, 3, 4]})
    summary = eda.missing_summary(df)

    assert summary.index.tolist() == ['b', 'c']
    assert summary['missing'].tolist() == [1, 2]
    assert summary['percent'].tolist() == [25.0, 50.0]


def test_seven_number_summary() -> None:
    """
    Tests seven_number_summary within eda.py.
    """
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    summary = eda.seven_number_summary(df, ['x'])

    assert summary.columns.tolist() == eda.SUMMARIES
    assert summary.loc['x'].tolist() == [3.0, 1.58, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_add_columns() -> None:
    """
    Tests add_columns within eda.py.
    """
    df = pd.DataFrame({'rushing_epa': [2.0, -1.0], 'carries': [10, 5],
                       'receptions': [2, 0], 'carries_team': [20, 25],
                       'receptions_team': [20, 25],
                       'opp_epa_allowed': [0.1, -0.1],
                       'backup_qb': [True, False]})
    result = eda.add_columns(df)

    assert math.isclose(result.loc[0, 'epa_per_rush'], 0.2)
    assert math.isclose(result.loc[0, 'touch_percentage'], 30.0)
    assert math.isclose(result.loc[0, 'adjusted_epa'], 0.1)
    assert result.loc[0, 'qb_role'] == 'Backup QB'
    assert math.isclose(result.loc[1, 'adjusted_epa'], -0.1)
    assert result.loc[1, 'qb_role'] == 'Starter QB'
    assert 'epa_per_rush' not in df.columns


def test_get_paired_differences() -> None:
    """
    Tests get_paired_differences within eda.py.
    """
    df = pd.DataFrame({
        'team': ['TM1', 'TM1', 'TM1', 'TM2', 'TM3', 'TM3'],
        'season': [2021] * 6,
        'backup_qb': [False, False, True, False, False, True],
        'carries': [10, 20, 5, 12, 8, 7],
        'epa_per_rush': [0.1, 0.3, -0.5, 0.0, 0.3, 0.1]})
    for metric in eda.RQ_METRICS:
        if metric not in df.columns:
            df[metric] = 0.0
    differences = eda.get_paired_differences(df)

    carries = differences[differences['metric'] == 'carries']
    carries = carries.set_index('team')['difference']
    assert carries.to_dict() == {'TM1': 10.0, 'TM3': 1.0}

    epa = differences[differences['metric'] == 'epa_per_rush']
    assert epa['team'].tolist() == ['TM3']
    assert math.isclose(epa['difference'].iloc[0], 0.2)


def main():
    test_load_weekly_stats()
    test_get_starting_qbs()
    test_get_qb_games()
    test_get_starter_rbs()
    test_get_starter_rb_games()

    df = analysis.read_csv_file('data/games.csv')
    combined_df = analysis.get_combined_dataframe(df, analysis.TEAMS)
    test_get_combined_dataframe(combined_df)

    test_missing_summary()
    test_seven_number_summary()
    test_add_columns()
    test_get_paired_differences()

    print('All tests passed!')


if __name__ == '__main__':
    main()
