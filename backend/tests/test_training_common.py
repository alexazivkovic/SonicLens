"""Artist-grouped splits and metrics used by the training scripts (no FMA data needed)."""

import numpy as np
import pandas as pd
import pytest

from training.common import N_FOLDS, TEST_FRACTION, assign_splits, regression_metrics


@pytest.fixture
def artists():
    rng = np.random.default_rng(0)
    # 5000 tracks by 1000 artists with uneven catalogue sizes.
    return pd.Series(rng.zipf(1.5, 5000) % 1000, index=pd.RangeIndex(5000, name="track_id"), name="artist_id")


def test_no_artist_on_both_sides_or_in_two_folds(artists):
    splits = assign_splits(artists)
    test_artists = set(artists[splits.is_test])
    assert test_artists.isdisjoint(set(artists[~splits.is_test]))
    assert (artists[~splits.is_test].groupby(splits.fold[~splits.is_test]).unique().explode().value_counts() == 1).all()
    assert (splits.fold[splits.is_test] == -1).all()
    assert set(splits.fold[~splits.is_test]) == set(range(N_FOLDS))


def test_test_fraction_of_artists_is_about_right(artists):
    unique = pd.Series(artists.unique())
    assert assign_splits(unique).is_test.mean() == pytest.approx(TEST_FRACTION, abs=0.04)


def test_split_is_stable_for_subsets(artists):
    full = assign_splits(artists)
    subset = artists.sample(frac=0.3, random_state=1)
    pd.testing.assert_frame_equal(assign_splits(subset), full.loc[subset.index])


def test_regression_metrics():
    y = np.array([0.1, 0.4, 0.5, 0.9])
    perfect = regression_metrics(y, y)
    assert perfect == {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "spearman": 1.0}
    constant = regression_metrics(y, np.full_like(y, y.mean()))
    assert constant["r2"] == pytest.approx(0.0)
    assert constant["spearman"] == 0.0
