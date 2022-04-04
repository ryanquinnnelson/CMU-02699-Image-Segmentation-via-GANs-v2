import numpy as np
import torch
import pytest

import customized.losses as l


@pytest.fixture
def target1():
    t = torch.tensor([[0., 0., 1., 1., 1.],
                      [0., 0., 0., 1., 1.],
                      [0., 0., 0., 0., 1.],
                      [1., 0., 0., 0., 1.],
                      [1., 1., 0., 0., 1.]])

    return t


@pytest.fixture
def target2():
    t = torch.tensor([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]])

    return t


@pytest.fixture
def target3():
    t = torch.tensor([[1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1.]])

    return t


def test__get_random_segment_position(target1):
    segment_positions = np.nonzero(target1.numpy())
    available_positions = [[segment_positions[0][i], segment_positions[1][i]] for i in range(len(segment_positions[0]))]

    for i in range(1000):
        p = l._get_random_segment_position(segment_positions)
        if p not in available_positions:
            assert False


def test__get_new_anchor__no_existing_anchors(target1):
    segment_positions = np.nonzero(target1.numpy())
    anchor_positions = []

    available_positions = [[segment_positions[0][i], segment_positions[1][i]] for i in range(len(segment_positions[0]))]
    anchor_position = l._get_new_anchor(segment_positions, anchor_positions)
    if anchor_position not in available_positions:
        assert False


def test__get_new_anchor__existing_anchors(target1):
    segment_positions = np.nonzero(target1.numpy())
    anchor_positions = [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [3, 0], [3, 4], [4, 0], [4, 1]]

    anchor_position = l._get_new_anchor(segment_positions, anchor_positions)
    if anchor_position != [4, 4]:
        assert False


def test__get_new_anchor__no_available_anchors(target1):
    segment_positions = np.nonzero(target1.numpy())
    anchor_positions = [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [3, 0], [3, 4], [4, 0], [4, 1], [4, 4]]

    with pytest.raises(ValueError):
        l._get_new_anchor(segment_positions, anchor_positions)


def test__position_is_in_image():
    n_rows = 5
    n_cols = 5
    pos_false = [-1, -1]
    assert l._position_is_in_image(pos_false, n_rows, n_cols) is False

    pos_true = [0, 0]
    assert l._position_is_in_image(pos_true, n_rows, n_cols)


def test__is_background_pixel(target1):
    segment_positions = np.nonzero(target1.numpy())
    pos_false = [4, 0]
    assert l._is_background_pixel(pos_false, segment_positions) is False
    pos_true = [0, 0]
    assert l._is_background_pixel(pos_true, segment_positions)


def test__get_positions_at_radius():
    anchor_position = [3, 3]

    # 0
    radius = 0
    actual_positions = l._get_positions_at_radius(anchor_position, radius)
    assert len(actual_positions) == 0

    # 1
    radius = 1
    expected_positions = [[2, 2], [2, 3], [2, 4],
                          [3, 2], [3, 4],
                          [4, 2], [4, 3], [4, 4]]
    actual_positions = l._get_positions_at_radius(anchor_position, radius)
    for each in expected_positions:
        assert each in actual_positions

    # 2
    radius = 2
    expected_positions = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                          [2, 1], [2, 5],
                          [3, 1], [3, 5],
                          [4, 1], [4, 5],
                          [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]]
    actual_positions = l._get_positions_at_radius(anchor_position, radius)
    for each in expected_positions:
        assert each in actual_positions


def test__get_new_negative__no_available(target3):
    segment_positions = np.nonzero(target3.numpy())
    n_rows = target3.shape[0]
    n_cols = target3.shape[1]
    anchor_position = [4, 0]

    with pytest.raises(ValueError):
        l._get_new_negative(segment_positions, anchor_position, n_rows, n_cols)


def test__get_new_negative(target1):
    segment_positions = np.nonzero(target1.numpy())
    n_rows = target1.shape[0]
    n_cols = target1.shape[1]
    anchor_position = [4, 0]

    expected = [3, 1]
    actual = l._get_new_negative(segment_positions, anchor_position, n_rows, n_cols)
    assert actual == expected


def test__get_new_positive__no_available(target2):
    segment_positions = np.nonzero(target2.numpy())
    anchor_position = [0, 0]

    with pytest.raises(ValueError):
        l._get_new_positive(segment_positions, anchor_position)


def test__get_new_positive__no_available(target1):
    segment_positions = np.nonzero(target1.numpy())
    anchor_position = [4, 0]
    available_positions = [[segment_positions[0][i], segment_positions[1][i]] for i in range(len(segment_positions[0]))]

    for i in range(1000):
        positive_position = l._get_new_positive(segment_positions, anchor_position)
        assert positive_position != anchor_position
        assert positive_position in available_positions
