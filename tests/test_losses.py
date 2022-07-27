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


@pytest.fixture
def targets():
    t = torch.tensor([[[0., 0., 1., 1., 1.],
                       [0., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 1.],
                       [1., 1., 0., 0., 1.]],

                      [[0., 0., 1., 1., 1.],
                       [0., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 1.],
                       [1., 1., 0., 0., 1.]],

                      [[0., 0., 1., 1., 1.],
                       [0., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 1.],
                       [1., 1., 0., 0., 1.]]])

    return t


@pytest.fixture
def outputs_1d():
    # shape: (3,5,5)
    # batch of 3
    # each output has a single dimension
    o = torch.tensor([[[0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
                       [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
                       [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
                       [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
                       [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]],

                      [[0.2977, 0.4399, 0.4750, 0.8097, 0.7573],
                       [0.1771, 0.2021, 0.4243, 0.5265, 0.2079],
                       [0.9319, 0.9046, 0.2385, 0.4292, 0.1066],
                       [0.5421, 0.8858, 0.3479, 0.0413, 0.2431],
                       [0.2422, 0.2530, 0.5502, 0.7505, 0.4435]],

                      [[0.8230, 0.9267, 0.9465, 0.9285, 0.5393],
                       [0.3142, 0.2686, 0.0248, 0.8731, 0.8203],
                       [0.4774, 0.1330, 0.5620, 0.5715, 0.7523],
                       [0.9411, 0.5292, 0.0644, 0.1066, 0.1095],
                       [0.3790, 0.8215, 0.2276, 0.6534, 0.8838]]])
    return o


@pytest.fixture
def outputs_2d():
    # shape: (3,2,5,5)
    # batch of 3
    # each output has two dimensions
    o = torch.tensor([[[[0.3243, 0.8990, 0.0921, 0.2546, 0.9084],
                        [0.5298, 0.9023, 0.2922, 0.7029, 0.0408],
                        [0.3300, 0.9085, 0.5602, 0.9118, 0.9161],
                        [0.3091, 0.5567, 0.9685, 0.8120, 0.1874],
                        [0.3743, 0.9884, 0.8920, 0.4025, 0.9539]],

                       [[0.3390, 0.1962, 0.3514, 0.1017, 0.7659],
                        [0.8812, 0.9223, 0.7047, 0.0409, 0.2640],
                        [0.6041, 0.9155, 0.6221, 0.8875, 0.5795],
                        [0.4273, 0.3435, 0.9991, 0.3975, 0.6576],
                        [0.3570, 0.8038, 0.5411, 0.5832, 0.6003]]],

                      [[[0.3664, 0.0723, 0.8448, 0.4502, 0.1831],
                        [0.7847, 0.5026, 0.4738, 0.9825, 0.0244],
                        [0.8613, 0.9222, 0.6209, 0.7973, 0.3739],
                        [0.0735, 0.1107, 0.6716, 0.2892, 0.0741],
                        [0.3668, 0.6573, 0.3271, 0.8849, 0.0231]],

                       [[0.5387, 0.6941, 0.0724, 0.7723, 0.5286],
                        [0.5985, 0.2464, 0.9890, 0.9410, 0.0336],
                        [0.5769, 0.9429, 0.7046, 0.8622, 0.8496],
                        [0.2527, 0.1093, 0.8799, 0.8268, 0.3146],
                        [0.8545, 0.2649, 0.1595, 0.0731, 0.8068]]],

                      [[[0.8836, 0.2200, 0.3727, 0.3172, 0.1490],
                        [0.2958, 0.3237, 0.4373, 0.1085, 0.0594],
                        [0.4614, 0.5124, 0.3648, 0.4952, 0.7492],
                        [0.2285, 0.9710, 0.7652, 0.3734, 0.9514],
                        [0.1786, 0.9691, 0.3321, 0.5972, 0.2977]],

                       [[0.8821, 0.4630, 0.6233, 0.5851, 0.1906],
                        [0.1751, 0.8617, 0.9453, 0.3574, 0.8914],
                        [0.0615, 0.5652, 0.7769, 0.9554, 0.1324],
                        [0.8907, 0.5693, 0.7974, 0.4085, 0.7908],
                        [0.5809, 0.9849, 0.5275, 0.3371, 0.7537]]]])
    return o


def test__get_random_segment_position(target1):
    segment_positions = torch.nonzero(target1)
    available_positions = segment_positions.tolist()

    for i in range(1000):
        p = l._get_random_segment_position(segment_positions)
        if p not in available_positions:
            assert False


def test__get_new_anchor__no_existing_anchors(target1):
    segment_positions = torch.nonzero(target1)
    anchor_positions = []

    available_positions = segment_positions.tolist()
    anchor_position = l._get_new_anchor(segment_positions, anchor_positions)
    if anchor_position not in available_positions:
        assert False


def test__get_new_anchor__existing_anchors(target1):
    segment_positions = torch.nonzero(target1)
    anchor_positions = [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [3, 0], [3, 4], [4, 0], [4, 1]]

    anchor_position = l._get_new_anchor(segment_positions, anchor_positions)
    if anchor_position != [4, 4]:
        assert False


def test__get_new_anchor__no_available_anchors(target1):
    segment_positions = torch.nonzero(target1)
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
    segment_positions = torch.nonzero(target1)
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
    segment_positions = torch.nonzero(target3)
    n_rows = target3.shape[0]
    n_cols = target3.shape[1]
    anchor_position = [4, 0]

    with pytest.raises(ValueError):
        l._get_new_negative(segment_positions, anchor_position, n_rows, n_cols)


def test__get_new_negative(target1):
    segment_positions = torch.nonzero(target1)
    n_rows = target1.shape[0]
    n_cols = target1.shape[1]
    anchor_position = [4, 0]

    expected = [3, 1]
    actual = l._get_new_negative(segment_positions, anchor_position, n_rows, n_cols)
    assert actual == expected


def test__get_new_positive__no_available(target2):
    segment_positions = torch.nonzero(target2)
    anchor_position = [0, 0]

    with pytest.raises(ValueError):
        l._get_new_positive(segment_positions, anchor_position)


def test__get_new_positive(target1):
    segment_positions = torch.nonzero(target1)
    anchor_position = [4, 0]
    available_positions = segment_positions.tolist()

    for i in range(1000):
        positive_position = l._get_new_positive(segment_positions, anchor_position)
        assert positive_position != anchor_position
        assert positive_position in available_positions


def test__get_triplet_positions(target1):
    # 0
    n_triplets = 0
    actual = l._get_triplet_positions(target1, n_triplets)
    assert len(actual[0]) == 0
    assert len(actual[1]) == 0
    assert len(actual[2]) == 0

    # 1
    n_triplets = 1
    actual_anchors, actual_negatives, actual_positives = l._get_triplet_positions(target1, n_triplets)
    assert len(actual_anchors) == 1
    assert len(actual_negatives) == 1
    assert len(actual_positives) == 1

    # 2
    n_triplets = 2
    actual_anchors, actual_negatives, actual_positives = l._get_triplet_positions(target1, n_triplets)
    assert len(actual_anchors) == 2
    assert len(actual_negatives) == 2
    assert len(actual_positives) == 2


def test__get_triplet_values__1d_one_index(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0]]
    negative_positions = [[3, 1]]
    positive_positions = [[4, 4]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = [torch.tensor([0.2775])]
    expected_negative_values = [torch.tensor([0.8106])]
    expected_positive_values = [torch.tensor([0.3095])]

    actual_triplet_values = l._get_triplet_values(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__get_triplet_values__1d_two_indices(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0], [3, 3]]
    negative_positions = [[3, 1], [3, 2]]
    positive_positions = [[4, 4], [0, 0]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = [torch.tensor([0.2775]), torch.tensor([0.4664])]
    expected_negative_values = [torch.tensor([0.8106]), torch.tensor([0.0950])]
    expected_positive_values = [torch.tensor([0.3095]), torch.tensor([0.3755])]

    actual_triplet_values = l._get_triplet_values(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__get_triplet_values2__1d_one_index(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0]]
    negative_positions = [[3, 1]]
    positive_positions = [[4, 4]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = torch.tensor([0.2775])
    expected_negative_values = torch.tensor([0.8106])
    expected_positive_values = torch.tensor([0.3095])

    actual_triplet_values = l._get_triplet_values2(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__get_triplet_values2__1d_two_indices(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0], [3, 3]]
    negative_positions = [[3, 1], [3, 2]]
    positive_positions = [[4, 4], [0, 0]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = torch.tensor([0.2775, 0.4664])
    expected_negative_values = torch.tensor([0.8106, 0.0950])
    expected_positive_values = torch.tensor([0.3095, 0.3755])

    actual_triplet_values = l._get_triplet_values2(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__get_triplet_values3__1d_one_index(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0]]
    negative_positions = [[3, 1]]
    positive_positions = [[4, 4]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = torch.tensor([0.2775])
    expected_negative_values = torch.tensor([0.8106])
    expected_positive_values = torch.tensor([0.3095])

    actual_triplet_values = l._get_triplet_values3(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__get_triplet_values3__1d_two_indices(outputs_1d):
    # print(outputs_1d.shape)
    output = outputs_1d[0]
    # print(output.shape)

    # each output image has two dimensions
    anchor_positions = [[4, 0], [3, 3]]
    negative_positions = [[3, 1], [3, 2]]
    positive_positions = [[4, 4], [0, 0]]
    triplet_positions = anchor_positions, negative_positions, positive_positions

    '''
    [[  [0.3755, 0.0455, 0.0532, 0.8096, 0.0012],
        [0.8413, 0.6249, 0.7906, 0.3437, 0.7179],
        [0.6032, 0.0878, 0.8019, 0.3762, 0.4233],
        [0.8584, 0.8106, 0.0950, 0.4664, 0.9527],
        [0.2775, 0.6072, 0.2494, 0.4346, 0.3095]]]
    '''

    expected_anchor_values = torch.tensor([0.2775, 0.4664])
    expected_negative_values = torch.tensor([0.8106, 0.0950])
    expected_positive_values = torch.tensor([0.3095, 0.3755])

    actual_triplet_values = l._get_triplet_values3(output, triplet_positions)

    # unpack
    actual_anchor_values = actual_triplet_values[0]
    actual_negative_values = actual_triplet_values[1]
    actual_positive_values = actual_triplet_values[2]

    for i in range(len(actual_anchor_values)):
        np.testing.assert_equal(actual_anchor_values[i].numpy(), expected_anchor_values[i].numpy())

    for i in range(len(actual_negative_values)):
        np.testing.assert_equal(actual_negative_values[i].numpy(), expected_negative_values[i].numpy())

    for i in range(len(actual_positive_values)):
        np.testing.assert_equal(actual_positive_values[i].numpy(), expected_positive_values[i].numpy())


def test__calculate_loss__two_triplet_pairs():
    anchor_values = [torch.tensor([0.0]),torch.tensor([0.0])]
    negative_values = [torch.tensor([2.0]), torch.tensor([1.0])]
    positive_values = [torch.tensor([0.0]), torch.tensor([2.0])]
    triplet_values = anchor_values, negative_values, positive_values

    expected = 1.2
    actual = l._calculate_loss(triplet_values)
    assert actual == pytest.approx(expected, rel=1e-3)


def test__calculate_loss2__two_triplet_pairs():
    anchor_values = torch.tensor([0.0, 0.0])
    negative_values = torch.tensor([2.0, 1.0])
    positive_values = torch.tensor([0.0, 2.0])
    triplet_values = anchor_values, negative_values, positive_values

    expected = 1.2
    actual = l._calculate_loss2(triplet_values)
    assert actual == pytest.approx(expected, rel=1e-3)


def test_TripletLoss_calculate_loss(outputs_1d, targets):
    triplet_loss = l.TripletLoss()

    # just check that the function works
    triplet_loss.calculate_loss(outputs_1d, targets)
