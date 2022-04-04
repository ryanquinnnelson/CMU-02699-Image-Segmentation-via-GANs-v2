import random

import numpy as np
import torch


def _get_random_segment_position(segment_positions):
    # select random index in positions array
    n_segment_pixels = len(segment_positions[0])
    idx = np.random.randint(n_segment_pixels)

    # get (i,j) values for the index into the positions array
    i = segment_positions[0][idx]
    j = segment_positions[1][idx]
    anchor_pos = [i, j]

    return anchor_pos


def _get_new_anchor(segment_positions, anchor_positions):
    # check if all possible anchor positions have already been selected
    n_segment_pixels = len(segment_positions[0])
    if len(anchor_positions) == n_segment_pixels:
        raise ValueError('All possible anchor positions have been already selected.')

    # select any segment pixel at random which has not yet been selected
    anchor_position = _get_random_segment_position(segment_positions)
    while anchor_position in anchor_positions:
        anchor_position = _get_random_segment_position(segment_positions)

    return anchor_position


def _position_is_in_image(p, n_rows, n_cols):
    is_valid_row = 0 <= p[0] < n_rows
    is_valid_col = 0 <= p[1] < n_cols

    position_is_in_image = is_valid_row and is_valid_col
    return position_is_in_image


def _is_background_pixel(p, segment_positions):
    is_background_pixel = True

    # get all segment row positions that match pos[0]
    matches_row_idx = np.where(segment_positions[0] == p[0])[0]  # extract array from np.where results

    # check all row position matches to see if matching column position exists
    for i in matches_row_idx:
        if segment_positions[1][i] == p[1]:
            is_background_pixel = False
            break  # stop searching

    return is_background_pixel


def _get_positions_at_radius(anchor_position, r):

    if r <= 0:
        return []

    # get (i,j) position of upper left corner at given radius from anchor even if invalid
    ul_i = anchor_position[0] - r
    ul_j = anchor_position[1] - r

    # set the length of the square formed by the current radius away from anchor
    square_length = 3 + 2 * (r - 1)

    # calc all positions that are radius pixels away from the anchor
    # include invalid positions
    radius_positions = []
    for i_offset in range(square_length):
        if i_offset == 0 or i_offset == square_length - 1:  # first row or last row

            # every position in first or last row is radius pixels from anchor
            for j_offset in range(square_length):
                pos = [ul_i + i_offset, ul_j + j_offset]
                radius_positions.append(pos)

        else:  # any row except first or last

            # only first and last columns are radius pixels from anchor
            for j_offset in [0, square_length - 1]:
                pos = [ul_i + i_offset, ul_j + j_offset]
                radius_positions.append(pos)

    return radius_positions


def _get_new_negative(segment_positions, anchor_position, n_rows, n_cols):
    # check for background pixels using increasing radius approach
    # closest background pixel to anchor is a hard negative

    negative_positions = None
    radius = 1  # start search 1 pixel away from anchor

    while negative_positions is None:

        # sanity check
        if radius > n_rows:
            raise ValueError('Target does not have any background pixels.')

        radius_positions = _get_positions_at_radius(anchor_position, radius)

        # check all radius positions for candidates that are valid and background
        candidates = []
        for pos in radius_positions:
            is_valid = _position_is_in_image(pos, n_rows, n_cols)
            if is_valid:
                is_background = _is_background_pixel(pos, segment_positions)
                if is_background:
                    candidates.append(pos)

        # take action based on whether any candidates have been found
        if len(candidates) > 0:
            # select single candidate at random
            negative_positions = random.choice(candidates)
        else:
            # no candidates exist at current radius
            # increase radius and search again
            radius += 1

    return negative_positions


def _get_new_positive(segment_positions, anchor_position):
    # check if positive sample is available
    n_segment_pixels = len(segment_positions[0])
    if n_segment_pixels < 2:
        raise ValueError('Target does not have enough segment pixels to choose a positive sample.')

    # choose any segment pixel except anchor as positive
    positive_position = _get_random_segment_position(segment_positions)
    while positive_position == anchor_position:
        positive_position = _get_random_segment_position(segment_positions)

    return positive_position


def get_triplet_positions(target, n_triplets):
    # for storing triplets
    anchor_positions = []
    positive_positions = []
    negative_positions = []

    # extract dimensions of target image for use in position validation
    img_rows = target.shape[0]
    img_cols = target.shape[1]

    # get indices of all nonzero positions (segment pixels)
    segment_positions = np.nonzero(target.numpy())

    # generate sets of triplets
    for i in range(n_triplets):
        # choose as anchor a segment pixel which has not yet been chosen to be an anchor
        anchor_position = _get_new_anchor(segment_positions, anchor_positions)

        # choose closest background pixel to be hard negative for this anchor
        negative_position = _get_new_negative(segment_positions, anchor_position, img_rows, img_cols)

        # choose as positive a segment pixel which is not the current anchor
        positive_position = _get_new_positive(segment_positions, anchor_position)

        # save triplet
        anchor_positions.append(anchor_position)
        negative_positions.append(negative_position)
        positive_positions.append(positive_position)

    return np.array(anchor_positions), np.array(negative_positions), np.array(positive_positions)
