import random
import logging

import numpy as np
import torch


def _get_random_segment_position(segment_positions):
    # select random index in positions array
    n_segment_pixels = len(segment_positions)
    idx = np.random.randint(n_segment_pixels)

    # get (i,j) values for the index into the positions array
    anchor_pos = segment_positions[idx].tolist()

    return anchor_pos


def _get_new_anchor(segment_positions, anchor_positions):
    # check if all possible anchor positions have already been selected
    n_segment_pixels = len(segment_positions)
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
    # get all segment row indices where first col matches p[0]
    matches = segment_positions[:, 0] == p[0]
    idx = torch.nonzero(matches)

    # from only the rows which match the first col, get all rows where second col matches p[1]
    matches = segment_positions[idx, 1] == p[1]
    idx = torch.nonzero(matches)

    # if there is a row that matches p[1], this position is segment, not background
    is_background_pixel = len(idx) == 0
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
    n_segment_pixels = len(segment_positions)
    if n_segment_pixels < 2:
        raise ValueError('Target does not have enough segment pixels to choose a positive sample.')

    # choose any segment pixel except anchor as positive
    positive_position = _get_random_segment_position(segment_positions)
    while positive_position == anchor_position:
        positive_position = _get_random_segment_position(segment_positions)

    return positive_position


# TODO: find a way to get all triplet positions simultaneously
def _get_triplet_positions(target, n_triplets):
    # for storing triplets
    anchor_positions = []
    positive_positions = []
    negative_positions = []

    # extract dimensions of target image for use in position validation
    img_rows = target.shape[0]
    img_cols = target.shape[1]

    # get indices of all nonzero positions (segment pixels)
    segment_positions = torch.nonzero(target)

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

    return anchor_positions, negative_positions, positive_positions


def _get_triplet_values(output, triplet_positions):
    # unpack
    anchor_positions, negative_positions, positive_positions = triplet_positions

    anchor_values = []
    negative_values = []
    positive_values = []

    for j, k in anchor_positions:
        output_anchor = output[j, k]
        anchor_values.append(output_anchor)

    for j, k in negative_positions:
        output_negative = output[j, k]
        negative_values.append(output_negative)

    for j, k in positive_positions:
        output_positive = output[j, k]
        positive_values.append(output_positive)

    return anchor_values, negative_values, positive_values


# TODO: vectorize so anchors, negatives, and positives are extracted at the same time
def _get_triplet_values2(output, triplet_positions):
    # assumes output is (H,W) tensor
    # unpack
    anchor_positions, negative_positions, positive_positions = triplet_positions

    # extract positions simultaneously
    anchor_values = output[np.array(anchor_positions).T]
    negative_values = output[np.array(negative_positions).T]
    positive_values = output[np.array(positive_positions).T]

    return anchor_values, negative_values, positive_values


def _get_triplet_values3(output, triplet_positions):
    # assumes output is (H,W) tensor
    # unpack
    anchor_positions, negative_positions, positive_positions = triplet_positions
    n_positions = len(anchor_positions)

    # combine positions into single list so we can interact only once with output tensor
    positions = anchor_positions + negative_positions + positive_positions

    # extract positions simultaneously
    values = output[np.array(positions).T]

    # split into anchor, negative, positive values
    anchor_values = values[:n_positions]
    negative_values = values[n_positions:n_positions * 2]
    positive_values = values[n_positions * 2:]

    return anchor_values, negative_values, positive_values


# TODO: vectorize calculation to improve efficiency
def _calculate_loss(triplet_values, margin=0.2):
    # expects lists of (1,) shape tensors as input
    total_loss = 0.0

    # unpack
    anchor_values, negative_values, positive_values = triplet_values
    n_triplets = len(anchor_values)

    for i in range(n_triplets):
        anchor = torch.unsqueeze(anchor_values[i], dim=0)  # cdist requires 2D tensors
        negative = torch.unsqueeze(negative_values[i], dim=0)
        positive = torch.unsqueeze(positive_values[i], dim=0)

        a = torch.cdist(anchor, positive, p=2.0)
        b = torch.cdist(anchor, negative, p=2.0)

        option1 = a - b + margin
        option2 = 0.0

        loss = max(option1, option2)
        total_loss += loss

    return total_loss


# TODO: consider optimizing triplet_values to not include anchor since it isn't needed in calculation
def _calculate_loss2(triplet_values, margin=0.2):
    # expects lists of (n,1) shape tensors as input, where n is the number of triplet pairs

    # calculating euclidean distance on scalar values means we don't need to square then take square root
    # (anchor - positive) - (anchor - negative) + margin === negative - positive + margin

    # unpack
    anchor_values, negative_values, positive_values = triplet_values
    option1 = negative_values - positive_values + margin
    option2 = torch.zeros(option1.shape)

    # for each row, if option1 is nonnegative, select that option, otherwise select 0.0
    selection = torch.where(option1 >= 0, option1, option2)

    # sum to get total loss
    loss = torch.sum(selection)
    return loss


class TripletLoss:

    # TODO: vectorize loss calculates on all target,output pairs at the same time
    def calculate_loss(self, outputs, targets, margin=0.2, n_triplets=1):
        total_loss = 0.0
        n_samples = outputs.shape[0]

        # for each sample in the batch output, calculate loss
        for i in range(n_samples):
            # calculate loss for one target,output pair at a time
            target = targets[i]
            output = outputs[i]

            # choose positions for all triplets
            triplet_positions = _get_triplet_positions(target, n_triplets)
            # logging.info(f'triplet_positions:{triplet_positions}')
            # get values for all triplets
            triplet_values = _get_triplet_values3(output, triplet_positions)
            # logging.info(f'triplet_values:{triplet_values}')
            # calculate loss for triplets
            loss = _calculate_loss2(triplet_values, margin)
            # logging.info(f'loss:{loss}')
            # add to running total
            total_loss += loss

        return total_loss
